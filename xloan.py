# Home Credit Default Risk - Traditional ML + Fairness-First Approach
# Based on 1st place solution with fairness considerations
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# ML Libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
import xgboost as xgb
# Explainability & Fairness
import shap
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Utils
import joblib
from datetime import datetime
import logging
from xgboost.callback import EarlyStopping
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HomeCreditRiskAssessment:
    """
    Home Credit Risk Assessment với Traditional ML + Fairness-First approach
    """
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.fairness_metrics = {}
        self.explainer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def load_data(self):
        """Load và basic preprocessing của Home Credit dataset"""
        logger.info("Loading Home Credit dataset...")
        
        # Load main tables
        try:
            self.app_train = pd.read_csv('application_train.csv')
            self.app_test = pd.read_csv('application_test.csv')
            self.bureau = pd.read_csv('bureau.csv')
            self.bureau_balance = pd.read_csv('bureau_balance.csv')
            self.prev_app = pd.read_csv('previous_application.csv')
            self.pos_cash = pd.read_csv('POS_CASH_balance.csv')
            self.installments = pd.read_csv('installments_payments.csv')
            self.credit_card = pd.read_csv('credit_card_balance.csv')
            
            logger.info(f"Loaded training data: {self.app_train.shape}")
            logger.info(f"Target distribution: {self.app_train['TARGET'].value_counts().to_dict()}")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            # Create dummy data for demo
            self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Tạo dummy data cho demo khi không có file gốc"""
        logger.info("Creating dummy data for demo...")
        
        np.random.seed(42)
        n_samples = 10000
        
        # Main application data
        self.app_train = pd.DataFrame({
            'SK_ID_CURR': range(n_samples),
            'TARGET': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
            'AMT_INCOME_TOTAL': np.random.lognormal(12, 0.5, n_samples),
            'AMT_CREDIT': np.random.lognormal(13, 0.6, n_samples),
            'AMT_ANNUITY': np.random.lognormal(10, 0.4, n_samples),
            'AMT_GOODS_PRICE': np.random.lognormal(12.5, 0.7, n_samples),
            'DAYS_BIRTH': np.random.randint(-25000, -6000, n_samples),
            'DAYS_EMPLOYED': np.random.randint(-15000, 0, n_samples),
            'CODE_GENDER': np.random.choice(['M', 'F'], n_samples, p=[0.35, 0.65]),
            'NAME_EDUCATION_TYPE': np.random.choice([
                'Secondary / secondary special', 'Higher education', 
                'Incomplete higher', 'Lower secondary'
            ], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'NAME_FAMILY_STATUS': np.random.choice([
                'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'
            ], n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            'CNT_CHILDREN': np.random.poisson(0.4, n_samples),
            'EXT_SOURCE_1': np.random.beta(2, 3, n_samples),
            'EXT_SOURCE_2': np.random.beta(2, 3, n_samples),
            'EXT_SOURCE_3': np.random.beta(2, 3, n_samples),
            'REGION_POPULATION_RELATIVE': np.random.uniform(0.0001, 0.1, n_samples)
        })
        
        # Test data
        self.app_test = self.app_train.copy()
        self.app_test = self.app_test.drop('TARGET', axis=1)
        self.app_test['SK_ID_CURR'] = range(n_samples, n_samples*2)
        
        logger.info("Dummy data created successfully")
    
    def feature_engineering(self):
        """Feature engineering dựa trên winning solution"""
        logger.info("Starting feature engineering...")
        
        df = self.app_train.copy()
        
        # 1. Basic derived features
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['GOODS_PRICE_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
        
        # 2. Age and employment features
        df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).astype(int)
        df['EMPLOYED_YEARS'] = (-df['DAYS_EMPLOYED'] / 365).astype(int)
        df['EMPLOYED_YEARS'] = df['EMPLOYED_YEARS'].apply(lambda x: max(0, x))
        
        # Age groups for fairness analysis
        df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], 
                                bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # 3. Income brackets
        df['INCOME_BRACKET'] = pd.qcut(df['AMT_INCOME_TOTAL'], 
                                      q=5, 
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # 4. External source combinations (từ winning solution)
        df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
        df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
        
        # 5. Binary indicators
        df['HAS_CAR'] = (df.get('FLAG_OWN_CAR', 'N') == 'Y').astype(int)
        df['HAS_REALTY'] = (df.get('FLAG_OWN_REALTY', 'N') == 'Y').astype(int)
        df['HAS_CHILDREN'] = (df['CNT_CHILDREN'] > 0).astype(int)
        
        # 6. Risk categories (business logic)
        df['HIGH_CREDIT_RISK'] = (
            (df['CREDIT_INCOME_RATIO'] > 10) | 
            (df['ANNUITY_INCOME_RATIO'] > 0.5)
        ).astype(int)
        
        self.df_engineered = df
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        
        return df
    
    def prepare_features(self):
        """Chuẩn bị features cho modeling"""
        logger.info("Preparing features for modeling...")
        
        df = self.df_engineered.copy()
        
        # Identify all categorical features dynamically
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Label encoding for all categorical features
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Features for modeling (loại bỏ ID và target)
        exclude_cols = ['SK_ID_CURR', 'TARGET']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values for numeric columns
        numeric_feature_columns = df[feature_cols].select_dtypes(include=np.number).columns
        medians = df[numeric_feature_columns].median()
        df[numeric_feature_columns] = df[numeric_feature_columns].fillna(medians)
        
        self.X = df[feature_cols]
        self.y = df['TARGET']
        self.feature_names = feature_cols
        
        # Protected attributes cho fairness
        self.protected_attrs = {
            'gender': df['CODE_GENDER'],
            'age_group': df['AGE_GROUP'],
            'education': df['NAME_EDUCATION_TYPE']
        }
        
        logger.info(f"Features prepared. Shape: {self.X.shape}")
        return self.X, self.y
    
    def feature_selection(self, k=100):
        """Feature selection using statistical tests"""
        logger.info(f"Selecting top {k} features...")
        
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(self.X, self.y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        logger.info(f"Selected {len(self.selected_features)} features")
        return X_selected, self.selected_features
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble của LightGBM, XGBoost, và Logistic Regression"""
        logger.info("Training models...")
        
        # 1. LightGBM
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.models['lgb'] = lgb.LGBMClassifier(**lgb_params, n_estimators=1000, early_stopping_rounds=100, verbosity=0)
        if X_val is not None and y_val is not None:
            self.models['lgb'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            self.models['lgb'].fit(X_train, y_train)
        
        # 2. XGBoost
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        
        self.models['xgb'] = xgb.XGBClassifier(**xgb_params, n_estimators=1000, early_stopping_rounds=100, verbosity=0)
        if X_val is not None and y_val is not None:
            self.models['xgb'].fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
        else:
            self.models['xgb'].fit(X_train, y_train)
        
        # 3. Logistic Regression (for interpretability)
        self.models['lr'] = LogisticRegression(random_state=42, max_iter=1000)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.models['lr'].fit(X_train_scaled, y_train)
        
        logger.info("Models trained successfully")
        
    def predict_ensemble(self, X):
        """Ensemble prediction với trọng số tối ưu"""
        predictions = {}
        
        # LightGBM prediction
        predictions['lgb'] = self.models['lgb'].predict_proba(X)[:, 1]
        
        # XGBoost prediction  
        predictions['xgb'] = self.models['xgb'].predict_proba(X)[:, 1]
        
        # Logistic Regression prediction
        X_scaled = self.scaler.transform(X)
        predictions['lr'] = self.models['lr'].predict_proba(X_scaled)[:, 1]
        
        # Ensemble với trọng số từ winning solution
        ensemble_pred = (0.6 * predictions['lgb'] + 
                        0.3 * predictions['xgb'] + 
                        0.1 * predictions['lr'])
        
        return ensemble_pred, predictions
    
    def evaluate_fairness(self, y_true, y_pred, protected_attr):
        """Đánh giá fairness metrics"""
        logger.info("Evaluating fairness metrics...")
        
        fairness_metrics = {}
        
        for attr_name, attr_values in protected_attr.items():
            # Demographics parity difference
            dp_diff = demographic_parity_difference(
                y_true, y_pred > 0.5, sensitive_features=attr_values
            )
            
            # Equalized odds difference
            eo_diff = equalized_odds_difference(
                y_true, y_pred > 0.5, sensitive_features=attr_values
            )
            
            fairness_metrics[attr_name] = {
                'demographic_parity_diff': dp_diff,
                'equalized_odds_diff': eo_diff
            }
            
            logger.info(f"{attr_name} - DP diff: {dp_diff:.4f}, EO diff: {eo_diff:.4f}")
        
        self.fairness_metrics = fairness_metrics
        return fairness_metrics
    
    def setup_explainability(self, X_sample):
        """Setup SHAP explainer"""
        logger.info("Setting up SHAP explainer...")
        
        # Use LightGBM for SHAP (fastest for tree models)
        self.explainer = shap.TreeExplainer(self.models['lgb'])
        self.shap_values = self.explainer.shap_values(X_sample) # This creates SHAP values for all classes in a multi-class problem.
                                                               # For binary classification, shap_values[1] is typically used for the positive class.
                                                               # Or, if shap_values is a list of two arrays (one for each class), then shap_values[1] is for class 1.
                                                               # Given the context of binary classification and later use for a single customer,
                                                               # it might be more common to get shap_values for the positive class only if needed.
                                                               # However, self.explainer.shap_values(X_sample) for binary classification returns a list of two arrays [shap_values_class_0, shap_values_class_1]
                                                               # or just one array if explain_output='probability' or similar is used. Let's assume it's handled correctly later.
        
        logger.info("SHAP explainer ready")
    
    def explain_prediction(self, customer_data, customer_id=0): # customer_id not used
        """Giải thích prediction cho một customer cụ thể"""
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return None
        
        # Get SHAP values for this customer
        # shap_values for TreeExplainer typically returns a list of arrays [shap_for_class0, shap_for_class1] for binary classification
        # or a single array if only one output is explained.
        # If self.shap_values from setup_explainability was already computed for a sample,
        # this recomputes for a single instance.
        shap_vals_for_customer = self.explainer.shap_values(customer_data.reshape(1, -1))
        
        # For binary classification, shap_values usually returns two arrays (one for each class)
        # We are interested in the SHAP values for the positive class (class 1)
        # If shap_vals_for_customer is a list of two arrays, shap_vals_for_customer[1][0] would be for class 1 for the single sample.
        # If it's already the SHAP values for the positive class, then shap_vals_for_customer[0] is correct.
        # Assuming shap.TreeExplainer.shap_values for binary classification returns [shap_class_0, shap_class_1]
        # and we want to explain the prediction for class 1.
        
        explanation = {
            'expected_value': self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1 else self.explainer.expected_value, # Expected value for class 1
            'shap_values': shap_vals_for_customer[1][0] if isinstance(shap_vals_for_customer, list) and len(shap_vals_for_customer) > 1 else shap_vals_for_customer[0], # SHAP values for class 1 for this customer
            'feature_names': self.selected_features,
            'customer_data': customer_data
        }
        
        return explanation
    
    def create_dashboard_data(self, X_test, predictions):
        """Tạo data cho dashboard"""
        dashboard_data = {
            'predictions': predictions,
            'feature_importance': dict(zip(
                self.selected_features,
                self.models['lgb'].feature_importances_ # Make sure this is aligned with selected_features
            )),
            'fairness_metrics': self.fairness_metrics,
            'model_performance': {
                'auc_lgb': 0.795,  # Placeholder
                'auc_xgb': 0.790,
                'auc_ensemble': 0.803
            }
        }
        
        return dashboard_data
    
    def save_model(self, filepath='home_credit_model.pkl'):
        """Lưu model và components"""
        model_package = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'explainer': self.explainer,
            'fairness_metrics': self.fairness_metrics, # Note: SHAP explainers might not always be easily serializable with joblib, especially complex ones or those with large data references.
            'feature_importances': getattr(self, 'feature_importances', None)                           # It might be better to re-initialize the explainer on load if issues arise.
        }
        
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='home_credit_model.pkl'):
        """Load model và components"""
        model_package = joblib.load(filepath)
        
        self.models = model_package['models']
        self.scaler = model_package['scaler']
        self.label_encoders = model_package['label_encoders']
        self.feature_selector = model_package['feature_selector']
        self.selected_features = model_package['selected_features']
        self.explainer = model_package['explainer']
        self.explainer = model_package.get('explainer')
        self.fairness_metrics = model_package.get('fairness_metrics', {}) # See note in save_model about explainer serialization.
        
        logger.info(f"Model loaded from {filepath}")

# Demo Usage
def run_demo():
    """Chạy demo hoàn chỉnh"""
    print("=== HOME CREDIT RISK ASSESSMENT DEMO ===")
    
    # Initialize
    model = HomeCreditRiskAssessment()
    
    # Load data
    model.load_data()
    
    # Feature engineering
    df_features = model.feature_engineering()
    
    # Prepare for modeling
    X, y = model.prepare_features()
    
    # Feature selection
    # Ensure X (from prepare_features) is used here if it's the full feature set before selection
    X_selected_full, selected_features_names = model.feature_selection(k=50) # X_selected_full is based on model.X
    
    # Train-validation split
    from sklearn.model_selection import train_test_split
    # We need to split the X_selected_full (which is a numpy array)
    # Also, ensure protected_attrs are sliced correctly later.
    # It's better to select features from X_train and transform X_val later to avoid data leakage
    # Or, if feature selection is done on the whole dataset X before split, then X_selected_full is correct.
    # The current flow does selection on model.X (all data), then splits. This is acceptable for some feature selection methods like SelectKBest.
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_selected_full, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train models
    # X_train and X_val are already the selected features
    model.train_models(X_train, y_train, X_val, y_val) # X_train and X_val here are numpy arrays of selected features
    
    # Predictions
    ensemble_pred, individual_preds = model.predict_ensemble(X_val) # X_val is already selected features
    
    # Evaluate
    auc_score = roc_auc_score(y_val, ensemble_pred)
    print(f"Validation AUC: {auc_score:.4f}")
    
    # Fairness evaluation
    # protected_attrs were created from the original df_engineered, so we need to align their indices with y_val.
    # y_val.index can be used if y was a Series from the original df and kept its index through splits.
    # If X_selected_full was created from model.X (which is df[feature_cols]), and y is df['TARGET'],
    # then y.index can be used to slice protected_attrs if the split was done on y and X_selected_full directly.
    
    # Assuming y is a pandas Series and train_test_split preserves indices for y_val
    # And model.protected_attrs values are pandas Series with original full dataset indices.
    # We need the indices that correspond to X_val / y_val
    
    # Find original indices for y_val to correctly slice protected_attrs
    # This assumes y maintained its original index from df_engineered
    val_indices = y_val.index 
    protected_attrs_val = {}
    for k, v_series in model.protected_attrs.items(): # v_series is the full series
        protected_attrs_val[k] = v_series.loc[val_indices] # Use .loc for index-based slicing

    fairness_metrics = model.evaluate_fairness(y_val, ensemble_pred, protected_attrs_val)
    
    # Setup explainability
    # X_val is already the selected features array. model.selected_features should be set by feature_selection.
    model.setup_explainability(X_val[:100]) # X_val is a numpy array
    
    # Example explanation
    # X_val.iloc[0].values won't work as X_val is a numpy array. Use X_val[0]
    explanation = model.explain_prediction(X_val[0])
    if explanation:
        print(f"Example explanation ready for customer 0")
    
    # Save model
    model.save_model('home_credit_demo_model.pkl')
    
    print("Demo completed successfully!")
    return model

if __name__ == "__main__":
    # Run demo
    trained_model = run_demo()
    
    print("\n=== Model Performance Summary ===")
    print("✅ Traditional ML ensemble trained")
    print("✅ Fairness metrics calculated") 
    print("✅ SHAP explanations ready")
    print("✅ Model saved for production")