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
    def __init__(self):
        self.models = {}
        self.feature_importance = {} # Bạn có thể xem xét việc loại bỏ nếu self.feature_importances được dùng thay thế
        self.fairness_metrics = {}
        self.explainer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = [] # Khởi tạo selected_features
        self.feature_importances = None # Để lưu feature importances từ LGBM
        
        # THÊM CÁC THUỘC TÍNH AUC
        self.lgb_auc = None
        self.xgb_auc = None
        self.ensemble_auc = None

        # Khởi tạo các thuộc tính khác để tránh lỗi nếu các bước không được chạy tuần tự
        self.app_train = pd.DataFrame()
        self.app_test = pd.DataFrame()
        self.bureau = pd.DataFrame()
        self.bureau_balance = pd.DataFrame()
        self.prev_app = pd.DataFrame()
        self.pos_cash = pd.DataFrame()
        self.installments = pd.DataFrame()
        self.credit_card = pd.DataFrame()
        self.df_engineered = pd.DataFrame()
        self.X = pd.DataFrame()
        self.y = pd.Series(dtype='float64')
        self.feature_names = []
        self.protected_attrs = {}
        self.shap_values = None # Nếu bạn lưu shap_values tổng thể
        self.classification_report_val = None
        self.confusion_matrix_val = None
        
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
        # df.get with a scalar default would return a single value when the column
        # is missing, leading to a boolean rather than a Series and causing
        # ``AttributeError: 'bool' object has no attribute 'astype'`` when
        # running ``.astype(int)``. Provide a Series default aligned with the
        # dataframe index so the expression always yields a Series.
        df['HAS_CAR'] = (
            df.get('FLAG_OWN_CAR', pd.Series('N', index=df.index)).eq('Y').astype(int)
        )
        df['HAS_REALTY'] = (
            df.get('FLAG_OWN_REALTY', pd.Series('N', index=df.index)).eq('Y').astype(int)
        )
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
    
    def explain_prediction(self, customer_data, customer_id=0):  # customer_id not used
        """Giải thích prediction cho một customer cụ thể"""
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return None
        
        # Prepare input for SHAP: ensure it's a 2D NumPy array
        if isinstance(customer_data, (pd.Series, pd.DataFrame)):
            input_data = customer_data.values.reshape(1, -1)
        elif isinstance(customer_data, np.ndarray):
            # Ensure it's 2D; if 1D, reshape to (1, n_features)
            input_data = customer_data.reshape(1, -1) if customer_data.ndim == 1 else customer_data
        else:
            logger.error("Unsupported customer_data type: must be pandas Series/DataFrame or NumPy array")
            return None
        
        # Get SHAP values for this customer
        shap_vals_for_customer = self.explainer.shap_values(input_data)
        
        # For binary classification, shap_values typically returns [shap_for_class0, shap_for_class1]
        # Select SHAP values for class 1 (positive class) or single output for regression
        explanation = {
            'expected_value': (self.explainer.expected_value[1] 
                            if isinstance(self.explainer.expected_value, (list, np.ndarray)) 
                            and len(self.explainer.expected_value) > 1 
                            else self.explainer.expected_value),
            'shap_values': (shap_vals_for_customer[1][0] 
                            if isinstance(shap_vals_for_customer, list) 
                            and len(shap_vals_for_customer) > 1 
                            else shap_vals_for_customer[0]),
            'feature_names': self.selected_features,
        'customer_data': customer_data  # Keep original format for compatibility
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
        # Đảm bảo feature_importances được lấy từ model lgb nếu có và selected_features đã được thiết lập
        if 'lgb' in self.models and hasattr(self.models['lgb'], 'feature_importances_') and self.selected_features:
            # Kiểm tra độ dài để đảm bảo khớp
            if len(self.selected_features) == len(self.models['lgb'].feature_importances_):
                self.feature_importances = dict(zip(
                    self.selected_features,
                    self.models['lgb'].feature_importances_
                ))
            else:
                logger.warning(f"Length mismatch: selected_features ({len(self.selected_features)}) vs lgb.feature_importances_ ({len(self.models['lgb'].feature_importances_)}). Not saving feature_importances.")
                self.feature_importances = {} # Hoặc None
        else:
            logger.warning("LGBM model, feature_importances_, or selected_features not available. Not saving feature_importances.")
            self.feature_importances = {} # Hoặc None

        model_package = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'explainer': self.explainer, 
            'fairness_metrics': self.fairness_metrics,
            'feature_importances': self.feature_importances,
            # LƯU CÁC THUỘC TÍNH AUC
            'lgb_auc': self.lgb_auc,
            'xgb_auc': self.xgb_auc,
            'ensemble_auc': self.ensemble_auc,
            'classification_report_val': self.classification_report_val,
            'confusion_matrix_val': self.confusion_matrix_val,
        }
        
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath='home_credit_model.pkl'):
        """Load model và components"""
        logger.info(f"Attempting to load model from {filepath}")
        model_package = joblib.load(filepath)
        
        self.models = model_package.get('models', {})
        self.scaler = model_package.get('scaler', StandardScaler()) # Cung cấp giá trị mặc định
        self.label_encoders = model_package.get('label_encoders', {})
        self.feature_selector = model_package.get('feature_selector')
        self.selected_features = model_package.get('selected_features', [])
        self.explainer = model_package.get('explainer') 
        self.fairness_metrics = model_package.get('fairness_metrics', {})
        self.feature_importances = model_package.get('feature_importances')

        # TẢI CÁC GIÁ TRỊ AUC ĐÃ LƯU
        self.lgb_auc = model_package.get('lgb_auc')
        self.xgb_auc = model_package.get('xgb_auc')
        self.ensemble_auc = model_package.get('ensemble_auc')
        self.classification_report_val = model_package.get('classification_report_val')
        self.confusion_matrix_val = model_package.get('confusion_matrix_val')
    # ...
        logger.info(f"Loaded Classification Report: {'Available' if self.classification_report_val else 'Not Available'}")
        logger.info(f"Loaded Confusion Matrix: {'Available' if self.confusion_matrix_val is not None else 'Not Available'}")
        logger.info(f"Model loaded successfully from {filepath}")
        # Log các giá trị AUC đã tải để kiểm tra
        logger.info(f"Loaded LGBM AUC: {self.lgb_auc}")
        logger.info(f"Loaded XGBoost AUC: {self.xgb_auc}")
        logger.info(f"Loaded Ensemble AUC: {self.ensemble_auc}")

# Demo Usage
def run_demo():
    """Chạy demo hoàn chỉnh"""
    print("=== HOME CREDIT RISK ASSESSMENT DEMO ===")
    # Initialize
    model = HomeCreditRiskAssessment()
    
    # Load data
    model.load_data()
    
    # Feature engineering
    df_features = model.feature_engineering() # df_features không được dùng sau đó, model.df_engineered được dùng
    
    # Prepare for modeling
    X, y = model.prepare_features() # X và y là pd.DataFrame/Series với index gốc
    
    # Feature selection
    # model.feature_selection() sẽ cập nhật self.selected_features
    # và trả về X_selected_full là numpy array không có tên cột
    X_selected_full_array, selected_features_names = model.feature_selection(k=50) 
    
    # Tạo DataFrame từ X_selected_full_array với tên cột đúng để train_test_split giữ lại tên cột
    # Điều này quan trọng cho việc lấy X_val với tên cột cho SHAP sau này
    X_selected_df = pd.DataFrame(X_selected_full_array, columns=selected_features_names, index=X.index)

    # Train-validation split
    from sklearn.model_selection import train_test_split
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_selected_df, y, test_size=0.2, stratify=y, random_state=42
    ) # Bây giờ X_train_df và X_val_df là DataFrames
    
    # Train models - các mô hình sẽ nhận DataFrame
    model.train_models(X_train_df, y_train, X_val_df, y_val)
    
    # Predictions trên tập validation để tính AUC
    # predict_ensemble mong đợi một DataFrame với các cột đã chọn
    ensemble_pred_val, individual_preds_val = model.predict_ensemble(X_val_df)
    y_pred_val_labels = (ensemble_pred_val > 0.5).astype(int)
    # Tính toán và gán các giá trị AUC vào instance của model
    report_dict = classification_report(y_val, y_pred_val_labels, output_dict=True, zero_division=0)
    cm_array = confusion_matrix(y_val, y_pred_val_labels)

    model.lgb_auc = roc_auc_score(y_val, individual_preds_val['lgb'])
    model.xgb_auc = roc_auc_score(y_val, individual_preds_val['xgb'])
    # Logistic Regression AUC (nếu bạn muốn hiển thị riêng)
    # model.lr_auc = roc_auc_score(y_val, individual_preds_val['lr']) 
    model.ensemble_auc = roc_auc_score(y_val, ensemble_pred_val)
    model.classification_report_val = report_dict
    model.confusion_matrix_val = cm_array

    logger.info("Classification Report (Validation):\n" + classification_report(y_val, y_pred_val_labels, zero_division=0))
    logger.info(f"Confusion Matrix (Validation):\n{cm_array}")
    logger.info(f"Validation LGBM AUC: {model.lgb_auc:.4f}")
    logger.info(f"Validation XGBoost AUC: {model.xgb_auc:.4f}")
    logger.info(f"Validation Ensemble AUC: {model.ensemble_auc:.4f}")
    
    # Fairness evaluation
    val_indices = y_val.index 
    protected_attrs_val = {}
    for k, v_series in model.protected_attrs.items():
        if not v_series.empty: # Kiểm tra xem series có dữ liệu không
             protected_attrs_val[k] = v_series.loc[val_indices]
        else:
            logger.warning(f"Protected attribute series '{k}' is empty. Skipping for fairness evaluation on validation set.")
            protected_attrs_val[k] = pd.Series(dtype='object', index=val_indices) # Tạo series rỗng để tránh lỗi
    
    # Chỉ đánh giá fairness nếu protected_attrs_val không rỗng hoàn toàn
    if any(not s.empty for s in protected_attrs_val.values()):
        fairness_metrics = model.evaluate_fairness(y_val, ensemble_pred_val, protected_attrs_val)
    else:
        logger.warning("No valid protected attributes for validation set. Skipping fairness evaluation.")


    # Setup explainability
    # setup_explainability mong muốn một DataFrame (hoặc numpy array nếu explainer hỗ trợ)
    # X_val_df là DataFrame các features đã chọn
    model.setup_explainability(X_val_df.sample(min(100, len(X_val_df)), random_state=42)) # Lấy sample nhỏ hơn nếu X_val_df < 100
    
    # Example explanation
    # model.explain_prediction mong đợi một hàng dữ liệu (1D numpy array hoặc pd.Series)
    if not X_val_df.empty:
        explanation = model.explain_prediction(X_val_df.iloc[0].values) # Truyền dưới dạng numpy array
        if explanation:
            print(f"Example explanation ready for customer 0")
    else:
        print("X_val_df is empty, cannot generate example explanation.")

    # Save model (Bây giờ model.save_model() sẽ lưu cả các giá trị AUC)
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