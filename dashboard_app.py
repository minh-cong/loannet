# dashboard_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import model class from your model file
# Ensure xloan_model.py is in the same directory or your PYTHONPATH
try:
    from xloan import HomeCreditRiskAssessment
except ImportError:
    st.error("Failed to import HomeCreditRiskAssessment. Make sure xloan_model.py is in the correct path.")
    # Provide a dummy class if import fails to allow dashboard structure to load
    class HomeCreditRiskAssessment:
        def __init__(self): self.models = {}
        def load_model(self, path): raise FileNotFoundError("Model file not found or class not loaded.")
        def predict_ensemble(self, X): return np.random.rand(X.shape[0]), {} # Dummy
        def explain_prediction(self, X_row): return {'expected_value': 0.1, 'shap_values': np.random.rand(len(self.selected_features)), 'feature_names': self.selected_features, 'customer_data': X_row} # Dummy
        selected_features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE_YEARS', 'CREDIT_INCOME_RATIO'] # Dummy
        scaler = None
        label_encoders = {}
        fairness_metrics = {}


class RiskAssessmentDashboard:
    """Dashboard cho Home Credit Risk Assessment"""

    def __init__(self):
        self.model_instance = HomeCreditRiskAssessment() # Instantiate your model class
        self.model_loaded = False
        self.selected_features = []
        self.label_encoders = {}
        self.scaler = None
        self.explainer = None
        self.fairness_metrics = {}
        self.feature_importances_lgb = pd.DataFrame({'feature':[], 'importance':[]})

        self.load_model_components()

    def load_model_components(self):
        """Load trained model and its components if available"""
        try:
            self.model_instance.load_model('home_credit_demo_model.pkl') # Path to your saved model
            
            # Extract necessary components from the loaded model instance
            self.selected_features = self.model_instance.selected_features
            self.label_encoders = self.model_instance.label_encoders
            self.scaler = self.model_instance.scaler
            self.explainer = getattr(self.model_instance, 'explainer', None)
            self.fairness_metrics = getattr(self.model_instance, 'fairness_metrics', {}) # Assumes fairness_metrics were saved or part of the class

            if 'lgb' in self.model_instance.models:
                importances = self.model_instance.models['lgb'].feature_importances_
                self.feature_importances_lgb = pd.DataFrame({
                    'feature': self.selected_features, # Make sure these align
                    'importance': importances
                }).sort_values(by='importance', ascending=False)

            self.model_loaded = True
            # Use a less obtrusive success message, or remove if sidebar status is enough
            # st.sidebar.success("✅ Model loaded successfully") 
        except FileNotFoundError:
            st.sidebar.error("❌ Model file 'home_credit_demo_model.pkl' not found.")
            self.model_loaded = False
        except AttributeError as e:
            st.sidebar.error(f"❌ Model components missing or attribute error: {e}")
            self.model_loaded = False
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {e}")
            self.model_loaded = False

    # --- START OF SECTION TO BE REPLACED ---
    def _preprocess_input_data(self, data_dict):
        """
        Basic preprocessing for individual assessment.
        This is a SIMPLIFIED version. Real-world usage would require
        replicating ALL feature engineering steps from your training pipeline
        that apply to the 'selected_features'.
        """
        if not self.model_loaded or not self.selected_features:
            st.error("Model or selected features not loaded. Cannot preprocess.")
            return None

        # Create a DataFrame with a single row
        df_input = pd.DataFrame([data_dict])

        # 1. Basic Derived Features (example, must match your model's features)
        if 'AMT_CREDIT' in df_input.columns and 'AMT_INCOME_TOTAL' in df_input.columns:
            df_input['CREDIT_INCOME_RATIO'] = df_input['AMT_CREDIT'] / (df_input['AMT_INCOME_TOTAL'] + 1e-6) # Avoid division by zero
        if 'DAYS_BIRTH' in df_input.columns: # Assuming age is DAYS_BIRTH in your features
            df_input['AGE_YEARS'] = -df_input['DAYS_BIRTH'] / 365
        
        # Add missing derived features
        if 'AMT_CREDIT' in df_input.columns and 'AMT_ANNUITY' in df_input.columns:
            df_input['CREDIT_ANNUITY_RATIO'] = df_input['AMT_CREDIT'] / (df_input['AMT_ANNUITY'] + 1e-6)
        
        if 'AMT_GOODS_PRICE' in df_input.columns and 'AMT_CREDIT' in df_input.columns:
            df_input['GOODS_PRICE_CREDIT_RATIO'] = df_input['AMT_GOODS_PRICE'] / (df_input['AMT_CREDIT'] + 1e-6)
        
        if 'DAYS_EMPLOYED' in df_input.columns:
            # Calculate employment years, handle unemployed case (365243 is the flag for unemployed)
            df_input['EMPLOYED_YEARS'] = np.where(
                df_input['DAYS_EMPLOYED'] == 365243, 
                0,  # Unemployed
                -df_input['DAYS_EMPLOYED'] / 365
            )
        
        # Age group categorization
        if 'AGE_YEARS' in df_input.columns:
            # Ensure AGE_YEARS is calculated first if not already present
            if 'AGE_YEARS' not in df_input.columns and 'DAYS_BIRTH' in df_input.columns:
                df_input['AGE_YEARS'] = -df_input['DAYS_BIRTH'] / 365
            
            if 'AGE_YEARS' in df_input.columns: # Check again after potential creation
                df_input['AGE_GROUP'] = pd.cut(
                    df_input['AGE_YEARS'], 
                    bins=[0, 25, 35, 50, 100], 
                    labels=[0, 1, 2, 3], # Creates integer labels
                    right=False 
                ).astype(int) 
        
        # External source features
        ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        if all(col in df_input.columns for col in ext_cols):
            df_input['EXT_SOURCE_MEAN'] = df_input[ext_cols].mean(axis=1)
            df_input['EXT_SOURCE_STD'] = df_input[ext_cols].std(axis=1)
            df_input['EXT_SOURCE_PROD'] = df_input[ext_cols].prod(axis=1)

        # 2. Label Encoding (use loaded encoders)
        for col, le in self.label_encoders.items():
            if col in df_input.columns:
                if col == 'AGE_GROUP' and pd.api.types.is_integer_dtype(df_input[col]):
                    continue
                try:
                    df_input[col] = le.transform(df_input[col].astype(str))
                except ValueError as e:
                    st.warning(f"Unseen value in '{col}'. Assigning -1 (unknown). Prediction may be less accurate. Original error: {e}")
                    df_input[col] = -1 

        # 3. Ensure all selected_features are present, fill NaNs if necessary
        processed_data = pd.DataFrame(columns=self.selected_features)
        for feature in self.selected_features:
            if feature in df_input.columns:
                processed_data[feature] = df_input[feature]
            else:
                processed_data[feature] = np.nan 
                st.warning(f"Feature '{feature}' not derivable from form or created features. Will be imputed. Prediction accuracy affected.")

        for col in processed_data.columns:
            if processed_data[col].isnull().any():
                if pd.api.types.is_numeric_dtype(processed_data[col]):
                    processed_data[col] = processed_data[col].fillna(0) 
                else:
                    processed_data[col] = processed_data[col].fillna(-1) 
        
        for col in self.selected_features: # Ensure this loop is correct
            if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col]):
                processed_data[col] = processed_data[col].astype(float)

        # 4. Scaling (use loaded scaler)
        if self.scaler:
            scaled_features_list = []
            explicit_scaled_features_provided = False

            if hasattr(self.scaler, 'feature_names_in_'):
                scaled_features_list = self.scaler.feature_names_in_
                explicit_scaled_features_provided = True
            elif hasattr(self.model_instance, 'scaled_feature_names') and self.model_instance.scaled_feature_names is not None:
                # This assumes you might save a list of scaled feature names with your model
                scaled_features_list = self.model_instance.scaled_feature_names
                explicit_scaled_features_provided = True
                st.info("Using an explicitly saved list of scaled features.")

            if not explicit_scaled_features_provided:
                # Fallback: Identify numeric columns from selected_features present in processed_data
                candidate_scaled_features = [
                    col for col in self.selected_features 
                    if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col])
                ]
                
                if hasattr(self.scaler, 'n_features_in_'):
                    if len(candidate_scaled_features) == self.scaler.n_features_in_:
                        scaled_features_list = candidate_scaled_features
                        # Optional: st.info("Inferred scaled features list matches scaler's expected number of features.")
                    else:
                        st.error(
                            f"Scaler expects {self.scaler.n_features_in_} features, but found "
                            f"{len(candidate_scaled_features)} numeric features among selected_features. "
                            "Scaling might be incorrect or fail. Check your feature list for scaling."
                        )
                        # Decide if you want to proceed with candidate_scaled_features or stop/not scale
                        scaled_features_list = candidate_scaled_features # Proceed with caution
                else:
                    # No feature_names_in_ and no n_features_in_ (very old scaler or custom)
                    st.warning(
                        "Scaler metadata (feature_names_in_ or n_features_in_) not found. "
                        "Assuming all derived numeric features from selected_features should be scaled. "
                        "This is a strong assumption. Ensure this matches your training process."
                    )
                    scaled_features_list = candidate_scaled_features
            
            # Filter this list to ensure all columns are actually in processed_data and numeric
            # (redundant if candidate_scaled_features was the source, but good for explicit lists)
            actual_cols_to_scale = [
                col for col in scaled_features_list 
                if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col])
            ]

            # Check if the number of columns derived matches the original scaled_features_list intent
            # (especially if scaled_features_list came from feature_names_in_ or explicit list)
            if explicit_scaled_features_provided and len(actual_cols_to_scale) != len(scaled_features_list):
                missing_explicit_cols = set(scaled_features_list) - set(actual_cols_to_scale)
                st.warning(f"Some explicitly defined scaled features are not available or not numeric in processed data: {missing_explicit_cols}. They will not be scaled.")

            if actual_cols_to_scale:
                try:
                    data_for_scaling = processed_data[actual_cols_to_scale].copy()
                    
                    # Final check for NaNs before transform (should have been handled by imputation)
                    for col_ds in data_for_scaling.columns:
                        if data_for_scaling[col_ds].isnull().any():
                            st.warning(f"NaNs found in column '{col_ds}' just before scaling. Filling with 0. Review imputation.")
                            data_for_scaling[col_ds] = data_for_scaling[col_ds].fillna(0) 

                    # Verify number of features if n_features_in_ is available
                    if hasattr(self.scaler, 'n_features_in_') and data_for_scaling.shape[1] != self.scaler.n_features_in_:
                        st.error(f"Critical: Number of columns for scaling ({data_for_scaling.shape[1]}, names: {data_for_scaling.columns.tolist()}) "
                                f"does not match scaler's expected input features ({self.scaler.n_features_in_}). Scaling aborted for safety.")
                    else:
                        scaled_values = self.scaler.transform(data_for_scaling)
                        processed_data[actual_cols_to_scale] = scaled_values
                        
                except ValueError as e:
                    st.error(f"Error during scaling: {e}. Input features might not match scaler's expectations. "
                            f"Attempted to scale: {actual_cols_to_scale}. Some features might remain unscaled.")
                except Exception as e:
                    st.error(f"Unexpected error during scaling: {e}. Some features might remain unscaled.")
            elif scaled_features_list: # scaled_features_list was determined, but actual_cols_to_scale is empty
                st.warning("List of features to scale was determined, but none are present/numeric in the current processed data. Scaling skipped.")
            # else: No features determined for scaling, so do nothing.

        # Ensure the final DataFrame has columns in the exact order of self.selected_features
        try:
            final_processed_df = processed_data[self.selected_features]
        except KeyError as e:
            missing_cols_final = set(self.selected_features) - set(processed_data.columns)
            st.error(f"KeyError when selecting final features: {e}. Missing columns: {missing_cols_final}")
            st.error(f"Processed columns available: {processed_data.columns.tolist()}")
            st.error(f"Expected columns (self.selected_features): {self.selected_features}")
            return None 

        return final_processed_df
    # --- END OF SECTION TO BE REPLACED ---

    def main_dashboard(self):
        """Main dashboard interface"""
        st.set_page_config(
            page_title="Home Credit Risk Assessment",
            page_icon="🏦",
            layout="wide"
        )

        st.title("🏦 Home Credit Risk Assessment Dashboard")
        st.markdown("**Traditional ML + Fairness-First Approach**")

        self.create_sidebar()

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Risk Assessment",
            "🔍 Model Explainability",
            "⚖️ Fairness Analysis",
            "📈 Model Performance"
        ])

        with tab1:
            self.risk_assessment_tab()
        with tab2:
            self.explainability_tab()
        with tab3:
            self.fairness_tab()
        with tab4:
            self.performance_tab()

    def create_sidebar(self):
        st.sidebar.header("🎛️ Model Controls")
        if self.model_loaded:
            st.sidebar.success("Model Status: Ready")
            st.sidebar.caption(f"Features: {len(self.selected_features if self.selected_features else [])}")
        else:
            st.sidebar.error("Model Status: Not Loaded/Error")

        st.sidebar.markdown("---")
        st.sidebar.header("🚀 Quick Actions")
        if st.sidebar.button("🔄 Reload Model Components"):
            self.load_model_components()
            st.experimental_rerun()


        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Settings")
        self.decision_threshold = st.sidebar.slider(
            "Decision Threshold", 0.0, 1.0, 0.5, 0.01
        )
        # self.fairness_tolerance = st.sidebar.slider( # Not used yet
        # "Fairness Tolerance", 0.0, 0.2, 0.1, 0.01
        # )


    def risk_assessment_tab(self):
        st.header("📊 Individual Risk Assessment")
        if not self.model_loaded:
            st.warning("Model not loaded. Risk assessment functionality is limited/disabled.")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Customer Information")
            with st.form("customer_input_form"):
                # Basic Information
                st.markdown("### Basic Information")
                amt_income_total = st.number_input("Annual Income ($)", min_value=0, value=50000, key="income")
                amt_credit = st.number_input("Credit Amount ($)", min_value=0, value=200000, key="credit")
                amt_goods_price = st.number_input("Goods Price ($)", min_value=0, value=180000, key="goods_price")
                amt_annuity = st.number_input("Annuity Amount ($)", min_value=0, value=12000, key="annuity")
                
                age_years_form = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, key="age")
                days_birth = -(age_years_form * 365)
                
                employed_years_form = st.number_input("Years Employed", min_value=0, max_value=50, value=5, key="emp_years")
                days_employed = -(employed_years_form * 365) if employed_years_form > 0 else 365243
                
                cnt_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, key="children")

                # Personal Details
                st.markdown("### Personal Details")
                code_gender = st.selectbox("Gender", ["F", "M"], key="gender")
                
                education_options = ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
                name_education_type = st.selectbox("Education", education_options, key="education")
                
                income_type_options = ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"]
                name_income_type = st.selectbox("Income Type", income_type_options, key="income_type")
                
                contract_type_options = ["Cash loans", "Revolving loans"]
                name_contract_type = st.selectbox("Contract Type", contract_type_options, key="contract_type")

                # Housing Information
                st.markdown("### Housing Information")
                housing_type_options = ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"]
                name_housing_type = st.selectbox("Housing Type", housing_type_options, key="housing_type")
                
                # Organization
                org_type_options = ["Business Entity Type 3", "School", "Government", "Religion", "Other", "XNA", "Electricity", "Medicine", "Business Entity Type 2", "Self-employed", "Transport: type 2", "Construction", "Housing", "Kindergarten", "Trade: type 7", "Industry: type 11", "Military", "Services", "Security Ministries", "Transport: type 4", "Industry: type 1", "Emergency", "Security", "Trade: type 2", "University", "Transport: type 3", "Police", "Business Entity Type 1", "Postal", "Industry: type 4", "Agriculture", "Restaurant", "Culture", "Hotel", "Industry: type 7", "Trade: type 3", "Industry: type 3", "Bank", "Industry: type 9", "Insurance", "Trade: type 6", "Industry: type 2", "Transport: type 1", "Industry: type 12", "Mobile", "Trade: type 1", "Industry: type 5", "Industry: type 10", "Legal Services", "Advertising", "Trade: type 5", "Cleaning", "Industry: type 13", "Trade: type 4", "Telecom", "Industry: type 8", "Realtor", "Industry: type 6"]
                organization_type = st.selectbox("Organization Type", org_type_options, key="org_type")

                # Regional Information
                st.markdown("### Regional Information")
                region_population_relative = st.slider("Region Population Relative", 0.0, 0.1, 0.02, 0.001, key="region_pop")
                region_rating_client = st.selectbox("Region Rating Client", [1, 2, 3], key="region_rating")
                region_rating_client_w_city = st.selectbox("Region Rating Client with City", [1, 2, 3], key="region_rating_city")
                
                # Registration and ID
                days_registration = st.number_input("Days since Registration", min_value=0, max_value=25000, value=5000, key="days_reg")
                days_registration = -days_registration
                days_id_publish = st.number_input("Days since ID Published", min_value=0, max_value=10000, value=3000, key="days_id")
                days_id_publish = -days_id_publish
                days_last_phone_change = st.number_input("Days since Last Phone Change", min_value=0, max_value=5000, value=365, key="phone_change")
                days_last_phone_change = -days_last_phone_change

                # Contact Information
                st.markdown("### Contact Information")
                flag_emp_phone = st.selectbox("Has Employment Phone", [0, 1], key="emp_phone")
                flag_work_phone = st.selectbox("Has Work Phone", [0, 1], key="work_phone")
                
                # Location Flags
                reg_city_not_live_city = st.selectbox("Registration City ≠ Live City", [0, 1], key="reg_live")
                reg_city_not_work_city = st.selectbox("Registration City ≠ Work City", [0, 1], key="reg_work")
                live_city_not_work_city = st.selectbox("Live City ≠ Work City", [0, 1], key="live_work")

                # Building Information (if available)
                st.markdown("### Building Information")
                elevators_avg = st.number_input("Elevators Average", min_value=0.0, max_value=1.0, value=0.1, key="elevators_avg")
                floorsmax_avg = st.number_input("Floors Max Average", min_value=0.0, max_value=1.0, value=0.2, key="floors_avg")
                livingarea_avg = st.number_input("Living Area Average", min_value=0.0, max_value=1.0, value=0.3, key="living_avg")
                
                elevators_mode = st.number_input("Elevators Mode", min_value=0.0, max_value=1.0, value=0.1, key="elevators_mode")
                floorsmax_mode = st.number_input("Floors Max Mode", min_value=0.0, max_value=1.0, value=0.2, key="floors_mode")
                livingarea_mode = st.number_input("Living Area Mode", min_value=0.0, max_value=1.0, value=0.3, key="living_mode")
                
                elevators_medi = st.number_input("Elevators Median", min_value=0.0, max_value=1.0, value=0.1, key="elevators_medi")
                floorsmax_medi = st.number_input("Floors Max Median", min_value=0.0, max_value=1.0, value=0.2, key="floors_medi")
                livingarea_medi = st.number_input("Living Area Median", min_value=0.0, max_value=1.0, value=0.3, key="living_medi")

                # Building details
                fondkapremont_mode_options = ["not specified", "reg oper account", "org spec account", "reg oper spec account"]
                fondkapremont_mode = st.selectbox("Fund Repair Mode", fondkapremont_mode_options, key="fond_mode")
                
                housetype_mode_options = ["block of flats", "terraced house", "specific housing"]
                housetype_mode = st.selectbox("House Type Mode", housetype_mode_options, key="house_mode")
                
                totalarea_mode = st.number_input("Total Area Mode", min_value=0.0, max_value=1.0, value=0.1, key="total_area")
                
                wallsmaterial_mode_options = ["Panel", "Block", "Mixed", "Brick", "Monolithic", "Others", "Stone, brick", "Wooden"]
                wallsmaterial_mode = st.selectbox("Walls Material Mode", wallsmaterial_mode_options, key="walls_mode")
                
                emergencystate_mode_options = ["No", "Yes"]
                emergencystate_mode = st.selectbox("Emergency State Mode", emergencystate_mode_options, key="emergency_mode")

                # Social Circle
                st.markdown("### Social Circle")
                def_30_cnt_social_circle = st.number_input("30 Days Default Count Social Circle", min_value=0, max_value=50, value=0, key="def_30")
                def_60_cnt_social_circle = st.number_input("60 Days Default Count Social Circle", min_value=0, max_value=50, value=0, key="def_60")

                # Documents
                st.markdown("### Documents")
                flag_document_3 = st.selectbox("Has Document 3", [0, 1], key="doc_3")
                flag_document_6 = st.selectbox("Has Document 6", [0, 1], key="doc_6")

                # External Sources
                st.markdown("### External Sources")
                ext_source_1 = st.slider("External Source 1", 0.0, 1.0, 0.5, 0.01, key="ext1")
                ext_source_2 = st.slider("External Source 2", 0.0, 1.0, 0.5, 0.01, key="ext2")
                ext_source_3 = st.slider("External Source 3", 0.0, 1.0, 0.5, 0.01, key="ext3")

                submitted = st.form_submit_button("🔍 Assess Risk")

            if submitted:
                if not self.model_loaded:
                    st.error("Cannot assess risk: Model is not loaded.")
                    return

                input_data = {
                    # Basic fields
                    'AMT_INCOME_TOTAL': amt_income_total,
                    'AMT_CREDIT': amt_credit,
                    'AMT_GOODS_PRICE': amt_goods_price,
                    'AMT_ANNUITY': amt_annuity,
                    'DAYS_BIRTH': days_birth,
                    'DAYS_EMPLOYED': days_employed,
                    'CNT_CHILDREN': cnt_children,
                    
                    # Personal
                    'CODE_GENDER': code_gender,
                    'NAME_EDUCATION_TYPE': name_education_type,
                    'NAME_INCOME_TYPE': name_income_type,
                    'NAME_CONTRACT_TYPE': name_contract_type,
                    'NAME_HOUSING_TYPE': name_housing_type,
                    'ORGANIZATION_TYPE': organization_type,
                    
                    # Regional
                    'REGION_POPULATION_RELATIVE': region_population_relative,
                    'REGION_RATING_CLIENT': region_rating_client,
                    'REGION_RATING_CLIENT_W_CITY': region_rating_client_w_city,
                    
                    # Dates
                    'DAYS_REGISTRATION': days_registration,
                    'DAYS_ID_PUBLISH': days_id_publish,
                    'DAYS_LAST_PHONE_CHANGE': days_last_phone_change,
                    
                    # Flags
                    'FLAG_EMP_PHONE': flag_emp_phone,
                    'FLAG_WORK_PHONE': flag_work_phone,
                    'REG_CITY_NOT_LIVE_CITY': reg_city_not_live_city,
                    'REG_CITY_NOT_WORK_CITY': reg_city_not_work_city,
                    'LIVE_CITY_NOT_WORK_CITY': live_city_not_work_city,
                    
                    # Building info
                    'ELEVATORS_AVG': elevators_avg,
                    'FLOORSMAX_AVG': floorsmax_avg,
                    'LIVINGAREA_AVG': livingarea_avg,
                    'ELEVATORS_MODE': elevators_mode,
                    'FLOORSMAX_MODE': floorsmax_mode,
                    'LIVINGAREA_MODE': livingarea_mode,
                    'ELEVATORS_MEDI': elevators_medi,
                    'FLOORSMAX_MEDI': floorsmax_medi,
                    'LIVINGAREA_MEDI': livingarea_medi,
                    'FONDKAPREMONT_MODE': fondkapremont_mode,
                    'HOUSETYPE_MODE': housetype_mode,
                    'TOTALAREA_MODE': totalarea_mode,
                    'WALLSMATERIAL_MODE': wallsmaterial_mode,
                    'EMERGENCYSTATE_MODE': emergencystate_mode,
                    
                    # Social
                    'DEF_30_CNT_SOCIAL_CIRCLE': def_30_cnt_social_circle,
                    'DEF_60_CNT_SOCIAL_CIRCLE': def_60_cnt_social_circle,
                    
                    # Documents
                    'FLAG_DOCUMENT_3': flag_document_3,
                    'FLAG_DOCUMENT_6': flag_document_6,
                    
                    # External sources
                    'EXT_SOURCE_1': ext_source_1,
                    'EXT_SOURCE_2': ext_source_2,
                    'EXT_SOURCE_3': ext_source_3,
                }
                
                processed_df = self._preprocess_input_data(input_data)

                if processed_df is not None and not processed_df.empty:
                    try:
                        risk_score_proba, _ = self.model_instance.predict_ensemble(processed_df)
                        risk_score = risk_score_proba[0] # Get score for the single instance

                        st.session_state.risk_assessment = {
                            'risk_score': risk_score,
                            'decision': 'REJECT' if risk_score > self.decision_threshold else 'APPROVE',
                            'confidence': abs(risk_score - self.decision_threshold) * 2, # Simple confidence
                            'raw_input': input_data,
                            'processed_input_for_shap': processed_df.iloc[0].values # For SHAP
                        }
                        # For simplicity, risk factors are not dynamically generated from SHAP here
                        # but could be in a more advanced version.
                        st.session_state.risk_assessment['factors'] = self.get_demo_risk_factors(input_data, processed_df.iloc[0]) # Pass processed data too

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        if 'risk_assessment' in st.session_state:
                            del st.session_state['risk_assessment']
                else:
                    st.error("Failed to preprocess input data for prediction.")
                    if 'risk_assessment' in st.session_state:
                        del st.session_state['risk_assessment']


        with col2:
            st.subheader("Risk Assessment Results")
            self.display_risk_results()

    def get_demo_risk_factors(self, raw_input_data, processed_data_row=None):
        """Generates simplified demo risk factors based on raw and processed input."""
        factors = []
        
        # Use processed_data_row if available for derived features, otherwise calculate from raw
        if processed_data_row is not None and 'CREDIT_INCOME_RATIO' in self.selected_features:
             # Assuming self.selected_features is a list and processed_data_row is a pandas Series or similar
            if isinstance(processed_data_row, pd.Series):
                credit_income_ratio = processed_data_row.get('CREDIT_INCOME_RATIO', float('nan'))
            else: # if numpy array
                try:
                    idx = self.selected_features.index('CREDIT_INCOME_RATIO')
                    credit_income_ratio = processed_data_row[idx]
                except (ValueError, IndexError):
                    credit_income_ratio = (raw_input_data['AMT_CREDIT'] / (raw_input_data['AMT_INCOME_TOTAL'] + 1e-6))

        else: # Fallback to raw calculation
            credit_income_ratio = (raw_input_data['AMT_CREDIT'] / (raw_input_data['AMT_INCOME_TOTAL'] + 1e-6))

        if processed_data_row is not None and 'AGE_YEARS' in self.selected_features:
            if isinstance(processed_data_row, pd.Series):
                age_years = processed_data_row.get('AGE_YEARS', float('nan'))
            else: # if numpy array
                try:
                    idx = self.selected_features.index('AGE_YEARS')
                    age_years = processed_data_row[idx]
                except (ValueError, IndexError):
                    age_years = -raw_input_data['DAYS_BIRTH'] / 365
        else:
            age_years = -raw_input_data['DAYS_BIRTH'] / 365
        
        if processed_data_row is not None and 'EMPLOYED_YEARS' in self.selected_features:
            if isinstance(processed_data_row, pd.Series):
                employment_years = processed_data_row.get('EMPLOYED_YEARS', float('nan'))
            else: # if numpy array
                try:
                    idx = self.selected_features.index('EMPLOYED_YEARS')
                    employment_years = processed_data_row[idx]
                except (ValueError, IndexError):
                     employment_years = -raw_input_data['DAYS_EMPLOYED'] / 365 if raw_input_data['DAYS_EMPLOYED'] < 0 else 0
        else:
            employment_years = -raw_input_data['DAYS_EMPLOYED'] / 365 if raw_input_data['DAYS_EMPLOYED'] < 0 else 0


        if not np.isnan(credit_income_ratio) and credit_income_ratio > 5:
            factors.append(("High Credit-to-Income Ratio", f"{credit_income_ratio:.1f}x", "High Risk"))
        if not np.isnan(age_years) and age_years < 25:
            factors.append(("Young Age", f"{int(age_years)} years", "Medium Risk"))
        if not np.isnan(employment_years) and employment_years < 2:
            factors.append(("Short Employment History", f"{employment_years:.1f} years", "Medium Risk"))
        if raw_input_data.get('CNT_CHILDREN', 0) > 2:
            factors.append(("Many Dependents", f"{raw_input_data['CNT_CHILDREN']} children", "Low Risk"))
        
        # Check external sources (example)
        if raw_input_data.get('EXT_SOURCE_1', 0.5) < 0.2:
             factors.append(("Low External Source 1 Score", f"{raw_input_data['EXT_SOURCE_1']:.2f}", "High Risk"))
        if raw_input_data.get('EXT_SOURCE_2', 0.5) < 0.2:
             factors.append(("Low External Source 2 Score", f"{raw_input_data['EXT_SOURCE_2']:.2f}", "High Risk"))


        if not factors:
            factors.append(("Standard Profile", "Normal risk factors", "Low Risk"))
        return factors

    def display_risk_results(self):
        if 'risk_assessment' not in st.session_state:
            st.info("👆 Enter customer information and click 'Assess Risk' to see results.")
            return

        assessment = st.session_state.risk_assessment
        risk_score = assessment['risk_score']

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score (%) - Higher is Riskier"},
            delta={'reference': self.decision_threshold * 100, 'increasing': {'color': "Red"}, 'decreasing': {'color': "Green"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, (self.decision_threshold - 0.2)*100 if self.decision_threshold > 0.2 else 20], 'color': "lightgreen"}, # Low risk
                    {'range': [(self.decision_threshold - 0.2)*100 if self.decision_threshold > 0.2 else 20, (self.decision_threshold + 0.2)*100 if self.decision_threshold < 0.8 else 80], 'color': "yellow"}, # Medium
                    {'range': [(self.decision_threshold + 0.2)*100 if self.decision_threshold < 0.8 else 80, 100], 'color': "lightcoral"}  # High
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': self.decision_threshold * 100
                }
            }
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=50))
        st.plotly_chart(fig, use_container_width=True)

        decision_color = "🔴" if assessment['decision'] == 'REJECT' else "🟢"
        st.markdown(f"### {decision_color} Decision: **{assessment['decision']}**")
        # st.markdown(f"**Confidence:** {assessment['confidence']:.1%}") # Confidence metric was simplistic

        st.subheader("📝 Key Factors (Demo)")
        if 'factors' in assessment and assessment['factors']:
            for factor, value, risk_level in assessment['factors']:
                color_map = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
                st.markdown(f"{color_map.get(risk_level, '⚪')} **{factor}**: {value} ({risk_level})")
        else:
            st.markdown("No specific risk factors identified for this demo profile.")


    def explainability_tab(self):
        st.header("🔍 Model Explainability")
        if not self.model_loaded:
            st.warning("Model not loaded. Explainability features are disabled.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Global Feature Importance (LGBM)")
            if not self.feature_importances_lgb.empty:
                fig = px.bar(self.feature_importances_lgb.head(15),
                             x='importance', y='feature', orientation='h',
                             title="Top 15 Feature Importances (LGBM)",
                             height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("LGBM feature importances not available.")

        with col2:
            st.subheader("💡 Local Explanation (SHAP for last assessed customer)")
            if 'risk_assessment' in st.session_state and 'processed_input_for_shap' in st.session_state.risk_assessment:
                if self.explainer or hasattr(self.model_instance, 'explain_prediction'): # Check for either explainer or method
                    try:
                        # Get the processed input used for prediction
                        customer_data_for_shap_values = st.session_state.risk_assessment['processed_input_for_shap']
                        
                        # Ensure it's a 1D array for explain_prediction if it takes a single row
                        if customer_data_for_shap_values.ndim > 1:
                            customer_data_for_shap_values = customer_data_for_shap_values.flatten()

                        # Create a DataFrame for display/context if needed by explain_prediction or for SHAP plots directly
                        # The column names must match self.selected_features
                        customer_df_for_shap_display = pd.DataFrame([customer_data_for_shap_values], columns=self.selected_features)

                        # Call the model's explain_prediction method
                        explanation_object = self.model_instance.explain_prediction(customer_df_for_shap_display.iloc[0])

                        if explanation_object:
                            shap_values_for_class1 = explanation_object['shap_values']
                            expected_value_class1 = explanation_object['expected_value']
                            feature_names_for_shap = explanation_object.get('feature_names', self.selected_features)
                            # Customer data for shap might be different from processed_input_for_shap if explain_prediction re-fetches or transforms
                            customer_data_from_expl = explanation_object.get('customer_data', customer_df_for_shap_display.iloc[0].values)


                            st.write(f"SHAP Base Value (Average Model Output): {expected_value_class1:.4f}")

                            # Create SHAP Explanation object
                            shap_explanation = shap.Explanation(
                                values=shap_values_for_class1,
                                base_values=expected_value_class1,
                                data=customer_data_from_expl, # Use data that matches shap_values
                                feature_names=feature_names_for_shap
                            )

                            # Waterfall plot
                            fig_waterfall, ax_waterfall = plt.subplots()
                            shap.waterfall_plot(shap_explanation, max_display=10, show=False)
                            ax_waterfall.tick_params(axis='y', labelsize=8) # Adjust label size if needed
                            ax_waterfall.tick_params(axis='x', labelsize=8)
                            plt.tight_layout()
                            st.pyplot(fig_waterfall)
                            plt.close(fig_waterfall)

                            st.info("Force plot display can be complex in Streamlit. Waterfall plot shown above.")
                        else:
                            st.error("Could not generate SHAP explanation object from model_instance.explain_prediction.")
                    except Exception as e:
                        st.error(f"Error generating SHAP plot: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                else:
                    st.info("SHAP explainer or explain_prediction method not available (was not loaded/defined with the model).")
            else:
                st.info("Assess a customer first to see their specific SHAP explanation.")


    def fairness_tab(self):
        st.header("⚖️ Fairness Analysis")
        if not self.model_loaded:
            st.warning("Model not loaded. Fairness analysis features are disabled.")
            return

        if self.fairness_metrics:
            st.subheader("Fairness Metrics (from training/validation set)")
            for group, metrics in self.fairness_metrics.items():
                st.markdown(f"**Protected Attribute: {group.replace('_', ' ').title()}**")
                dpd = metrics.get('demographic_parity_diff', 'N/A')
                eod = metrics.get('equalized_odds_diff', 'N/A')
                st.metric(label="Demographic Parity Difference", value=f"{dpd:.4f}" if isinstance(dpd, float) else dpd)
                st.metric(label="Equalized Odds Difference", value=f"{eod:.4f}" if isinstance(eod, float) else eod)
                st.markdown("---")
        else:
            st.info("Fairness metrics not available (were not loaded with the model).")

        st.markdown("Fairness metrics are typically calculated on a validation set during model training. These values reflect the model's performance on that dataset with respect to different demographic groups.")


    def performance_tab(self):
        st.header("📈 Model Performance")
        if not self.model_loaded:
            st.warning("Model not loaded. Performance metrics are not available.")
            return

        # These would ideally be loaded from the model package or re-calculated
        # For now, using placeholders or values from your last run
        st.subheader("Overall Performance (on Validation Set during training)")

        auc_lgb = "N/A"
        auc_xgb = "N/A"
        auc_ensemble = "N/A"

        if hasattr(self.model_instance, 'models') and isinstance(self.model_instance.models, dict):
            auc_lgb = self.model_instance.models.get('lgb_auc', "N/A") 
            auc_xgb = self.model_instance.models.get('xgb_auc', "N/A")
            auc_ensemble = self.model_instance.models.get('ensemble_auc', "N/A")

        col1, col2, col3 = st.columns(3)
        col1.metric("LGBM AUC", f"{auc_lgb:.4f}" if isinstance(auc_lgb, float) else auc_lgb)
        col2.metric("XGBoost AUC", f"{auc_xgb:.4f}" if isinstance(auc_xgb, float) else auc_xgb)
        col3.metric("Ensemble AUC", f"{auc_ensemble:.4f}" if isinstance(auc_ensemble, float) else auc_ensemble)


        st.markdown("---")
        st.subheader("Classification Report (Example)")
        # This would be from y_val and predictions on validation set
        # For demo, showing a static image or a pre-formatted text
        # You could generate this if you have y_val and predictions
        report_data = {
            'precision': [0.75, 0.25],
            'recall': [0.90, 0.15],
            'f1-score': [0.82, 0.19],
            'support': [1800, 200]
        }
        report_df = pd.DataFrame(report_data, index=['Class 0 (No Default)', 'Class 1 (Default)'])
        st.table(report_df)

        st.subheader("Confusion Matrix (Example)")
        # Dummy confusion matrix
        cm_fig = px.imshow([[1620, 180], [170, 30]],
                           labels=dict(x="Predicted Label", y="True Label", color="Count"),
                           x=['No Default', 'Default'],
                           y=['No Default', 'Default'],
                           text_auto=True, color_continuous_scale='Blues',
                           height=400)
        cm_fig.update_layout(title_text='Confusion Matrix (Validation Set Example)')
        st.plotly_chart(cm_fig, use_container_width=True)

# Ensure matplotlib is imported for SHAP plots if not already
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dashboard = RiskAssessmentDashboard()
    dashboard.main_dashboard()