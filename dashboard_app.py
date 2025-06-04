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
        # ... Add other critical derived features that are in self.selected_features

        # 2. Label Encoding (use loaded encoders)
        for col, le in self.label_encoders.items():
            if col in df_input.columns:
                # Handle unseen labels: map to a specific category (e.g., -1 or len(classes))
                # For simplicity, this example might error or mis-encode if new value
                try:
                    df_input[col] = le.transform(df_input[col].astype(str))
                except ValueError:
                    # Simplistic handling: assign a common value or a special "unknown" code
                    # This should ideally be a robust strategy defined during model training
                    st.warning(f"Unseen value in '{col}'. Assigning default. Prediction may be less accurate.")
                    df_input[col] = -1 # Or some other placeholder your model might expect for unknowns

        # 3. Ensure all selected_features are present, fill NaNs if necessary
        # This part is crucial: the final DataFrame must have exactly the selected_features
        # with appropriate values (e.g., median imputation for missing ones if that's what training did)
        processed_data = pd.DataFrame(columns=self.selected_features)
        for feature in self.selected_features:
            if feature in df_input.columns:
                processed_data[feature] = df_input[feature]
            else:
                # Fill with a default value (e.g., 0, mean, median) if a selected feature
                # cannot be derived from form inputs. This is a HUGE assumption.
                # A better approach is to ensure form inputs can generate all selected features.
                processed_data[feature] = 0 # Placeholder - VERY LIKELY TO BE WRONG
                st.warning(f"Feature '{feature}' not derivable from form, defaulted to 0. Prediction accuracy will be affected.")

        # Fill any NaNs that might have resulted (e.g., from ratios with zero income)
        # Use medians from training if available, or a simple fillna(0) for this demo
        processed_data = processed_data.fillna(0)


        # 4. Scaling (use loaded scaler) - only on numeric columns among selected_features
        numeric_cols_to_scale = [col for col in self.selected_features if processed_data[col].dtype in [np.number] and col not in self.label_encoders.keys()] # Rough check
        if self.scaler and numeric_cols_to_scale:
            try:
                # Ensure columns are in the same order as during fit
                # This is tricky. Scaler expects all columns it was fit on.
                # For this demo, we assume the scaler was fit on self.selected_features or a subset
                # A more robust way is to save the list of scaled columns with the scaler.
                
                # Create a DataFrame with all selected features, then scale.
                # The scaler was likely fit on X_train which had selected_features.
                scaled_values = self.scaler.transform(processed_data[self.selected_features])
                processed_df_scaled = pd.DataFrame(scaled_values, columns=self.selected_features, index=processed_data.index)
                return processed_df_scaled[self.selected_features] # Ensure column order
            except Exception as e:
                st.error(f"Error during scaling: {e}. Using unscaled data for selected numeric features.")
                return processed_data[self.selected_features] # Fallback, but likely wrong
        
        return processed_data[self.selected_features] # Return only selected features in correct order

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
            # return # Optionally return if model is critical

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Customer Information")
            # These inputs need to map to features your model expects,
            # particularly those in `self.selected_features`.
            # This form is a simplification.
            with st.form("customer_input_form"):
                # Map to features model might use (examples)
                amt_income_total = st.number_input("Annual Income ($)", min_value=0, value=50000, key="income")
                amt_credit = st.number_input("Credit Amount ($)", min_value=0, value=200000, key="credit")
                # Model expects DAYS_BIRTH (negative days from today)
                age_years_form = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, key="age")
                days_birth = - (age_years_form * 365) # Convert to DAYS_BIRTH

                # Example: 'CODE_GENDER' might be a selected feature
                code_gender = st.selectbox("Gender", ["F", "M"], key="gender") # Assuming model uses 'F'/'M'

                # Example: 'NAME_EDUCATION_TYPE'
                education_options = list(self.label_encoders.get('NAME_EDUCATION_TYPE', pd.Series(dtype=object)).classes_) \
                                    if 'NAME_EDUCATION_TYPE' in self.label_encoders and hasattr(self.label_encoders['NAME_EDUCATION_TYPE'], 'classes_') \
                                    else ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"]
                name_education_type = st.selectbox("Education", education_options, key="education")

                # Example: 'DAYS_EMPLOYED' (negative days)
                employed_years_form = st.number_input("Years Employed", min_value=0, max_value=50, value=5, key="emp_years")
                days_employed = - (employed_years_form * 365) # Convert
                if days_employed == 0: days_employed = 365243 # Special value for unemployed if model uses it

                # Example: 'CNT_CHILDREN'
                cnt_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, key="children")

                # Add more inputs as needed to cover your `selected_features`
                # For EXT_SOURCE_X features, it's hard to get from a simple form.
                # They are often pre-calculated scores. For demo, we might have to default them.
                ext_source_1 = st.slider("External Source 1 (Normalized Score)", 0.0, 1.0, 0.5, 0.01, key="ext1", help="If model uses this feature directly")
                ext_source_2 = st.slider("External Source 2 (Normalized Score)", 0.0, 1.0, 0.5, 0.01, key="ext2")
                ext_source_3 = st.slider("External Source 3 (Normalized Score)", 0.0, 1.0, 0.5, 0.01, key="ext3")


                submitted = st.form_submit_button("🔍 Assess Risk")

            if submitted:
                if not self.model_loaded:
                    st.error("Cannot assess risk: Model is not loaded.")
                    return

                input_data = {
                    'AMT_INCOME_TOTAL': amt_income_total,
                    'AMT_CREDIT': amt_credit,
                    'DAYS_BIRTH': days_birth,
                    'CODE_GENDER': code_gender,
                    'NAME_EDUCATION_TYPE': name_education_type,
                    'DAYS_EMPLOYED': days_employed,
                    'CNT_CHILDREN': cnt_children,
                    'EXT_SOURCE_1': ext_source_1, # Assuming these are directly usable
                    'EXT_SOURCE_2': ext_source_2,
                    'EXT_SOURCE_3': ext_source_3,
                    # Add ALL other features your model's feature_engineering creates
                    # and are part of 'selected_features', or ensure _preprocess_input_data can derive them.
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
                        st.session_state.risk_assessment['factors'] = self.get_demo_risk_factors(input_data)

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

    def get_demo_risk_factors(self, raw_input_data):
        """Generates simplified demo risk factors based on raw input."""
        factors = []
        credit_income_ratio = (raw_input_data['AMT_CREDIT'] / (raw_input_data['AMT_INCOME_TOTAL'] + 1e-6))
        age_years = -raw_input_data['DAYS_BIRTH'] / 365
        employment_years = -raw_input_data['DAYS_EMPLOYED'] / 365 if raw_input_data['DAYS_EMPLOYED'] < 0 else 0

        if credit_income_ratio > 5:
            factors.append(("High Credit-to-Income Ratio", f"{credit_income_ratio:.1f}x", "High Risk"))
        if age_years < 25:
            factors.append(("Young Age", f"{int(age_years)} years", "Medium Risk"))
        if employment_years < 2:
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
                if self.explainer:
                    try:
                        customer_data_np = st.session_state.risk_assessment['processed_input_for_shap']
                        
                        # We need to ensure customer_data_np is a 2D array for some SHAP explainers
                        if customer_data_np.ndim == 1:
                            customer_data_np = customer_data_np.reshape(1, -1)
                        
                        # Create a DataFrame for SHAP, as TreeExplainer often works best with feature names
                        customer_df_for_shap = pd.DataFrame(customer_data_np, columns=self.selected_features)

                        # shap_values = self.explainer.shap_values(customer_data_np) # This might be for lgb model output (log-odds)
                        explanation_object = self.model_instance.explain_prediction(customer_df_for_shap.iloc[0].values) # using the class method

                        if explanation_object:
                            shap_values_for_class1 = explanation_object['shap_values']
                            expected_value_class1 = explanation_object['expected_value']
                            
                            st.write(f"SHAP Base Value (Average Model Output): {expected_value_class1:.4f}")

                            # Waterfall plot
                            fig_waterfall, ax_waterfall = plt.subplots() # Create matplotlib figure
                            shap.waterfall_plot(shap.Explanation(values=shap_values_for_class1,
                                                                 base_values=expected_value_class1,
                                                                 data=customer_df_for_shap.iloc[0].values,
                                                                 feature_names=self.selected_features),
                                                max_display=10, show=False)
                            st.pyplot(fig_waterfall)
                            plt.close(fig_waterfall) # Close plot to free memory

                            # Force plot
                            # shap.initjs() # Needed if not run in Jupyter
                            # force_plot = shap.force_plot(expected_value_class1,
                            #                              shap_values_for_class1,
                            #                              customer_df_for_shap.iloc[0],
                            #                              matplotlib=False) # Use matplotlib=True for static image
                            # # For Streamlit, saving to HTML and displaying as component is more robust for interactive plots
                            # shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
                            # st.components.v1.html(shap_html, height=200, scrolling=True)
                            st.info("Force plot display can be complex in Streamlit. Waterfall plot shown above.")


                        else:
                            st.error("Could not generate SHAP explanation object.")
                    except Exception as e:
                        st.error(f"Error generating SHAP plot: {e}")
                else:
                    st.info("SHAP explainer not available (was not loaded with the model).")
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

        # Example: Extracting AUC from the model if it was stored (e.g. as an attribute)
        # This is a placeholder, adapt to how your model stores this.
        auc_lgb = self.model_instance.models.get('lgb_auc', 0.76) # Example
        auc_xgb = self.model_instance.models.get('xgb_auc', 0.75)
        auc_ensemble = self.model_instance.models.get('ensemble_auc', 0.7683) # From your last run log

        col1, col2, col3 = st.columns(3)
        col1.metric("LGBM AUC", f"{auc_lgb:.4f}")
        col2.metric("XGBoost AUC", f"{auc_xgb:.4f}")
        col3.metric("Ensemble AUC", f"{auc_ensemble:.4f}")

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