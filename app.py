import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Loan Classification App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🏦 Loan Status Classification System")
st.markdown("""
This application demonstrates machine learning classification models for predicting loan approval status.
Upload test data and select a model to see predictions and evaluation metrics.
""")

# ============================================================================
# SIDEBAR - Configuration
# ============================================================================

# Model selection
st.sidebar.markdown("<h3 style='text-align: center;'><b>Select ML Model</b></h3>", unsafe_allow_html=True)
model_name = st.sidebar.selectbox(
    "",
    [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor",
        "Naive Bayes (Gaussian)",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with test data",
    type=["csv"],
    help="Upload a CSV file containing loan data for predictions"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Sample Test Dataset")
st.sidebar.markdown(
    "Download the sample test data: [loan_data_test.csv](https://github.com/rdeepika06/mlassignment2/blob/main/model/loan_data_test.csv)"
)
st.sidebar.caption("Use this file to test model predictions and evaluate metrics.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**Note:** This app uses the loan dataset with 13 features. "
    "Ensure your CSV matches the required format."
)

# ============================================================================
# MAIN CONTENT - Load and preprocess data
# ============================================================================

@st.cache_resource
def load_and_prepare_data(csv_path):
    """Load original dataset for model training"""
    try:
        data = pd.read_csv(csv_path)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data, fit_encoders=False, encoders=None, scaler=None):
    """Preprocess the data"""
    data_processed = data.copy()
    
    # Identify categorical and numeric columns
    categorical_cols = data_processed.select_dtypes(include='object').columns
    numeric_cols = data_processed.select_dtypes(include=['int64', 'float64']).columns
    
    if fit_encoders:
        encoders_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data_processed[col] = le.fit_transform(data_processed[col].astype(str))
            encoders_dict[col] = le
        
        # Scale features
        scaler_obj = MinMaxScaler()
        columns = data_processed.columns
        data_processed[columns] = scaler_obj.fit_transform(data_processed[columns])
        
        return data_processed, encoders_dict, scaler_obj
    else:
        for col in categorical_cols:
            if col in encoders:
                data_processed[col] = encoders[col].transform(data_processed[col].astype(str))
        
        columns = data_processed.columns
        data_processed[columns] = scaler.transform(data_processed[columns])
        
        return data_processed, encoders, scaler

@st.cache_resource
def train_models(X_train, y_train):
    """Train all 6 models"""
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=10),
        "K-Nearest Neighbor": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost (Ensemble)": xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def calculate_metrics(y_true, y_pred, model):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_true, y_pred_proba)
    except:
        auc_score = roc_auc_score(y_true, y_pred)
    
    return {
        "Accuracy": accuracy,
        "AUC Score": auc_score,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc
    }

# ============================================================================
# LOAD DATASET
# ============================================================================

try:
    # Load the original dataset
    original_data = load_and_prepare_data("model/loan_data.csv")
    
    if original_data is not None:
        # Separate features and target
        y = original_data['loan_status']
        X = original_data.drop('loan_status', axis=1)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocess training data
        X_train_processed, encoders, scaler = preprocess_data(X_train, fit_encoders=True)
        X_test_processed, _, _ = preprocess_data(X_test, fit_encoders=False, encoders=encoders, scaler=scaler)
        
        # Train all models
        with st.spinner("🔄 Training models..."):
            trained_models = train_models(X_train_processed, y_train)
        
        st.success("✅ All models trained successfully!")
        
        # ====================================================================
        # DISPLAY DATASET INFORMATION - Creative Display
        # ====================================================================
        st.markdown("### Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(original_data))
        with col2:
            st.metric("Training Samples", len(X_train))
        with col3:
            st.metric("Test Samples", len(X_test))
        with col4:
            st.metric("Features", X.shape[1])
        
        # ====================================================================
        # SHOW MESSAGE TO UPLOAD TEST DATA
        # ====================================================================
        st.markdown("---")
        st.info("**Upload a CSV file in the sidebar to view evaluation metrics, confusion matrix, and predictions for Test Data**")
        
        # ====================================================================
        # PROCESS UPLOADED DATA AND SHOW METRICS
        # ====================================================================
        if uploaded_file is not None:
            try:
                # Read uploaded file
                test_data = pd.read_csv(uploaded_file)
                st.success("✅ Test data uploaded successfully!")
                
                st.subheader("Uploaded Data Preview:")
                st.dataframe(test_data.head(10), use_container_width=True)
                
                # Check if target column exists
                if 'loan_status' in test_data.columns:
                    test_data_features = test_data.drop('loan_status', axis=1)
                    test_data_target = test_data['loan_status']
                    has_target = True
                else:
                    test_data_features = test_data
                    test_data_target = None
                    has_target = False
                
                # Preprocess test data
                test_data_processed, _, _ = preprocess_data(
                    test_data_features,
                    fit_encoders=False,
                    encoders=encoders,
                    scaler=scaler
                )
                
                # Get selected model
                selected_model = trained_models[model_name]
                
                # Make predictions
                predictions = selected_model.predict(test_data_processed)
                
                # Get prediction probabilities if available
                try:
                    probabilities = selected_model.predict_proba(test_data_processed)
                    prob_approved = probabilities[:, 1]
                except:
                    prob_approved = None
                
                # ====================================================================
                # EVALUATION METRICS ON UPLOADED DATA
                # ====================================================================
                st.markdown("---")
                st.header("Model Evaluation Metrics For Uploaded Data")
                
                if has_target:
                    # Calculate metrics
                    metrics = calculate_metrics(test_data_target, predictions, selected_model)
                    
                    # Display metrics in columns
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}", delta=f"{metrics['Accuracy']*100:.2f}%")
                    with col2:
                        st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                    with col3:
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    with col4:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    with col5:
                        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                    with col6:
                        st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
                else:
                    st.warning("⚠️ Target column (loan_status) not found. Metrics cannot be calculated. Only predictions will be shown.")
                
                # ====================================================================
                # PREDICTIONS DISPLAY
                # ====================================================================
                st.markdown("---")
                st.header("Predictions")
                
                # Create results dataframe
                results_df = test_data.copy()
                results_df['Prediction'] = predictions
                results_df['Prediction_Label'] = results_df['Prediction'].apply(
                    lambda x: '✅ Approved' if x == 1 else '❌ Rejected'
                )
                
                if prob_approved is not None:
                    results_df['Approval_Probability'] = prob_approved
                
                st.subheader(f"Predictions using {model_name}:")
                st.dataframe(results_df, use_container_width=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{model_name.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # ====================================================================
                # CONFUSION MATRIX AND CLASSIFICATION REPORT
                # ====================================================================
                if has_target:
                    st.markdown("---")
                    st.header("Confusion Matrix & Classification Report")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(test_data_target, predictions)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {model_name}')
                        ax.set_xticklabels(['Rejected', 'Approved'])
                        ax.set_yticklabels(['Rejected', 'Approved'])
                        
                        # Add text annotations
                        tn, fp, fn, tp = cm.ravel()
                        st.pyplot(fig)
                        
                        st.markdown(f"""
                        **Matrix Breakdown:**
                        - True Negatives (TN): {tn}
                        - False Positives (FP): {fp}
                        - False Negatives (FN): {fn}
                        - True Positives (TP): {tp}
                        """)
                    
                    with col2:
                        st.subheader("Classification Report")
                        report = classification_report(test_data_target, predictions, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                        
                        # Text report
                        st.markdown("**Detailed Report:**")
                        st.text(classification_report(test_data_target, predictions))
                
                # Display prediction statistics
                st.markdown("---")
                st.header("Prediction Statistics")
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    st.metric("Total Predictions", len(predictions))
                with pred_col2:
                    st.metric("Approved (1)", int(predictions.sum()))
                with pred_col3:
                    st.metric("Rejected (0)", int(len(predictions) - predictions.sum()))
            
            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {e}")
        
        # ====================================================================
        # ALL MODELS COMPARISON - Evaluation Metrics & Overall Confusion Matrix
        # ====================================================================
        st.markdown("---")
        st.header("All Models Comparison")
        
        # Calculate metrics for all models on built-in test set
        all_models_metrics = {}
        all_models_predictions = {}
        
        for model_key, model_obj in trained_models.items():
            pred = model_obj.predict(X_test_processed)
            all_models_predictions[model_key] = pred
            all_models_metrics[model_key] = calculate_metrics(y_test, pred, model_obj)
        
        # Display metrics comparison table with bold headers and model names
        st.subheader("Evaluation Metrics for All Models (Built-in Test Set)")
        metrics_df = pd.DataFrame(all_models_metrics).T
        metrics_df = metrics_df.round(4)
        
        # Create HTML table with bold headers and model names
        html_table = "<table style='width:100%; border-collapse: collapse;'>"
        html_table += "<tr style='background-color: #f0f0f0;'>"
        html_table += "<th style='padding: 10px; border: 1px solid #ddd; text-align: left;'><b>Model Name</b></th>"
        for col in metrics_df.columns:
            html_table += f"<th style='padding: 10px; border: 1px solid #ddd; text-align: center;'><b>{col}</b></th>"
        html_table += "</tr>"
        
        for idx, row in metrics_df.iterrows():
            html_table += "<tr>"
            html_table += f"<td style='padding: 10px; border: 1px solid #ddd;'><b>{idx}</b></td>"
            for val in row:
                html_table += f"<td style='padding: 10px; border: 1px solid #ddd; text-align: center;'>{val:.4f}</td>"
            html_table += "</tr>"
        
        html_table += "</table>"
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Calculate ensemble predictions (majority voting)
        ensemble_predictions_array = np.array([all_models_predictions[key] for key in all_models_predictions.keys()])
        ensemble_predictions = (ensemble_predictions_array.sum(axis=0) >= 3).astype(int)  # Majority vote (3+ out of 6)
        
        # Calculate ensemble metrics
        ensemble_metrics = calculate_metrics(y_test, ensemble_predictions, trained_models[model_name])
        
        # Display overall confusion matrix
        st.subheader("Overall Confusion Matrix (Ensemble Voting)")
        
        col_cm_left, col_cm_right = st.columns(2)
        
        with col_cm_left:
            cm_ensemble = confusion_matrix(y_test, ensemble_predictions)
            
            fig, ax = plt.subplots(figsize=(7, 6))
            sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Ensemble Voting (Majority Vote from All 6 Models)')
            ax.set_xticklabels(['Rejected', 'Approved'])
            ax.set_yticklabels(['Rejected', 'Approved'])
            st.pyplot(fig)
        
        with col_cm_right:
            st.markdown("**Ensemble Voting Metrics:**")
            st.metric("Accuracy", f"{ensemble_metrics['Accuracy']:.4f}")
            st.metric("AUC Score", f"{ensemble_metrics['AUC Score']:.4f}")
            st.metric("Precision", f"{ensemble_metrics['Precision']:.4f}")
            st.metric("Recall", f"{ensemble_metrics['Recall']:.4f}")
            st.metric("F1 Score", f"{ensemble_metrics['F1 Score']:.4f}")
            st.metric("MCC Score", f"{ensemble_metrics['MCC Score']:.4f}")
            
            tn, fp, fn, tp = cm_ensemble.ravel()
            st.markdown(f"""
            **Matrix Breakdown:**
            - True Negatives (TN): {tn}
            - False Positives (FP): {fp}
            - False Negatives (FN): {fn}
            - True Positives (TP): {tp}
            """)
        
        # ====================================================================
        # FOOTER
        # ====================================================================
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>🏦 Loan Status Classification System | Built with Streamlit & Scikit-learn</p>
            <p>Dataset: 45,002 loan applications | Models: 6 Classification Algorithms</p>
        </div>
        """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ Dataset file not found. Please ensure 'model/loan_data.csv' exists in the project directory.")
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
