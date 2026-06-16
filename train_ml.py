import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
import pickle
import optuna  # For hyperparameter optimization
from functools import partial
import xgboost as xgb
from catboost import CatBoost, Pool, CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score
import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import math
import matplotlib.pyplot as plt

# Configure matplotlib for proper display
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# --- Constants and Configuration ---
TRAIN_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/train.csv'
TEST_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/test.csv'
SUBMISSION_CSV_PATH = r'submission_ml.csv'
MODEL_DIR = 'models'  # Directory to save model checkpoints
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_ml.pkl')  # Path for best model

# Hyperparameters (will be tuned or set via args)
DEFAULT_VAL_SPLIT = 0.3
DEFAULT_N_TRIALS = 100 # Number of Optuna trials
DEFAULT_MODEL_TYPE = 'xgboost'  # Default model type

# Feature lists based on notebook analysis
NUMERICAL_COLS = [
    'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Dates', 'Recency',
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
    'MntSweetProducts', 'MntGoldProds', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumDealsPurchases', 'NumWebVisitsMonth', 'AcceptedCmp1',
    'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Complain'
]
EDU_COL = 'Education'
MARITAL_COL = 'Marital_Status'
TARGET_COL = 'Target'
DATE_COL = 'Dt_Customer'
ID_COL = 'ID'  # ID column for submission file

# --- Notebook Parameters (replace argparse) ---
# These can be modified directly in the notebook
mode = 'train'  # Options: 'train', 'tune', 'predict', 'ensemble'
model_type = 'catboost'  # Options: 'xgboost', 'catboost', 'lightgbm', 'ensemble'
val_split = DEFAULT_VAL_SPLIT
data_path = TRAIN_CSV_PATH
test_path = TEST_CSV_PATH
submission_path = SUBMISSION_CSV_PATH
n_trials = DEFAULT_N_TRIALS
model_path = BEST_MODEL_PATH
feature_engineering = True
timeout = 7200
seed = 42
use_smote = True
use_cv = False  # Changed from True to False
cv_folds = 5
threshold = 0.5
hyperparameter_tuning = False

# Set random seed for reproducibility
np.random.seed(seed)

# --- Data Preprocessing and Model Functions ---

def preprocess_data(df, add_features=True, reference_columns=None):
    """Applies preprocessing steps and feature engineering to prepare data for ML models.
    
    Args:
        df: The input DataFrame to preprocess
        add_features: Whether to add engineered features
        reference_columns: Optional list of column names to ensure consistent feature order
    """
    df_processed = df.copy()

    # Keep the ID column for later if it exists
    id_values = None
    if ID_COL in df_processed.columns:
        id_values = df_processed[ID_COL].copy()

    # 1. Handle Date Feature ('Dt_Customer')
    try:
        df_processed['Dt_Customer_1'] = pd.to_datetime(df_processed[DATE_COL], format='mixed', errors='coerce')
        min_date = df_processed['Dt_Customer_1'].min()
        df_processed['Dates'] = (df_processed['Dt_Customer_1'] - min_date).dt.days
        
        # Extract additional date features if add_features is True
        if add_features:
            df_processed['Dt_Year'] = df_processed['Dt_Customer_1'].dt.year
            df_processed['Dt_Month'] = df_processed['Dt_Customer_1'].dt.month
            df_processed['Dt_Day'] = df_processed['Dt_Customer_1'].dt.day
            df_processed['Dt_DayOfWeek'] = df_processed['Dt_Customer_1'].dt.dayofweek
            df_processed['Dt_DayOfYear'] = df_processed['Dt_Customer_1'].dt.dayofyear
            df_processed['Dt_Quarter'] = df_processed['Dt_Customer_1'].dt.quarter
            df_processed['Dt_Is_Weekend'] = (df_processed['Dt_Customer_1'].dt.dayofweek >= 5).astype(int)
            df_processed['Dt_Is_Month_Start'] = df_processed['Dt_Customer_1'].dt.is_month_start.astype(int)
            df_processed['Dt_Is_Month_End'] = df_processed['Dt_Customer_1'].dt.is_month_end.astype(int)
            
            # Drop the temporary datetime column
            df_processed.drop('Dt_Customer_1', axis=1, inplace=True)
    except Exception as e:
        print(f"Error processing date column {DATE_COL}: {e}")
        # Handle cases where date conversion might fail entirely
        df_processed['Dates'] = 0  # Simple fallback

    # Check for NaNs introduced by date conversion errors
    if df_processed['Dates'].isnull().any():
        print(f"Warning: NaNs found in 'Dates' column after conversion. Filling with 0.")
        df_processed['Dates'].fillna(0, inplace=True)

    # 2. Feature Engineering (if enabled)
    if add_features:
        # Customer Age from Year_Birth
        if 'Year_Birth' in df_processed.columns:
            current_year = datetime.datetime.now().year
            df_processed['Age'] = current_year - df_processed['Year_Birth']
            df_processed['Age_Group'] = pd.cut(df_processed['Age'], 
                                             bins=[0, 30, 40, 50, 60, 100], 
                                             labels=['<30', '30-40', '40-50', '50-60', '60+'])
            
            # Age squared (non-linear relationships)
            df_processed['Age_Squared'] = df_processed['Age'] ** 2
            
            # Convert Age_Group to one-hot encoding later
        
        # Total children
        if 'Kidhome' in df_processed.columns and 'Teenhome' in df_processed.columns:
            df_processed['Total_Children'] = df_processed['Kidhome'] + df_processed['Teenhome']
            df_processed['Has_Children'] = (df_processed['Total_Children'] > 0).astype(int)
            df_processed['Has_Kid'] = (df_processed['Kidhome'] > 0).astype(int)
            df_processed['Has_Teen'] = (df_processed['Teenhome'] > 0).astype(int)
            df_processed['Child_Ratio'] = df_processed['Kidhome'] / (df_processed['Total_Children'] + 0.001)
        
        # Total purchases across channels
        purchase_cols = [col for col in df_processed.columns if col.startswith('Num') and col.endswith('Purchases')]
        if len(purchase_cols) > 0:
            df_processed['Total_Purchases'] = df_processed[purchase_cols].sum(axis=1)
            
            # Purchase channel ratios
            for col in purchase_cols:
                ratio_col = f"{col}_Ratio"
                df_processed[ratio_col] = df_processed[col] / (df_processed['Total_Purchases'] + 0.001)
                
            # Purchase diversity metric (unique channels used)
            df_processed['Purchase_Channels_Used'] = (df_processed[purchase_cols] > 0).sum(axis=1)
            
            # Web to store purchase ratio
            if 'NumWebPurchases' in df_processed.columns and 'NumStorePurchases' in df_processed.columns:
                df_processed['Web_Store_Ratio'] = df_processed['NumWebPurchases'] / (df_processed['NumStorePurchases'] + 0.001)
        
        # Total spent across categories
        spent_cols = [col for col in df_processed.columns if col.startswith('Mnt')]
        if len(spent_cols) > 0:
            df_processed['Total_Spent'] = df_processed[spent_cols].sum(axis=1)
            
            # Spending ratios (percentage of total spent on each category)
            for col in spent_cols:
                ratio_col = f"{col}_Ratio"
                df_processed[ratio_col] = df_processed[col] / (df_processed['Total_Spent'] + 0.001)
                
            # Spending diversity metric (number of categories with non-zero spending)
            df_processed['Spending_Categories_Used'] = (df_processed[spent_cols] > 0).sum(axis=1)
                
            # Average spending per purchase
            if 'Total_Purchases' in df_processed.columns:
                df_processed['Avg_Spent_Per_Purchase'] = df_processed['Total_Spent'] / (df_processed['Total_Purchases'] + 0.001)
                
            # Log transformations for spending amounts (often better for skewed data)
            for col in spent_cols:
                log_col = f"{col}_Log"
                df_processed[log_col] = np.log1p(df_processed[col])
                
            # Total spent squared (non-linear relationships)
            df_processed['Total_Spent_Squared'] = df_processed['Total_Spent'] ** 2
        
        # Campaign response rate and patterns
        campaign_cols = [col for col in df_processed.columns if col.startswith('Accepted')]
        if len(campaign_cols) > 0:
            df_processed['Campaign_Response_Rate'] = df_processed[campaign_cols].sum(axis=1) / len(campaign_cols)
            df_processed['Campaign_Response_Count'] = df_processed[campaign_cols].sum(axis=1)
            
            # Create last campaign acceptance feature
            last_campaign_col = max(campaign_cols, key=lambda x: int(x.replace('AcceptedCmp', '')))
            df_processed['Last_Campaign_Accepted'] = df_processed[last_campaign_col]
            
            # Calculate response rate excluding last campaign
            previous_campaigns = [col for col in campaign_cols if col != last_campaign_col]
            if previous_campaigns:
                df_processed['Previous_Response_Rate'] = df_processed[previous_campaigns].sum(axis=1) / len(previous_campaigns)
        
        # Income features with more sophistication
        if 'Income' in df_processed.columns:
            df_processed['Income_Log'] = np.log1p(df_processed['Income'])
            df_processed['Income_Sqrt'] = np.sqrt(df_processed['Income'] + 0.001)
            
            if 'Total_Children' in df_processed.columns:
                df_processed['Income_Per_Child'] = df_processed['Income'] / (df_processed['Total_Children'] + 1)
                
            if 'Age' in df_processed.columns:
                df_processed['Income_Per_Age'] = df_processed['Income'] / (df_processed['Age'] + 0.001)
                
            if 'Total_Spent' in df_processed.columns:
                df_processed['Income_Spent_Ratio'] = df_processed['Total_Spent'] / (df_processed['Income'] + 0.001)
                
        # Recency features
        if 'Recency' in df_processed.columns:
            df_processed['Recency_Squared'] = df_processed['Recency'] ** 2
            df_processed['Recency_Log'] = np.log1p(df_processed['Recency'])
            
            # Recency to purchase ratio
            if 'Total_Purchases' in df_processed.columns:
                df_processed['Recency_Purchase_Ratio'] = df_processed['Recency'] / (df_processed['Total_Purchases'] + 0.001)
                
        # Web visits features
        if 'NumWebVisitsMonth' in df_processed.columns:
            df_processed['NumWebVisitsMonth_Log'] = np.log1p(df_processed['NumWebVisitsMonth'])
            
            # Web visits to web purchases ratio (conversion rate)
            if 'NumWebPurchases' in df_processed.columns:
                df_processed['Web_Conversion_Rate'] = df_processed['NumWebPurchases'] / (df_processed['NumWebVisitsMonth'] + 0.001)
        
        # Add interaction features between important variables
        # Based on top 5 important features from the model output
        if all(col in df_processed.columns for col in ['Recency', 'Campaign_Response_Rate', 'NumWebPurchases']):
            df_processed['Recency_X_Campaign'] = df_processed['Recency'] * df_processed['Campaign_Response_Rate']
            df_processed['Recency_X_WebPurchases'] = df_processed['Recency'] * df_processed['NumWebPurchases']
            df_processed['Campaign_X_WebPurchases'] = df_processed['Campaign_Response_Rate'] * df_processed['NumWebPurchases']
            
        # Additional marital status and education derived features
        # Segment combinations that might be meaningful
        df_processed['Edu_Marital_Combined'] = df_processed[EDU_COL].astype(str) + "_" + df_processed[MARITAL_COL].astype(str)
    
    # 3. Handle Categorical Features with one-hot encoding
    # Get dummies for education and marital status
    edu_dummies = pd.get_dummies(df_processed[EDU_COL].fillna('Unknown'), prefix=EDU_COL)
    marital_dummies = pd.get_dummies(df_processed[MARITAL_COL].fillna('Unknown'), prefix=MARITAL_COL)
    
    # If we created Age_Group, also get dummies for it
    age_group_dummies = None
    if add_features and 'Age_Group' in df_processed.columns:
        age_group_dummies = pd.get_dummies(df_processed['Age_Group'].fillna('<30'), prefix='Age_Group')
    
    # Handle combined education and marital status
    edu_marital_dummies = None
    if add_features and 'Edu_Marital_Combined' in df_processed.columns:
        edu_marital_dummies = pd.get_dummies(df_processed['Edu_Marital_Combined'].fillna('Unknown_Unknown'), 
                                            prefix='Edu_Marital')
    
    # Update numerical columns to include new engineered features
    engineered_numerical_cols = NUMERICAL_COLS.copy()
    if add_features:
        for col in df_processed.columns:
            if col not in NUMERICAL_COLS and col not in [EDU_COL, MARITAL_COL, 'Age_Group', 'Edu_Marital_Combined',
                                                       TARGET_COL, ID_COL, DATE_COL, 'Dt_Customer_1']:
                if df_processed[col].dtype in ['int64', 'float64']:
                    engineered_numerical_cols.append(col)
    
    # Combine numerical columns with dummy columns
    all_columns = df_processed[engineered_numerical_cols].copy()
    all_columns = pd.concat([all_columns, edu_dummies, marital_dummies], axis=1)
    
    # Add age group dummies if they exist
    if age_group_dummies is not None:
        all_columns = pd.concat([all_columns, age_group_dummies], axis=1)
        
    # Add education-marital combined dummies if they exist
    if edu_marital_dummies is not None:
        all_columns = pd.concat([all_columns, edu_marital_dummies], axis=1)
    
    # Add target column if it exists
    if TARGET_COL in df_processed.columns:
        all_columns[TARGET_COL] = df_processed[TARGET_COL]
    
    # Handle NaNs in numerical columns
    all_columns[engineered_numerical_cols] = all_columns[engineered_numerical_cols].fillna(all_columns[engineered_numerical_cols].median())
    
    # If reference columns are provided, ensure consistent feature order
    if reference_columns is not None:
        # First ensure all reference columns exist in our dataframe
        for col in reference_columns:
            if col not in all_columns.columns and col != TARGET_COL:
                print(f"Adding missing column {col} with zeros")
                all_columns[col] = 0
                
        # Now reorder the columns to match reference order
        # Keep only the columns that are in reference_columns
        common_cols = [col for col in reference_columns if col in all_columns.columns]
        all_columns = all_columns[common_cols]
    
    return all_columns, id_values

# --- Model Training and Hyperparameter Tuning ---

def train_xgboost_model(X_train, y_train, X_val, y_val, params=None):
    """Train an XGBoost model with given parameters."""
    # Default parameters if none provided
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'alpha': 0.1,
            'lambda': 1.0,
            'scale_pos_weight': 1.0,
            'tree_method': 'hist',
            'grow_policy': 'lossguide',
            'max_leaves': 64,
            'max_bin': 256
        }
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set up evaluation list
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    # Number of training rounds
    num_boost_round = params.pop('num_boost_round', 2000) if 'num_boost_round' in params else 2000
    early_stopping = params.pop('early_stopping', 50) if 'early_stopping' in params else 50
    
    # Train the model
    model = xgb.train(params, 
                     dtrain, 
                     num_boost_round=num_boost_round,
                     evals=evallist,
                     early_stopping_rounds=early_stopping,
                     verbose_eval=100)
    
    # Evaluate on validation set
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, y_pred)
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Calculate and print additional metrics
    try:
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"Validation AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
    
    print(classification_report(y_val, y_pred))
    
    # Feature importance analysis
    if model.booster in ['gbtree', 'dart']:
        print("\nTop 10 Feature Importance (Weight):")
        importance = model.get_score(importance_type='weight')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, score in sorted_importance[:10]:
            print(f"  {feature}: {score}")
    
    return model, val_accuracy

def train_catboost_model(X_train, y_train, X_val, y_val, params=None, use_cv=False, cv_folds=5):
    """Train a CatBoost model with given parameters."""
    # Default parameters if none provided
    if params is None:
        params = {
            'iterations': 5000,  # Increased maximum iterations
            'learning_rate': 0.02,  # Lower initial learning rate for better convergence
            'depth': 6,  # Reduced depth (was 8)
            'l2_leaf_reg': 10.0,  # Increased L2 regularization (was 3)
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 0.2,
            'random_strength': 1.0,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'leaf_estimation_method': 'Newton',
            'grow_policy': 'Lossguide',
            'min_data_in_leaf': 10,
            'max_leaves': 64,
            'border_count': 254,
            'feature_border_type': 'UniformAndQuantiles',
            'rsm': 0.9,  # Random subspace method
            'boosting_type': 'Plain',
            'verbose': 100,
            'task_type': 'CPU',  # Change to 'GPU' if available
            'use_best_model': True,
            'auto_class_weights': 'Balanced',  # Automatically balance class weights
            # Add train_dir to save logs and plots
            'train_dir': 'catboost_info'
        }
    
    # Create output directory for CatBoost logs and plots
    os.makedirs('catboost_info', exist_ok=True)
    
    # Check for GPU
    try:
        import cupy
        params['task_type'] = 'GPU'
        print("GPU detected, using GPU acceleration for CatBoost")
    except ImportError:
        print("GPU libraries not found, using CPU for CatBoost")
        
    # Helper function to display and save plots
    def display_model_plots(model, fold_name=None):
        try:
            # Create folder for plots
            plots_dir = os.path.join('catboost_info', 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot metrics vs iterations
            plt.figure(figsize=(14, 10))
            
            # Get metrics history
            train_loss = model.get_evals_result()['learn']['Logloss']
            val_loss = model.get_evals_result()['validation']['Logloss']
            
            # Plot loss
            plt.subplot(2, 1, 1)
            plt.plot(train_loss, label='Train Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Loss vs Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Plot AUC if available
            if 'AUC' in model.get_evals_result()['learn']:
                plt.subplot(2, 1, 2)
                train_auc = model.get_evals_result()['learn']['AUC']
                val_auc = model.get_evals_result()['validation']['AUC'] 
                plt.plot(train_auc, label='Train AUC')
                plt.plot(val_auc, label='Validation AUC')
                plt.title('AUC vs Iterations')
                plt.xlabel('Iterations')
                plt.ylabel('AUC')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"catboost_training_{'_' + fold_name if fold_name else ''}.png"
            plt.savefig(os.path.join(plots_dir, plot_filename))
            
            # Display in interactive environment
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(12, 8))
            feature_importances = model.get_feature_importance()
            feature_names = model.feature_names_
            
            # Sort by importance
            indices = np.argsort(feature_importances)[-20:]  # Top 20 features
            plt.barh(range(len(indices)), [feature_importances[i] for i in indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.title('Top 20 Feature Importances')
            
            # Save importance plot
            importance_filename = f"feature_importance_{'_' + fold_name if fold_name else ''}.png"
            plt.savefig(os.path.join(plots_dir, importance_filename))
            
            # Display in interactive environment
            plt.show()
            
            print(f"Plots saved to {plots_dir}")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    # Feature preprocessing and selection
    print("Performing feature analysis for enhanced performance...")
    
    # Check for highly correlated features
    correlation_threshold = 0.95
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    
    if highly_correlated:
        print(f"Removing {len(highly_correlated)} highly correlated features...")
        X_train = X_train.drop(columns=highly_correlated)
        X_val = X_val.drop(columns=highly_correlated)
        print(f"Reduced feature set: {X_train.shape[1]} features")
    
    # Advanced class imbalance handling
    class_counts = np.bincount(y_train)
    if len(class_counts) > 1 and min(class_counts) / max(class_counts) < 0.2:
        print("Significant class imbalance detected. Applying advanced balancing techniques...")
        
        try:
            # Apply SMOTE for class balancing
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"After resampling - X shape: {X_train_resampled.shape}, y distribution: {np.bincount(y_train_resampled)}")
        except Exception as e:
            print(f"Resampling failed, using original data: {e}")
            X_train_resampled, y_train_resampled = X_train, y_train
    else:
        print("Class balance acceptable, using original data")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # Create CatBoost Pool objects for single model training
    train_pool = Pool(X_train_resampled, y_train_resampled)
    val_pool = Pool(X_val, y_val)
    
    # Initialize model with train_dir for plot saving
    single_params = params.copy()
    single_params['train_dir'] = os.path.join('catboost_info', 'single_model')
    os.makedirs(single_params['train_dir'], exist_ok=True)
    
    # Initialize model
    model = CatBoostClassifier(**single_params)
    
    # Train model
    model.fit(train_pool, 
              eval_set=val_pool,
              verbose=100,
              plot=False)  # Disable built-in plotting to use our custom plotting
    
    # Create and display plots
    display_model_plots(model, "single_model")
    
    # Evaluate on validation set
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"Model - Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    
    # Detailed evaluation on validation set
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        # Find optimal threshold
        print("\nFinding optimal classification threshold...")
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba > threshold).astype(int)
            f1 = f1_score(y_val, y_pred_threshold)
            
            print(f"Threshold {threshold:.2f} - F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
        y_pred = (y_pred_proba > best_threshold).astype(int)
    else:
        y_pred = model.predict(X_val)
        y_pred_proba = None
    
    val_accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
    
    # Calculate and print additional metrics
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
            print(f"Final Validation AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate AUC: {e}")
    
    print(classification_report(y_val, y_pred))
    
    # Feature importance analysis if available
    if hasattr(model, 'feature_importances_') or (hasattr(model, 'get_feature_importance') and callable(model.get_feature_importance)):
        print("\nTop 10 Feature Importance:")
        
        if hasattr(model, 'get_feature_importance') and callable(model.get_feature_importance):
            feature_importance = model.get_feature_importance()
        else:
            feature_importance = model.feature_importances_
            
        feature_names = X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']}")
    
    # Store best threshold as model attribute
    if hasattr(model, 'predict_proba') and y_pred_proba is not None:
        model.best_threshold = best_threshold
        
        # Monkey patch predict method to use best threshold
        original_predict = model.predict
        def predict_with_threshold(X, *args, **kwargs):
            if hasattr(model, 'predict_proba'):
                return (model.predict_proba(X)[:, 1] > model.best_threshold).astype(int)
            else:
                return original_predict(X, *args, **kwargs)
        
        model.original_predict = original_predict
        model.predict = predict_with_threshold
    
    return model, val_accuracy

def tune_model_hyperparameters(X_train, y_train, X_val, y_val, model_type='catboost', n_trials=20, timeout=3600):
    """Tune model hyperparameters using Optuna and return the best model."""
    print(f"\nTuning hyperparameters for {model_type} model with {n_trials} trials...")
    
    # Create Optuna study
    study_name = f"{model_type}_hyperparameter_optimization"
    storage_name = f"sqlite:///{study_name}.db"
    
    try:
        # Create or load study with appropriate direction (maximize accuracy)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize"
        )
    except Exception as e:
        print(f"Error creating Optuna study: {e}. Creating without storage.")
        study = optuna.create_study(direction="maximize")
    
    # Set objective function based on model type
    if model_type == 'xgboost':
        # Use partial to fix non-changing arguments
        objective = partial(xgboost_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    elif model_type == 'catboost':
        objective = partial(catboost_objective, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    else:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")
    
    # Run optimization with timeout
    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        print("Optimization interrupted.")
    except Exception as e:
        print(f"Error during optimization: {e}")
    
    # Print optimization results
    print("\nHyperparameter optimization results:")
    print(f"Number of completed trials: {len(study.trials)}")
    
    if len(study.trials) > 0:
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.4f}")
        
        # Print AUC if available
        if 'auc' in study.best_trial.user_attrs:
            best_auc = study.best_trial.user_attrs['auc']
            print(f"Best AUC: {best_auc:.4f}")
        
        print("\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Save best parameters to a file
        best_params = study.best_trial.params
        best_params_path = os.path.join(MODEL_DIR, f"best_{model_type}_params.json")
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        
        try:
            import json
            with open(best_params_path, 'w') as f:
                json.dump(best_params, f, indent=4)
            print(f"Best parameters saved to {best_params_path}")
        except Exception as e:
            print(f"Error saving best parameters: {e}")
        
        # Train final model with best parameters
        print("\nTraining final model with best parameters...")
        
        if model_type == 'xgboost':
            # Process parameters for XGBoost
            best_params_copy = best_params.copy()
            num_boost_round = best_params_copy.pop('num_boost_round', 2000)
            early_stopping = best_params_copy.pop('early_stopping', 50)
            
            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Train model with best parameters
            best_model = xgb.train(
                best_params_copy,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'validation')],
                early_stopping_rounds=early_stopping,
                verbose_eval=100
            )
            
            # Evaluate best model
            y_pred_proba = best_model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)
            best_accuracy = accuracy_score(y_val, y_pred)
            
        elif model_type == 'catboost':
            # Add CV option for CatBoost
            use_cv = best_params.pop('use_cv', True) if 'use_cv' in best_params else True
            cv_folds = best_params.pop('cv_folds', 5) if 'cv_folds' in best_params else 5
            
            # Train CatBoost with best parameters
            best_model, best_accuracy = train_catboost_model(
                X_train, y_train, X_val, y_val,
                params=best_params,
                use_cv=use_cv,
                cv_folds=cv_folds
            )
        
        print(f"\nFinal model accuracy with best parameters: {best_accuracy:.4f}")
        return best_model, best_params
    
    else:
        print("No trials were completed successfully.")
        return None, None

def save_model(model, model_path):
    """Save the trained model to a file."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(model_path):
    """Load a trained model from a file."""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# --- Optuna Objective Functions ---

def xgboost_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost with expanded hyperparameter space."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': trial.suggest_categorical('eval_metric', ['error', 'auc', 'logloss']),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'eta': trial.suggest_float('eta', 0.005, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 10.0),
    }
    
    # Add parameters specific to gbtree and dart
    if params['booster'] == 'gbtree' or params['booster'] == 'dart':
        params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        params['tree_method'] = trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
        
        # Add parameters specific to lossguide grow policy
        if params['grow_policy'] == 'lossguide':
            params['max_leaves'] = trial.suggest_int('max_leaves', 0, 256)
            params['max_bin'] = trial.suggest_int('max_bin', 256, 512)
    
    # Add parameters specific to dart
    if params['booster'] == 'dart':
        params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        params['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.5)
        params['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.5)
    
    # Create data matrices
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Use cross-validation for more robust evaluation
    num_boost_round = trial.suggest_int('num_boost_round', 100, 2000)
    early_stopping = trial.suggest_int('early_stopping', 10, 100)
    
    # Train with early stopping
    model = xgb.train(params,
                      dtrain,
                      num_boost_round=num_boost_round,
                      evals=[(dtrain, 'train'), (dval, 'validation')],
                      early_stopping_rounds=early_stopping,
                      verbose_eval=False)
    
    # Predict and calculate accuracy and other metrics
    y_pred = model.predict(dval)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_val, y_pred_binary)
    
    # Calculate additional metrics for reporting
    try:
        auc = roc_auc_score(y_val, y_pred)
        trial.set_user_attr('auc', auc)
    except Exception:
        pass
    
    return accuracy

def catboost_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for CatBoost with expanded hyperparameter space."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
        'depth': trial.suggest_int('depth', 3, 8),  # Reduced maximum depth from 12 to 8
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 30.0, log=True),  # Increased minimum from 0.1 to 3.0
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
        'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'CrossEntropy']),
        'eval_metric': 'Accuracy',
        'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'max_leaves': trial.suggest_int('max_leaves', 10, 64),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'feature_border_type': trial.suggest_categorical('feature_border_type', 
                                                        ['Median', 'Uniform', 'UniformAndQuantiles', 'MaxLogSum']),
        'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['None', 'Balanced', 'SqrtBalanced']),
        'rsm': trial.suggest_float('rsm', 0.5, 1.0),  # Random subspace method (feature sampling)
    }
    
    # Add parameters specific to bootstrap types
    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.1, 10.0)
    elif params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
    
    # Create train and validation pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    
    # Initialize model
    model = CatBoostClassifier(**params)
    
    # Train model
    model.fit(train_pool,
              eval_set=val_pool,
              verbose=False,
              plot=False)
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    # Calculate additional metrics for reporting
    try:
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        trial.set_user_attr('auc', auc)
    except Exception:
        pass
    
    return accuracy

# --- Prediction Function ---

def predict(model, X_test, model_type='xgboost'):
    """Generate predictions using a trained model."""
    if model_type == 'xgboost':
        # Create DMatrix
        dtest = xgb.DMatrix(X_test)
        # Predict probabilities
        y_pred_proba = model.predict(dtest)
        # Convert to binary predictions
        y_pred = (y_pred_proba > 0.5).astype(int)
    elif model_type == 'catboost':
        # Ensure feature compatibility for CatBoost
        if hasattr(model, 'feature_names_'):
            # Get the feature names from the model
            model_features = model.feature_names_
            
            # Check for missing features
            missing_features = set(model_features) - set(X_test.columns)
            if missing_features:
                print(f"Warning: Missing features in test data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X_test[feature] = 0
                    
            # Check for extra features
            extra_features = set(X_test.columns) - set(model_features)
            if extra_features:
                print(f"Warning: Extra features in test data: {extra_features}")
                # Remove them
                X_test = X_test.drop(columns=list(extra_features))
                
            # Reorder columns to match model's expected order
            X_test = X_test[model_features]
            
        # Predict directly with CatBoost
        y_pred = model.predict(X_test)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return y_pred

# --- Main Execution Code for Notebooks ---

def run_training(params=None):
    """Run the training workflow with current parameters."""
    print("--- Configuration ---")
    print(f"Mode: {mode}")
    print(f"Model Type: {model_type}")
    print(f"Feature Engineering: {feature_engineering}")
    print(f"Use SMOTE: {use_smote}")
    print(f"Use Cross-Validation: {use_cv}")
    print(f"Hyperparameter Tuning: {hyperparameter_tuning}")
    if use_cv:
        print(f"CV Folds: {cv_folds}")
    print(f"Classification Threshold: {threshold}")
    if mode == 'tune' or hyperparameter_tuning:
        print(f"Optuna Trials: {n_trials}")
        print(f"Optimization Timeout: {timeout} seconds")
    elif mode == 'predict':
        print(f"Test Data Path: {test_path}")
        print(f"Submission Path: {submission_path}")
        print(f"Model Path for Prediction: {model_path}")
    print(f"Validation Split: {val_split}")
    print(f"Train Data Path: {data_path}")
    print(f"Random Seed: {seed}")
    print("---------------------")

    # --- Load & Preprocess Data ---
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: Training file not found at {data_path}")
        return None
        
    try:
        train_df_full = pd.read_csv(data_path)
        print(f"Loaded training data: {train_df_full.shape}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

    # 2. Preprocess Data
    print("Preprocessing data...")
    try:
        processed_df, _ = preprocess_data(train_df_full, add_features=feature_engineering)
        print(f"Preprocessing complete. Processed data shape: {processed_df.shape}")
        
        # Print summary of features
        print(f"\nFeature set summary:")
        print(f"  Total features: {processed_df.shape[1] - (1 if TARGET_COL in processed_df.columns else 0)}")
        
        # Analyze class distribution in target
        if TARGET_COL in processed_df.columns:
            target_counts = processed_df[TARGET_COL].value_counts()
            print(f"\nTarget distribution:")
            for label, count in target_counts.items():
                percentage = count / len(processed_df) * 100
                print(f"  Class {label}: {count} samples ({percentage:.2f}%)")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Training Mode ---
    if mode == 'train':
        print("\n--- Starting Training Mode ---")
        
        # 3. Prepare Data for ML Model
        X = processed_df.drop([TARGET_COL], axis=1)
        y = processed_df[TARGET_COL]

        # 4. Split Data for Training
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed, stratify=y
        )
        print(f"Data split: X_train {X_train.shape}, X_val {X_val.shape}")
        
        # Perform hyperparameter tuning if requested
        if hyperparameter_tuning:
            print(f"\nPerforming hyperparameter tuning for {model_type}...")
            try:
                model, best_params = tune_model_hyperparameters(
                    X_train, y_train, X_val, y_val, 
                    model_type=model_type,
                    n_trials=n_trials,
                    timeout=timeout
                )
                if model is not None:
                    print("Hyperparameter tuning completed successfully.")
                else:
                    print("Hyperparameter tuning failed, falling back to default parameters.")
            except Exception as e:
                print(f"Error during hyperparameter tuning: {e}")
                print("Falling back to default parameters.")
                model = None
                
            # If tuning failed, train with default parameters
            if model is None:
                if model_type == 'catboost':
                    print("Training CatBoost model with default parameters...")
                    model, val_accuracy = train_catboost_model(
                        X_train, y_train, X_val, y_val, 
                        params=params,
                        use_cv=use_cv, 
                        cv_folds=cv_folds
                    )
        else:
            # Train with default parameters
            if model_type == 'catboost':
                print("Training CatBoost model with default parameters...")
                model, val_accuracy = train_catboost_model(
                    X_train, y_train, X_val, y_val, 
                    params=params,
                    use_cv=use_cv, 
                    cv_folds=cv_folds
                )
        
        # Save the trained model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f'enhanced_{model_type}_model.pkl')
        save_model(model, model_path)
        print(f"\nTraining finished. Enhanced model saved to: {model_path}")
        
        # Store the training columns for later use in prediction
        train_columns = X_train.columns.tolist()
        train_columns_path = os.path.join(MODEL_DIR, f'train_columns_{model_type}.pkl')
        with open(train_columns_path, 'wb') as f:
            pickle.dump(train_columns, f)
        print(f"Training columns saved to: {train_columns_path}")
        
        # Generate predictions on test data if available
        if os.path.exists(test_path):
            print("\nGenerating predictions with the enhanced model...")
            try:
                test_df = pd.read_csv(test_path)
                # Use the training columns as reference for consistent preprocessing
                processed_test_df, test_ids = preprocess_data(test_df, add_features=feature_engineering, 
                                                            reference_columns=train_columns)
                
                # Remove target column if it exists in test data
                X_test = processed_test_df.drop([TARGET_COL], errors='ignore')
                
                # Make predictions
                predictions = predict(model, X_test, model_type=model_type)
                
                # Create submission file
                enhanced_submission_path = submission_path.replace('.csv', f'_enhanced_{model_type}.csv')
                submission_df = pd.DataFrame({
                    ID_COL: test_ids,
                    TARGET_COL: predictions
                })
                submission_df.to_csv(enhanced_submission_path, index=False)
                print(f"Enhanced model submission saved to {enhanced_submission_path}")
            except Exception as e:
                print(f"Error creating submission with enhanced model: {e}")
                import traceback
                traceback.print_exc()
                
        return model, X_train, y_train, X_val, y_val
    
    # --- Prediction Mode ---
    elif mode == 'predict':
        print("\n--- Starting Prediction Mode ---")
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        if not os.path.exists(test_path):
            print(f"Error: Test file not found at {test_path}")
            return None
            
        # Load the model
        model = load_model(model_path)
        if model is None:
            print("Failed to load model. Exiting.")
            return None
            
        # Try to load the training columns for feature compatibility
        train_columns_path = os.path.join(MODEL_DIR, f'train_columns_{model_type}.pkl')
        train_columns = None
        if os.path.exists(train_columns_path):
            try:
                with open(train_columns_path, 'rb') as f:
                    train_columns = pickle.load(f)
                print(f"Loaded training columns from {train_columns_path}")
            except Exception as e:
                print(f"Error loading training columns: {e}")
                
        # Load and preprocess test data
        test_df = pd.read_csv(test_path)
        processed_test_df, test_ids = preprocess_data(test_df, add_features=feature_engineering,
                                                    reference_columns=train_columns)
        
        # Remove target column if it exists in test data
        X_test = processed_test_df.drop([TARGET_COL], errors='ignore')
        
        # Generate predictions
        print("Generating predictions...")
        predictions = predict(model, X_test, model_type=model_type)
        
        # Create submission file
        submission_df = pd.DataFrame({
            ID_COL: test_ids,
            TARGET_COL: predictions
        })
        submission_df.to_csv(submission_path, index=False)
        print(f"Predictions saved to {submission_path}")
        
        return predictions, test_ids, model
    
    print("\n--- Script Finished ---")
    
# This function can be run directly in a Jupyter cell
# This function can be run directly in a Jupyter cell
def train_custom_model(custom_params=None):
    """Train a model with custom parameters in a notebook."""
    return run_training(params=custom_params)

if __name__ == "__main__":
    train_custom_model()