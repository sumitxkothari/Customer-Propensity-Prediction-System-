import pandas as pd
import numpy as np
import pickle
import os
import argparse
import datetime
import warnings
from functools import partial

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_curve
from sklearn.feature_selection import SelectFromModel, RFE, RFECV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Sampling techniques
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

# ML Models
import xgboost as xgb
from catboost import CatBoost, Pool, CatBoostClassifier
import lightgbm as lgb

# Import from train_ml.py (assuming it's in the same directory)
from train_ml import preprocess_data, NUMERICAL_COLS, EDU_COL, MARITAL_COL, TARGET_COL, ID_COL, DEFAULT_VAL_SPLIT

# Constants
MODEL_DIR = 'models'
TRAIN_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/train.csv'
TEST_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/test.csv'
SUBMISSION_CSV_PATH = r'submission_advanced_ml.csv'

# Suppress warnings
warnings.filterwarnings('ignore')

def save_model(model, model_path):
    """Save the trained model to a file."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def advanced_feature_engineering(df, cluster_count=2, polynomial_degree=2, add_pca=True, n_components=3):
    """Apply advanced feature engineering techniques with reduced complexity."""
    print("\nApplying advanced feature engineering...")
    
    # Store original shape for reporting
    original_shape = df.shape
    
    # Extract numerical features for transformation
    if TARGET_COL in df.columns:
        numerical_features = df.drop(columns=[TARGET_COL]).select_dtypes(include=['float64', 'int64']).columns
    else:
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Create copy to avoid modifying original
    df_engineered = df.copy()
    
    # 1. Polynomial features for key numerical features - REDUCED SCOPE
    print("  Creating polynomial features...")
    # Take only top 5 most likely important numerical features (instead of 10)
    likely_important = ['Recency', 'Income', 'MntWines', 'NumWebPurchases', 'NumCatalogPurchases']
    top_features = [col for col in likely_important if col in numerical_features][:5]  
    
    if len(top_features) > 0:
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df_engineered[top_features])
        
        # Create DataFrame with polynomial feature names
        feature_names = [f"Poly_{i}" for i in range(poly_features.shape[1])]
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df_engineered.index)
        
        # Drop original polynomial feature columns to avoid duplication
        poly_df = poly_df.iloc[:, len(top_features):]  # Keep only interaction terms
        
        # Add to engineered dataframe
        df_engineered = pd.concat([df_engineered, poly_df], axis=1)
    
    # 2. K-Means clustering features - REDUCED SCOPE
    print("  Adding clustering features...")
    if len(numerical_features) > 0:
        # Use only a subset of numerical features for clustering to reduce dimensionality
        cluster_features = numerical_features[:min(10, len(numerical_features))]
        
        # Scale features for clustering
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_engineered[cluster_features])
        
        # Create only 2 clusters for simplicity
        for n_clusters in range(2, cluster_count + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            df_engineered[f'Cluster_{n_clusters}'] = cluster_labels
            
            # Add only the distance to each cluster center (not all distance features)
            distances = kmeans.transform(scaled_features)
            for i in range(n_clusters):
                df_engineered[f'Cluster_{n_clusters}_Dist_{i}'] = distances[:, i]
    
    # 3. PCA components as features - REDUCED NUMBER
    if add_pca and len(numerical_features) > n_components:
        print("  Adding PCA components...")
        # Use only main numerical features for PCA
        pca_features = numerical_features[:min(15, len(numerical_features))]
        
        pca = PCA(n_components=n_components)
        try:
            pca_result = pca.fit_transform(scaler.transform(df_engineered[pca_features]))
            for i in range(n_components):
                df_engineered[f'PCA_{i+1}'] = pca_result[:, i]
            
            # Add explained variance as a feature 
            df_engineered['PCA_Variance_Ratio'] = np.sum(pca.explained_variance_ratio_)
        except Exception as e:
            print(f"  PCA calculation failed: {e}. Skipping PCA features.")
    
    # 4. Feature ratios and advanced interactions for key features - REDUCED SCOPE
    print("  Creating feature ratios and interactions...")
    # Only use the most important features
    likely_important = ['Recency', 'MntWines', 'NumWebPurchases', 'Income', 'NumCatalogPurchases']
    important_in_df = [col for col in likely_important if col in df_engineered.columns][:4] # Limit to top 4
    
    if len(important_in_df) >= 2:
        for i in range(len(important_in_df)):
            for j in range(i+1, len(important_in_df)):
                # Ratio features (only if both exist)
                col1, col2 = important_in_df[i], important_in_df[j]
                ratio_name = f"{col1}_div_{col2}"
                df_engineered[ratio_name] = df_engineered[col1] / (df_engineered[col2] + 0.001)
                
                # Difference features
                diff_name = f"{col1}_minus_{col2}"
                df_engineered[diff_name] = df_engineered[col1] - df_engineered[col2]
                
                # Skip product and sum features to reduce dimensionality
    
    # 5. Statistical aggregations for related features (keep these as they're valuable)
    print("  Creating statistical aggregations...")
    # Group related features
    purchase_cols = [col for col in df_engineered.columns if 'Purchase' in col and df_engineered[col].dtype in ['int64', 'float64']]
    spending_cols = [col for col in df_engineered.columns if 'Mnt' in col and df_engineered[col].dtype in ['int64', 'float64']]
    
    if purchase_cols:
        df_engineered['Purchase_Mean'] = df_engineered[purchase_cols].mean(axis=1)
        df_engineered['Purchase_Std'] = df_engineered[purchase_cols].std(axis=1)
        # Skip median, min, max features
    
    if spending_cols:
        df_engineered['Spending_Mean'] = df_engineered[spending_cols].mean(axis=1)
        df_engineered['Spending_Std'] = df_engineered[spending_cols].std(axis=1)
        # Skip median, min, max features
    
    # 6. Date-based features - KEEP THESE (they're valuable and low-dimensional)
    if all(col in df_engineered.columns for col in ['Dt_Year', 'Dt_Month']):
        df_engineered['Customer_Since_Years'] = datetime.datetime.now().year - df_engineered['Dt_Year']
        df_engineered['Is_Recent_Customer'] = (df_engineered['Customer_Since_Years'] <= 1).astype(int)
        df_engineered['Is_Season_Winter'] = ((df_engineered['Dt_Month'] == 12) | 
                                            (df_engineered['Dt_Month'] == 1) | 
                                            (df_engineered['Dt_Month'] == 2)).astype(int)
        df_engineered['Is_Season_Summer'] = ((df_engineered['Dt_Month'] >= 6) & 
                                            (df_engineered['Dt_Month'] <= 8)).astype(int)
    
    # 7. More non-linear transformations - REDUCED SCOPE
    print("  Adding non-linear transformations...")
    # Only transform most important features
    key_features = ['Recency', 'Income', 'NumWebPurchases', 'MntWines']
    transform_features = [col for col in key_features if col in numerical_features]
    
    for col in transform_features:
        # Log transformations (add small constant to avoid log(0))
        if (df_engineered[col] >= 0).all():
            df_engineered[f'{col}_Log'] = np.log1p(df_engineered[col])
        
        # Skip square root transformations to reduce dimensionality
        
        # Exponential decay (only for recency)
        if col == 'Recency' and col in df_engineered.columns:
            df_engineered[f'{col}_Exp_Decay'] = np.exp(-df_engineered[col]/10)  # Decay factor
    
    print(f"  Feature engineering complete: {original_shape} → {df_engineered.shape}")
    return df_engineered

def perform_feature_selection(X, y, model_type='catboost', feature_fraction=0.5, 
                             method='model', cv=3, step=10):
    """Select the most important features using various techniques with reduced complexity."""
    print("\nPerforming feature selection...")
    print(f"  Original feature count: {X.shape[1]}")
    
    # Ensure we don't select too few features
    min_features = min(30, X.shape[1])
    n_features_to_keep = max(int(X.shape[1] * feature_fraction), min_features)
    
    # Choose base model for feature selection based on model_type - SIMPLER MODELS
    if model_type == 'catboost':
        base_model = CatBoostClassifier(
            iterations=200,     # Reduced from 500
            learning_rate=0.1,  # Increased for faster convergence
            depth=4,            # Reduced from 6
            loss_function='Logloss',
            verbose=0
        )
    elif model_type == 'xgboost':
        base_model = xgb.XGBClassifier(
            n_estimators=200,   # Reduced from 500
            learning_rate=0.1,  # Increased for faster convergence
            max_depth=4,        # Reduced from 6
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=0
        )
    else:
        base_model = RandomForestClassifier(
            n_estimators=100,   # Reduced from 500
            max_depth=4,        # Reduced from 6
            random_state=42
        )
    
    if method == 'model':
        # 1. Use model's feature importances
        print("  Using model-based feature selection...")
        
        # Fit model to get feature importances
        try:
            base_model.fit(X, y)
            
            # Extract feature importances
            if hasattr(base_model, 'feature_importances_'):
                importances = base_model.feature_importances_
            elif hasattr(base_model, 'get_feature_importance'):
                importances = base_model.get_feature_importance()
            else:
                print("  Model doesn't provide feature importances, skipping selection")
                return X
            
            # Create selector with a threshold to keep top features
            selector = SelectFromModel(base_model, threshold=-np.inf, 
                                     max_features=n_features_to_keep, prefit=True)
            
        except Exception as e:
            print(f"  Error during model-based feature selection: {e}")
            print("  Returning original features")
            return X
            
    elif method == 'rfe':
        # 2. Recursive Feature Elimination - SIMPLIFIED
        print("  Using Recursive Feature Elimination (RFE)...")
        try:
            # Use a larger step size and fewer features to reduce computation
            selector = RFE(estimator=base_model, 
                          n_features_to_select=n_features_to_keep, 
                          step=step)
            selector.fit(X, y)
        except Exception as e:
            print(f"  Error during RFE feature selection: {e}")
            print("  Returning original features")
            return X
        
    else:
        print(f"  Unknown method '{method}', skipping selection")
        return X
    
    # Apply feature selection
    try:
        X_selected = selector.transform(X)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        print(f"  Selected {len(selected_features)}/{X.shape[1]} features")
        print(f"  Top 10 selected features: {selected_features[:min(10, len(selected_features))]}")
        
        # Return selected features dataframe with column names preserved
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        return X_selected_df
        
    except Exception as e:
        print(f"  Error applying feature selection: {e}")
        print("  Returning original features")
        return X

def train_stacked_ensemble(X_train, y_train, X_val, y_val, cv=3):
    """Train a stacked ensemble of models for higher accuracy with reduced complexity."""
    print("\nTraining stacked ensemble model...")
    
    # Base learners for the ensemble - REDUCED COMPLEXITY
    base_learners = [
        ('catboost', CatBoostClassifier(
            iterations=300,          # Reduced from 1000
            learning_rate=0.05,
            depth=6,                 # Reduced from 8
            l2_leaf_reg=5,
            bootstrap_type='Bayesian',
            verbose=0,
            random_state=42,
            thread_count=2          # Limit threads
        )),
        ('xgboost', xgb.XGBClassifier(
            n_estimators=300,        # Reduced from 1000
            learning_rate=0.05,
            max_depth=6,             # Reduced from 7
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2                 # Limit threads
        )),
        ('lightgbm', lgb.LGBMClassifier(
            n_estimators=300,        # Reduced from 1000
            learning_rate=0.05,
            max_depth=6,             # Reduced from 8
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=42,
            n_jobs=2                 # Limit threads
        ))
        # Removed RandomForest to reduce memory usage
    ]
    
    # Meta-learner (final estimator)
    meta_learner = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    
    # Create and train the stacking ensemble with reduced parallelism
    stacked_model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=cv,                     # Reduced from 5 to 3
        stack_method='predict_proba',
        n_jobs=1                  # Single job to prevent memory issues
    )
    
    # Train stacked model with error handling
    print("  Training stacked model (this may take a while)...")
    try:
        # Train models one by one first to reduce memory pressure
        print("  Pre-training individual models...")
        for name, model in base_learners:
            print(f"    Training {name}...")
            model.fit(X_train, y_train)
            # Validate each model
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            print(f"    {name} validation accuracy: {acc:.4f}")
        
        # Now train the stacked model
        print("  Training ensemble...")
        stacked_model.fit(X_train, y_train)
    except MemoryError:
        print("  MemoryError: Not enough memory to train the full stacked model.")
        print("  Falling back to the best individual model.")
        
        # Use the best individual model instead
        best_acc = 0
        best_model = None
        best_name = None
        
        for name, model in base_learners:
            # Each model was already fit above
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_model = model
                best_name = name
        
        print(f"  Using {best_name} as the final model with accuracy: {best_acc:.4f}")
        return best_model
    except Exception as e:
        print(f"  Error training stacked model: {e}")
        print("  Falling back to CatBoost model.")
        # Fall back to CatBoost as a safe option
        catboost_model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=42
        )
        catboost_model.fit(X_train, y_train)
        return catboost_model
    
    # Evaluate on validation set
    val_pred = stacked_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    try:
        val_proba = stacked_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        print(f"  Validation Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    except:
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
    
    print("  Classification Report:")
    print(classification_report(y_val, val_pred))
    
    return stacked_model

def find_optimal_threshold(y_val, y_pred_proba):
    """Find the optimal classification threshold for best accuracy."""
    print("\nFinding optimal classification threshold...")
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)
    
    # Find threshold with highest accuracy
    best_threshold = thresholds[np.argmax(accuracies)]
    best_accuracy = max(accuracies)
    
    print(f"  Optimal threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")
    print(f"  Improvement over 0.5 threshold: {best_accuracy - accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int)):.4f}")
    
    return best_threshold

def main():
    parser = argparse.ArgumentParser(description='Advanced ML training with stacking and feature engineering')
    parser.add_argument('--data_path', type=str, default=TRAIN_CSV_PATH,
                       help='Path to the training CSV file')
    parser.add_argument('--test_path', type=str, default=TEST_CSV_PATH,
                       help='Path to the test CSV file')
    parser.add_argument('--submission_path', type=str, default=SUBMISSION_CSV_PATH,
                       help='Path to save the submission CSV file')
    parser.add_argument('--val_split', type=float, default=DEFAULT_VAL_SPLIT,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--feature_engineering', action='store_true', default=True,
                       help='Apply feature engineering')
    parser.add_argument('--feature_selection', action='store_true', default=True,
                       help='Perform feature selection')
    parser.add_argument('--selection_method', type=str, default='model', 
                       choices=['model', 'rfe', 'rfecv'],
                       help='Feature selection method')
    parser.add_argument('--create_submission', action='store_true', default=True,
                       help='Create submission file after training')
    parser.add_argument('--simple_mode', action='store_true', default=False,
                       help='Run in simple mode for limited resources')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("=== Advanced ML Training with Stacking and Feature Engineering ===")
    print(f"Data Path: {args.data_path}")
    print(f"Test Path: {args.test_path}")
    print(f"Submission Path: {args.submission_path}")
    print(f"Validation Split: {args.val_split}")
    print(f"Feature Engineering: {args.feature_engineering}")
    print(f"Feature Selection: {args.feature_selection}")
    print(f"Selection Method: {args.selection_method}")
    print(f"Create Submission: {args.create_submission}")
    print(f"Simple Mode: {args.simple_mode}")
    print("===================================================================")
    
    # If running in simple mode, use a single CatBoost model
    if args.simple_mode:
        print("\nRunning in simple mode for limited resources.")
        args.feature_engineering = True
        args.feature_selection = True
    
    # 1. Load and preprocess data
    print("\nLoading and preprocessing data...")
    try:
        train_df = pd.read_csv(args.data_path)
        print(f"Loaded training data: {train_df.shape}")
        
        # Apply initial preprocessing
        processed_df, _ = preprocess_data(train_df, add_features=True)
        print(f"Initial preprocessing complete: {processed_df.shape}")
        
    except Exception as e:
        print(f"Error loading/preprocessing data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. Apply advanced feature engineering if enabled
    if args.feature_engineering:
        try:
            processed_df = advanced_feature_engineering(processed_df)
        except Exception as e:
            print(f"Error during advanced feature engineering: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Split data into train and validation sets
    print("\nSplitting data into train and validation sets...")
    if TARGET_COL not in processed_df.columns:
        print(f"Error: Target column '{TARGET_COL}' not found in data")
        return
    
    X = processed_df.drop(columns=[TARGET_COL])
    y = processed_df[TARGET_COL]
    
    # Show class distribution
    class_counts = y.value_counts()
    print(f"Class distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} ({count/len(y)*100:.2f}%)")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}")
    
    # 4. Perform feature selection if enabled
    if args.feature_selection:
        try:
            X_train = perform_feature_selection(
                X_train, y_train, 
                model_type='catboost',
                method=args.selection_method,
                feature_fraction=0.5  # More aggressive feature selection
            )
            
            if X_train.shape[1] > 50:
                print(f"Selected features still exceed 50, taking top 50 only for efficiency")
                # Get feature importances from a simple model
                selector_model = CatBoostClassifier(iterations=100, verbose=0)
                selector_model.fit(X_train, y_train)
                
                # Find top 50 features
                importances = selector_model.get_feature_importance()
                top_indices = np.argsort(importances)[-50:]
                top_features = X_train.columns[top_indices]
                
                # Keep only top 50 features
                X_train = X_train[top_features]
            
            # Ensure validation data has the same features
            X_val = X_val[X_train.columns]
            print(f"Feature selection complete: {X_train.shape}")
        except Exception as e:
            print(f"Error during feature selection: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. Apply SMOTE for class balancing
    print("\nApplying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=args.seed)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - X shape: {X_train_resampled.shape}")
        # Show resampled class distribution
        resampled_counts = pd.Series(y_train_resampled).value_counts()
        for cls, count in resampled_counts.items():
            print(f"  Class {cls}: {count} ({count/len(y_train_resampled)*100:.2f}%)")
    except Exception as e:
        print(f"SMOTE failed, using original data: {e}")
        X_train_resampled, y_train_resampled = X_train, y_train
    
    # 6. Train model
    model = None
    try:
        if args.simple_mode:
            # Skip ensemble, use single CatBoost model for limited resources
            print("\nTraining single CatBoost model (simple mode)...")
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=5,
                verbose=100,
                random_state=42
            )
            model.fit(X_train_resampled, y_train_resampled, 
                     eval_set=(X_val, y_val), 
                     early_stopping_rounds=50,
                     verbose=50)
            
            # Evaluate
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(classification_report(y_val, y_val_pred))
        else:
            # Train stacked ensemble model
            model = train_stacked_ensemble(X_train_resampled, y_train_resampled, X_val, y_val, cv=3)
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simple CatBoost model
        print("\nFalling back to simple CatBoost model...")
        model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=100,
            random_state=42
        )
        model.fit(X_train_resampled, y_train_resampled)
    
    if model is None:
        print("Error: Failed to train any model. Exiting.")
        return
    
    # 7. Find optimal threshold
    optimal_threshold = 0.5  # Default
    try:
        if hasattr(model, 'predict_proba'):
            y_val_proba = model.predict_proba(X_val)[:, 1]
            optimal_threshold = find_optimal_threshold(y_val, y_val_proba)
        else:
            print("Model doesn't support predict_proba, using default threshold of 0.5")
    except Exception as e:
        print(f"Error finding optimal threshold: {e}")
        print("Using default threshold of 0.5")
    
    # 8. Save model
    model_path = os.path.join(MODEL_DIR, f'advanced_ensemble_model.pkl')
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_info = {
            'model': model,
            'threshold': optimal_threshold,
            'features': X_train.columns.tolist(),
            'feature_engineering': args.feature_engineering,
            'feature_selection': args.feature_selection
        }
        save_model(model_info, model_path)
        print(f"\nSaved model and settings to: {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # 9. Create submission if requested
    if args.create_submission and os.path.exists(args.test_path):
        print("\nCreating submission file...")
        try:
            # Load and preprocess test data
            test_df = pd.read_csv(args.test_path)
            print(f"Loaded test data: {test_df.shape}")
            
            # Apply initial preprocessing
            processed_test_df, test_ids = preprocess_data(test_df, add_features=True)
            print(f"Initial test preprocessing complete: {processed_test_df.shape}")
            
            # Apply advanced feature engineering if enabled
            if args.feature_engineering:
                processed_test_df = advanced_feature_engineering(processed_test_df)
            
            # Ensure correct features for prediction (align with training features)
            X_test = processed_test_df.drop([TARGET_COL], errors='ignore')
            
            # Select only the features used during training
            X_test_final = pd.DataFrame(index=X_test.index)
            for col in X_train.columns:
                if col in X_test.columns:
                    X_test_final[col] = X_test[col]
                else:
                    X_test_final[col] = 0  # Add missing columns with default values
            
            print(f"Final test features: {X_test_final.shape}")
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                test_proba = model.predict_proba(X_test_final)[:, 1]
                test_preds = (test_proba >= optimal_threshold).astype(int)
            else:
                test_preds = model.predict(X_test_final)
            
            # Create submission file
            submission_df = pd.DataFrame({
                ID_COL: test_ids,
                TARGET_COL: test_preds
            })
            
            submission_df.to_csv(args.submission_path, index=False)
            print(f"Submission file saved to: {args.submission_path}")
            
            # Print prediction distribution
            n_positive = sum(test_preds)
            print(f"Prediction distribution:")
            print(f"  Class 1: {n_positive} ({n_positive/len(test_preds)*100:.2f}%)")
            print(f"  Class 0: {len(test_preds)-n_positive} ({(len(test_preds)-n_positive)/len(test_preds)*100:.2f}%)")
            
        except Exception as e:
            print(f"Error creating submission: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== Advanced ML Training Complete ===")

if __name__ == "__main__":
    main() 