import pandas as pd
import numpy as np
import pickle
import os
import argparse
import warnings
from train_ml import preprocess_data, NUMERICAL_COLS, EDU_COL, MARITAL_COL, TARGET_COL, ID_COL
from advanced_train_ml import advanced_feature_engineering

# Suppress warnings
warnings.filterwarnings('ignore')

# Path constants
MODEL_DIR = 'models'
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'advanced_stacked_ensemble.pkl')
TRAIN_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/train.csv'
TEST_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/test.csv'
SUBMISSION_CSV_PATH = r'submission_advanced_ml.csv'

def load_model(model_path):
    """Load a trained model from a file."""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def align_features(train_features, X_test):
    """Align features between training features and test features."""
    print(f"Aligning features to match training data:")
    print(f"  Training features: {len(train_features)}")
    print(f"  Test features before alignment: {X_test.shape[1]}")
    
    # Features in test data but not in training data - drop them
    cols_to_drop = [col for col in X_test.columns if col not in train_features]
    if cols_to_drop:
        print(f"  Dropping {len(cols_to_drop)} columns from test data not in training data.")
        if len(cols_to_drop) < 20:  # Only show if not too many
            print(f"  Dropped columns: {cols_to_drop}")
        X_test = X_test.drop(columns=cols_to_drop)
    
    # Features in training data but not in test data - add them with 0s
    cols_to_add = [col for col in train_features if col not in X_test.columns]
    if cols_to_add:
        print(f"  Adding {len(cols_to_add)} missing columns to test data.")
        if len(cols_to_add) < 20:  # Only show if not too many
            print(f"  Added columns: {cols_to_add}")
        for col in cols_to_add:
            X_test[col] = 0
    
    # Reorder columns to match training data
    X_test = X_test[train_features]
    print(f"  Test features after alignment: {X_test.shape[1]}")
    
    return X_test

def main():
    parser = argparse.ArgumentParser(description='Make predictions with advanced stacked ensemble model')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained model file')
    parser.add_argument('--test_path', type=str, default=TEST_CSV_PATH,
                        help='Path to the test CSV file')
    parser.add_argument('--submission_path', type=str, default=SUBMISSION_CSV_PATH,
                        help='Path to save the submission CSV file')
    
    args = parser.parse_args()
    
    print("=== Advanced Model Prediction ===")
    print(f"Model Path: {args.model_path}")
    print(f"Test Path: {args.test_path}")
    print(f"Submission Path: {args.submission_path}")
    print("================================")
    
    # 1. Load model
    model_info = load_model(args.model_path)
    if model_info is None:
        print("Failed to load model. Exiting.")
        return
    
    # Extract model components
    model = model_info['model']
    threshold = model_info.get('threshold', 0.5)
    train_features = model_info.get('features', [])
    use_feature_engineering = model_info.get('feature_engineering', True)
    
    print(f"Using classification threshold: {threshold}")
    print(f"Model trained with {len(train_features)} features")
    print(f"Advanced feature engineering: {use_feature_engineering}")
    
    # 2. Load and preprocess test data
    print("\nLoading and preprocessing test data...")
    try:
        test_df = pd.read_csv(args.test_path)
        print(f"Loaded test data: {test_df.shape}")
        
        # Apply initial preprocessing
        processed_test_df, test_ids = preprocess_data(test_df, add_features=True)
        print(f"Initial preprocessing complete: {processed_test_df.shape}")
        
        if test_ids is None:
            print(f"Error: ID column ('{ID_COL}') not found or empty in test data.")
            return
    except Exception as e:
        print(f"Error loading/preprocessing test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Apply advanced feature engineering if used in training
    if use_feature_engineering:
        try:
            processed_test_df = advanced_feature_engineering(processed_test_df)
        except Exception as e:
            print(f"Error during advanced feature engineering: {e}")
            print("Continuing with basic features only...")
    
    # 4. Ensure feature compatibility
    X_test = processed_test_df.drop([TARGET_COL], errors='ignore')
    
    # If we have training features list, align to match
    if train_features:
        X_test_aligned = align_features(train_features, X_test)
    else:
        print("Warning: No feature list provided with model. Using all available features.")
        X_test_aligned = X_test
    
    # 5. Make predictions
    print("\nMaking predictions...")
    try:
        test_proba = model.predict_proba(X_test_aligned)[:, 1]
        test_preds = (test_proba >= threshold).astype(int)
        
        # Print prediction distribution
        n_positive = test_preds.sum()
        total = len(test_preds)
        print(f"Prediction distribution:")
        print(f"  Class 1: {n_positive} ({n_positive/total*100:.2f}%)")
        print(f"  Class 0: {total-n_positive} ({(total-n_positive)/total*100:.2f}%)")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Create submission file
    print("\nCreating submission file...")
    try:
        submission_df = pd.DataFrame({
            ID_COL: test_ids,
            TARGET_COL: test_preds
        })
        
        submission_df.to_csv(args.submission_path, index=False)
        print(f"Submission file saved to: {args.submission_path}")
    except Exception as e:
        print(f"Error creating submission file: {e}")
        return
    
    print("\n=== Prediction Complete ===")

if __name__ == "__main__":
    main() 