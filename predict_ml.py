import pandas as pd
import numpy as np
import pickle
import os
import argparse
from sklearn.preprocessing import OneHotEncoder
from train_ml import preprocess_data, NUMERICAL_COLS, EDU_COL, MARITAL_COL, TARGET_COL, ID_COL

# Path constants
MODEL_DIR = 'models'
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'enhanced_catboost_model.pkl')
TRAIN_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/train.csv'
TEST_CSV_PATH = r'appian-x-iit-madras-hackathon-april-2025/test.csv'
SUBMISSION_CSV_PATH = r'submission_ml_enhanced.csv'

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

def align_features(X_train_cols, X_test):
    """Align features between training and test data"""
    # Features in test data but not in training data - drop them
    cols_to_drop = [col for col in X_test.columns if col not in X_train_cols]
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns from test data not in training data: {cols_to_drop[:5]}...")
        X_test = X_test.drop(columns=cols_to_drop)
    
    # Features in training data but not in test data - add them with 0s
    cols_to_add = [col for col in X_train_cols if col not in X_test.columns]
    if cols_to_add:
        print(f"Adding {len(cols_to_add)} missing columns to test data: {cols_to_add[:5]}...")
        for col in cols_to_add:
            X_test[col] = 0
    
    # Reorder columns to match training data
    X_test = X_test[X_train_cols]
    
    return X_test

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained ML model')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained model file')
    parser.add_argument('--train_path', type=str, default=TRAIN_CSV_PATH,
                        help='Path to the training CSV file (for feature alignment)')
    parser.add_argument('--test_path', type=str, default=TEST_CSV_PATH,
                        help='Path to the test CSV file')
    parser.add_argument('--submission_path', type=str, default=SUBMISSION_CSV_PATH,
                        help='Path to save the submission CSV file')
    parser.add_argument('--feature_engineering', action='store_true', default=True,
                        help='Apply feature engineering')
    
    args = parser.parse_args()
    
    print("--- Making Predictions with Trained Model ---")
    print(f"Model Path: {args.model_path}")
    print(f"Train Data Path: {args.train_path}")
    print(f"Test Data Path: {args.test_path}")
    print(f"Submission Path: {args.submission_path}")
    print(f"Feature Engineering: {args.feature_engineering}")
    print("-------------------------------------------")
    
    # 1. Load model
    model = load_model(args.model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # 2. Load training data to get consistent feature set
    print("Loading training data for feature alignment...")
    try:
        train_df = pd.read_csv(args.train_path)
        processed_train_df, _ = preprocess_data(train_df, add_features=args.feature_engineering)
        X_train_cols = processed_train_df.drop(columns=[TARGET_COL]).columns.tolist()
        print(f"Training data loaded and processed: {processed_train_df.shape}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return
    
    # 3. Load and preprocess test data
    print("Loading test data...")
    try:
        test_df = pd.read_csv(args.test_path)
        processed_test_df, test_ids = preprocess_data(test_df, add_features=args.feature_engineering)
        print(f"Test data loaded and processed: {processed_test_df.shape}")
        
        if test_ids is None:
            print(f"Error: ID column ('{ID_COL}') not found or empty in test data.")
            return
    except Exception as e:
        print(f"Error loading/preprocessing test data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Ensure feature compatibility
    X_test = processed_test_df.drop([TARGET_COL], errors='ignore')
    print(f"Before alignment: Test data has {X_test.shape[1]} features")
    X_test_aligned = align_features(X_train_cols, X_test)
    print(f"After alignment: Test data has {X_test_aligned.shape[1]} features")
    
    # 5. Make predictions
    print("Making predictions...")
    try:
        if hasattr(model, 'predict_proba'):
            # For ensemble models or models with predict_proba
            y_pred_proba = model.predict_proba(X_test_aligned)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            # For models with only predict
            y_pred = model.predict(X_test_aligned)
            
        # Ensure predictions are integers (0 or 1)
        y_pred = y_pred.astype(int)
        
        # Print prediction statistics
        n_positive = np.sum(y_pred)
        n_total = len(y_pred)
        print(f"Prediction Statistics:")
        print(f"  Total Predictions: {n_total}")
        print(f"  Positive Predictions (1): {n_positive} ({n_positive/n_total*100:.2f}%)")
        print(f"  Negative Predictions (0): {n_total-n_positive} ({(n_total-n_positive)/n_total*100:.2f}%)")
        
    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Create submission file
    print("Creating submission file...")
    try:
        submission_df = pd.DataFrame({
            ID_COL: test_ids,
            TARGET_COL: y_pred
        })
        
        submission_df.to_csv(args.submission_path, index=False)
        print(f"Submission file saved to {args.submission_path}")
    except Exception as e:
        print(f"Error creating submission file: {e}")
        return
    
    print("--- Prediction Complete ---")

if __name__ == "__main__":
    main() 