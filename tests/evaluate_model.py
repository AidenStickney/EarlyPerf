import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, LeaveOneOut

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_utils import generator

def load_mappers(mappers_path):
    with open(mappers_path, 'r') as f:
        return json.load(f)

def get_mapped_value(mappers, feature_name, string_value):
    if "mappers" not in mappers:
        return None
    
    if feature_name in mappers["mappers"]:
        mapping = mappers["mappers"][feature_name]
        for k, v in mapping.items():
            if v == string_value:
                return int(k)
    
    for k_map in mappers["mappers"]:
        if k_map.lower() == feature_name.lower():
             mapping = mappers["mappers"][k_map]
             for k, v in mapping.items():
                if v == string_value:
                    return int(k)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate model by holding out specific configurations.")
    parser.add_argument("--pickle", type=str, required=True, help="Path to the pickle file containing parsed data.")
    parser.add_argument("--mode", type=str, choices=["random", "index", "group", "kfold", "loo"], default="random", help="Holdout mode: 'random' (single random), 'index' (specific index), 'group' (all with specific feature value), 'kfold' (k-fold cross val), or 'loo' (leave-one-out).")
    parser.add_argument("--index", type=int, default=0, help="Index of configuration to hold out (for mode='index').")
    parser.add_argument("--feature", type=str, help="Feature name to hold out (for mode='group'), e.g., 'BranchPredictor'.")
    parser.add_argument("--value", type=str, help="Feature value to hold out (for mode='group'), e.g., 'perceptron'.")
    parser.add_argument("--k", type=int, default=3, help="Number of folds for k-fold cross validation.")
    parser.add_argument("--mappers", type=str, default="mappers.json", help="Path to mappers.json (required for mode='group').")
    parser.add_argument("--duration", type=int, default=10, help="Duration of simulation steps to use (preview).")

    args = parser.parse_args()

    if not os.path.exists(args.pickle):
        print(f"Error: {args.pickle} not found.")
        return

    print(f"Loading data from {args.pickle}...")
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    keys, y_names_dict = generator(data, n=args.duration, train_after_warmup=False, warmup_period=80)
    
    target_key = "cumulative_ipc"
    if target_key not in keys:
        print("Error: Target 'cumulative_ipc' not found.")
        return

    X_list, y_list_series = keys[target_key]
    feature_names = y_names_dict[target_key]
    
    y_final = np.array([series[-1] for series in y_list_series]) # Final IPC
    X = pd.DataFrame(X_list, columns=feature_names)
    
    print(f"Total samples: {len(X)}")

    # Determine Train/Test Split
    splits = []
    
    if args.mode == "random":
        test_idx = np.random.randint(0, len(X))
        print(f"Holding out random index: {test_idx}")
        train_indices = np.delete(np.arange(len(X)), test_idx)
        test_indices = np.array([test_idx])
        splits.append((train_indices, test_indices))
        
    elif args.mode == "index":
        if args.index >= len(X):
             print(f"Error: Index {args.index} out of bounds.")
             return
        print(f"Holding out specific index: {args.index}")
        train_indices = np.delete(np.arange(len(X)), args.index)
        test_indices = np.array([args.index])
        splits.append((train_indices, test_indices))
        
    elif args.mode == "group":
        if not args.feature or not args.value:
            print("Error: --feature and --value required for group mode.")
            return
            
        col_match = None
        for col in X.columns:
            if col.lower() == args.feature.lower():
                col_match = col
                break
        
        if not col_match:
            print(f"Error: Feature '{args.feature}' not found in dataset columns.")
            print("Available columns:", list(X.columns))
            return

        target_val = args.value
        try:
            target_val = int(args.value)
            print(f"Using integer value: {target_val}")
        except ValueError:
            mappers = load_mappers(args.mappers)
            mapped_val = get_mapped_value(mappers, args.feature, args.value)
            if mapped_val is None:
                print(f"Error: Could not map string value '{args.value}' for feature '{args.feature}'. Check mappers.json.")
                return
            target_val = mapped_val
            print(f"Mapped '{args.value}' to integer: {target_val}")

        is_test = (X[col_match] == target_val)
        print(f"Holding out group: {col_match} = {target_val}")
        print(f"Found {is_test.sum()} samples matching criteria.")
        
        if is_test.sum() == 0:
            print("Warning: No samples in test set.")
            return
        if is_test.sum() == len(X):
            print("Warning: All samples are in test set! Nothing to train on.")
            return
            
        train_indices = np.where(~is_test)[0]
        test_indices = np.where(is_test)[0]
        splits.append((train_indices, test_indices))

    elif args.mode == "kfold":
        print(f"Performing {args.k}-Fold Cross Validation...")
        kf = KFold(n_splits=args.k, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    elif args.mode == "loo":
        print("Performing Leave-One-Out Cross Validation...")
        loo = LeaveOneOut()
        splits = list(loo.split(X))

    # Evaluate
    maes = []
    mapes = []
    
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_final[train_idx], y_final[test_idx]
        
        # Train
        model = ExtraTreesRegressor(n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, preds)
        
        # Div by zero safe MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_vals = np.abs((y_test - preds) / y_test) * 100
            mape_vals[~np.isfinite(mape_vals)] = 0
            mape = np.mean(mape_vals)

        maes.append(mae)
        mapes.append(mape)
        
        if len(splits) == 1:
            print(f"Training Configs: {len(X_train)}")
            print(f"Testing Configs:  {len(X_test)}")
            print("Results:")
            print(f"MAE:  {mae:.6f}")
            print(f"MAPE: {mape:.2f}%")
            
            if len(y_test) < 10:
                print("\nIndividual Predictions:")
                for j in range(len(y_test)):
                    print(f"Actual: {y_test[j]:.4f}, Pred: {preds[j]:.4f}, Diff: {abs(y_test[j]-preds[j]):.4f}")
    
    if len(splits) > 1:
        print("\nCross Validation Results:")
        print(f"Average MAE:  {np.mean(maes):.6f} (+/- {np.std(maes):.6f})")
        print(f"Average MAPE: {np.mean(mapes):.2f}% (+/- {np.std(mapes):.2f}%)")

if __name__ == "__main__":
    main()
