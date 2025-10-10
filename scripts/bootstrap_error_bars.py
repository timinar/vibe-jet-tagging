"""
Bootstrap error bars for baseline classifiers.

This script computes bootstrapped error bars for baseline jet classifiers 
to provide statistical context for LLM performance evaluation.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tqdm import tqdm


def extract_high_level_features(X):
    """Extract hand-crafted physics features from jet data (from EDA notebook)."""
    n_jets = X.shape[0]
    features = []
    
    for i in range(n_jets):
        jet = X[i]
        mask = jet[:, 0] > 0
        pt = jet[mask, 0]
        
        if len(pt) == 0:
            features.append([0] * 8)
            continue
        
        pt_sorted = np.sort(pt)[::-1]
        pt_sum = pt.sum()
        
        feat = [
            len(pt),                                              # multiplicity
            pt.mean(),                                            # mean pt
            pt.std(),                                             # pt spread
            pt.max(),                                             # leading pt
            np.median(pt),                                        # median pt
            pt_sorted[0] / pt_sum if pt_sum > 0 else 0,         # leading pt fraction
            pt_sorted[:3].sum() / pt_sum if pt_sum > 0 else 0,  # top-3 pt fraction
            pt_sorted[:5].sum() / pt_sum if pt_sum > 0 else 0,  # top-5 pt fraction
        ]
        features.append(feat)
    
    return np.array(features)


def simple_multiplicity_classifier(X, threshold=38):
    """
    Simple classifier based on multiplicity threshold.
    
    Jets with > threshold particles → gluon (0)
    Jets with ≤ threshold particles → quark (1)
    """
    multiplicities = np.sum(X[:, :, 0] > 0, axis=1)
    return (multiplicities <= threshold).astype(int)


def bootstrap_performance(y_true, y_pred, n_bootstrap=100):
    """
    Compute bootstrapped performance metrics from predictions.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    n_bootstrap : int
        Number of bootstrap samples (default: 100)
    
    Returns
    -------
    dict
        Dictionary with mean, std, and confidence intervals for accuracy and AUC
    """
    n_samples = len(y_true)
    accuracies = []
    aucs = []
    
    print(f"Bootstrapping {n_bootstrap} samples from {n_samples} predictions...")
    for _ in tqdm(range(n_bootstrap)):
        # Bootstrap resample indices
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        # Compute metrics
        acc = accuracy_score(y_true_boot, y_pred_boot)
        try:
            auc = roc_auc_score(y_true_boot, y_pred_boot)
        except:
            auc = 0.5  # If all same class
        
        accuracies.append(acc)
        aucs.append(auc)
    
    accuracies = np.array(accuracies)
    aucs = np.array(aucs)
    
    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'accuracy_ci_low': np.percentile(accuracies, 2.5),
        'accuracy_ci_high': np.percentile(accuracies, 97.5),
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'auc_ci_low': np.percentile(aucs, 2.5),
        'auc_ci_high': np.percentile(aucs, 97.5),
    }


def main():
    """Run bootstrap analysis for baseline classifiers."""
    print("="*70)
    print("BOOTSTRAP ERROR BAR ANALYSIS")
    print("="*70)
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'qg_jets.npz'
    print(f"\nLoading data from: {data_path}")
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"Dataset: {len(y)} jets ({np.mean(y)*100:.1f}% quark)")
    
    # Split into train and test sets
    n_train = 5000
    n_test = 1000
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:n_train+n_test]
    y_test = y[n_train:n_train+n_test]
    
    print(f"\nTrain set: {n_train} jets")
    print(f"Test set: {n_test} jets (for bootstrap evaluation)")
    
    # Define and train classifiers ONCE on train set
    print("\n" + "="*70)
    print("TRAINING CLASSIFIERS (once each on train set)")
    print("="*70)
    
    classifiers = {}
    
    # 1. Multiplicity cut (no training needed, just apply to test set)
    print("\n1. Multiplicity Cut (threshold=38)")
    y_pred = simple_multiplicity_classifier(X_test, threshold=38)
    classifiers['Multiplicity Cut (threshold=38)'] = y_pred
    print(f"   Predictions generated on test set")
    
    # 2. Logistic Regression (multiplicity)
    print("\n2. Logistic Regression (multiplicity only)")
    mult_train = np.sum(X_train[:, :, 0] > 0, axis=1).reshape(-1, 1)
    mult_test = np.sum(X_test[:, :, 0] > 0, axis=1).reshape(-1, 1)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(mult_train, y_train)
    y_pred = clf.predict(mult_test)
    classifiers['Logistic Regression (multiplicity)'] = y_pred
    print(f"   Trained on {n_train} jets, predictions on test set")
    
    # 3. Logistic Regression (8 features)
    print("\n3. Logistic Regression (8 features)")
    features_train = extract_high_level_features(X_train)
    features_test = extract_high_level_features(X_test)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_train, y_train)
    y_pred = clf.predict(features_test)
    classifiers['Logistic Regression (8 features)'] = y_pred
    print(f"   Trained on {n_train} jets, predictions on test set")
    
    # 4. XGBoost (8 features)
    print("\n4. XGBoost (8 features)")
    print("   Training (this may take a minute)...")
    clf = XGBClassifier(
        n_estimators=100, 
        max_depth=5, 
        random_state=42, 
        eval_metric='logloss',
        verbosity=1  # Show progress: 0=silent, 1=warning, 2=info, 3=debug
    )
    clf.fit(features_train, y_train, verbose=True)  # Show training progress every 10 rounds
    y_pred = clf.predict(features_test)
    classifiers['XGBoost (8 features)'] = y_pred
    print(f"   Trained on {n_train} jets, predictions on test set")
    
    # Run bootstrap on predictions for each classifier
    results = {}
    print("\n" + "="*70)
    print("BOOTSTRAPPING PREDICTIONS")
    print("="*70)
    
    for name, y_pred in classifiers.items():
        print(f"\n{'='*70}")
        print(f"Classifier: {name}")
        print(f"{'='*70}")
        
        result = bootstrap_performance(y_test, y_pred, n_bootstrap=100)
        results[name] = result
        
        print(f"\nResults (100 bootstrap resamples of {n_test} test predictions):")
        print(f"  Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"            95% CI: [{result['accuracy_ci_low']:.4f}, {result['accuracy_ci_high']:.4f}]")
        print(f"  AUC:      {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
        print(f"            95% CI: [{result['auc_ci_low']:.4f}, {result['auc_ci_high']:.4f}]")
    
    # Save results
    output_path = Path(__file__).parent.parent / 'results' / 'bootstrap_baselines.npz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(output_path, **{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', ''): result 
                             for name, result in results.items()})
    
    print(f"\n{'='*70}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Baseline Performance with Error Bars")
    print("="*70)
    print(f"{'Classifier':<40} {'Accuracy':<20} {'AUC':<20}")
    print("-"*70)
    for name, result in results.items():
        acc_str = f"{result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}"
        auc_str = f"{result['auc_mean']:.3f} ± {result['auc_std']:.3f}"
        print(f"{name:<40} {acc_str:<20} {auc_str:<20}")
    print("="*70)


if __name__ == "__main__":
    main()

