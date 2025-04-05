import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import os
import glob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, roc_curve, auc, precision_recall_curve, 
                            average_precision_score, classification_report)
from sklearn.model_selection import GridSearchCV

# Set up plot style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
sns.set_palette(sns.color_palette(colors))

def load_data():
    """Load preprocessed data"""
    data_dir = max(glob.glob('preprocessed_data_*'), key=os.path.getctime, default=None)
    if not data_dir:
        raise FileNotFoundError("No preprocessed data directory found")
    print(f"Loading preprocessed data from {data_dir}...")
    
    X_train = np.load(f'{data_dir}/X_train.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    try:
        with open(f'{data_dir}/selected_features.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
    except:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    return X_train, X_test, y_train, y_test, feature_names

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    results = {}
    y_pred_dict = {}
    y_prob_dict = {}
    training_times = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Store probability predictions if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        y_pred_dict[name] = y_pred
        y_prob_dict[name] = y_prob
        training_times[name] = train_time
        
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Training time: {train_time:.2f} seconds")
        print(f"Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    return models, results, y_pred_dict, y_prob_dict, training_times

def plot_confusion_matrices(y_test, y_pred_dict, output_dir):
    """Plot confusion matrices for all models"""
    plt.figure(figsize=(20, 15))
    
    for i, (name, y_pred) in enumerate(y_pred_dict.items(), 1):
        plt.subplot(3, 2, i)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malware'],
                   yticklabels=['Benign', 'Malware'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'))
    plt.close()

def plot_roc_curves(y_test, y_prob_dict, output_dir):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, y_prob in y_prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()

def plot_precision_recall_curves(y_test, y_prob_dict, output_dir):
    """Plot precision-recall curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for name, y_prob in y_prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()

def plot_metrics_comparison(results, output_dir):
    """Plot comparison of metrics across models"""
    df_results = pd.DataFrame(results).T
    
    plt.figure(figsize=(12, 8))
    df_results.plot(kind='bar', figsize=(12, 8))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    return df_results

def analyze_feature_importance(models, feature_names, output_dir):
    """Analyze feature importance for tree-based models"""
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance - {name}')
            plt.bar(range(len(indices[:20])), importances[indices[:20]], align='center')
            plt.xticks(range(len(indices[:20])), [feature_names[i] for i in indices[:20]], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_importance_{name.replace(" ", "_").lower()}.png'))
            plt.close()
            
            print(f"\nTop 10 features for {name}:")
            for i, idx in enumerate(indices[:10]):
                print(f"{i+1}. {feature_names[idx]} ({importances[idx]:.4f})")

def tune_best_model(X_train, X_test, y_train, y_test, model_name, results, output_dir):
    """Tune hyperparameters for the best performing model"""
    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    
    if model_name:
        best_model_name = model_name
    
    print(f"\nTuning hyperparameters for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)
    
    elif best_model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
        base_model = GradientBoostingClassifier(random_state=42)
    
    elif best_model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        base_model = SVC(probability=True, random_state=42)
    
    elif best_model_name == 'Neural Network':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        base_model = MLPClassifier(max_iter=1000, random_state=42)
    
    else:  # Logistic Regression
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2', None],  # Changed to only include compatible options
            'solver': ['lbfgs', 'newton-cg', 'sag']
        }
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nOptimized {best_model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    with open(os.path.join(output_dir, f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    return best_model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main(tune_model=None):
    """Main function"""
    # Define output directory
    output_dir = './trainModel'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # Train and evaluate multiple models
    models, results, y_pred_dict, y_prob_dict, training_times = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(y_test, y_pred_dict, output_dir)
    
    # Plot ROC curves
    plot_roc_curves(y_test, y_prob_dict, output_dir)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(y_test, y_prob_dict, output_dir)
    
    # Plot metrics comparison
    df_results = plot_metrics_comparison(results, output_dir)
    
    # Analyze feature importance
    analyze_feature_importance(models, feature_names, output_dir)
    
    # Print comparison table
    print("\nModel Performance Comparison:")
    print(df_results)
    
    # Print training times
    print("\nTraining Times (seconds):")
    for name, time_taken in training_times.items():
        print(f"{name}: {time_taken:.2f}")
    
    # Find the best model
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\nBest model based on F1 score: {best_model_name} (F1 = {results[best_model_name]['f1']:.4f})")
    
    # Tune the best model (or specified model)
    if tune_model == 'all':
        for model_name in results.keys():
            best_model, best_results = tune_best_model(X_train, X_test, y_train, y_test, model_name, results, output_dir)
            results[f"{model_name} (Tuned)"] = best_results
    else:
        best_model, best_results = tune_best_model(X_train, X_test, y_train, y_test, tune_model, results, output_dir)
        results[f"{best_model_name} (Tuned)"] = best_results
    
    # Final comparison
    df_final_results = pd.DataFrame(results).T
    print("\nFinal Model Performance Comparison (including tuned model):")
    print(df_final_results)
    
    # Save final results
    df_final_results.to_csv(os.path.join(output_dir, 'model_comparison_results.csv'))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate multiple models for Android malware detection')
    parser.add_argument('--tune', choices=['all', 'Random Forest', 'Gradient Boosting', 'SVM', 'Neural Network', 'Logistic Regression'],
                       help='Specify which model to tune hyperparameters for')
    
    args = parser.parse_args()
    
    main(tune_model=args.tune)