import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

def load_data():
    df1 = pd.read_csv('./数据集1/cleaned_data.csv')
    df2 = pd.read_csv('./数据集2/australian_preprocessed.csv')
    df3 = pd.read_csv('./数据集3/german_preprocessed.csv')
    
    df1['RESULT'] = df1['RESULT'].map({'Good': 1, 'Bad': 0})
    df3['class'] = df3['class'].map({'good': 1, 'bad': 0})
    
    return df1, df2, df3

def train_models(X_train, y_train, X_test, y_test, dataset_name, use_class_weight=False):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Linear SVM': LinearSVC(random_state=42, max_iter=1000),
        'RBF SVM': SVC(random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    if use_class_weight:
        models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        models['Linear SVM'] = LinearSVC(random_state=42, max_iter=1000, class_weight='balanced')
        models['RBF SVM'] = SVC(random_state=42, probability=True, class_weight='balanced')
        models['Random Forest'] = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    results = {}
    roc_data = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'roc_auc': round(roc_auc, 3),
            'confusion_matrix': cm
        }
        
        roc_data[name] = {'proba': y_pred_proba, 'y_true': y_test}
    
    print(f"\n=== {dataset_name} ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1:        {metrics['f1']}")
        print(f"  AUC-ROC:   {metrics['roc_auc']}")
        print(f"  Confusion Matrix:\n    {metrics['confusion_matrix'][0]}\n    {metrics['confusion_matrix'][1]}")
    
    return results, roc_data

def plot_roc_curves(roc_data_dict, dataset_name):
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(8, 6))
    
    for model_name, roc_data in roc_data_dict.items():
        fpr, tpr, _ = roc_curve(roc_data['y_true'], roc_data['proba'])
        roc_auc = roc_auc_score(roc_data['y_true'], roc_data['proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {dataset_name}')
    plt.legend(loc='lower right')
    plt.savefig(f'figures/roc_curve_{dataset_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df1, df2, df3 = load_data()
    
    X1 = df1.drop('RESULT', axis=1).values
    y1 = df1['RESULT'].values
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=42)
    scaler1 = StandardScaler()
    X_train1_scaled = scaler1.fit_transform(X_train1)
    X_test1_scaled = scaler1.transform(X_test1)
    
    X2 = df2.drop('A15', axis=1).values
    y2 = df2['A15'].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=42)
    scaler2 = StandardScaler()
    X_train2_scaled = scaler2.fit_transform(X_train2)
    X_test2_scaled = scaler2.transform(X_test2)
    
    X3 = df3.drop('class', axis=1).values
    y3 = df3['class'].values
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3, random_state=42)
    scaler3 = StandardScaler()
    X_train3_scaled = scaler3.fit_transform(X_train3)
    X_test3_scaled = scaler3.transform(X_test3)
    
    results1, roc_data1 = train_models(X_train1_scaled, y_train1, X_test1_scaled, y_test1, 'Dataset 1: Indonesian Credit Data')
    results2, roc_data2 = train_models(X_train2_scaled, y_train2, X_test2_scaled, y_test2, 'Dataset 2: Australian Credit Approval')
    results3, roc_data3 = train_models(X_train3_scaled, y_train3, X_test3_scaled, y_test3, 'Dataset 3: German Credit Data', use_class_weight=True)
    
    plot_roc_curves(roc_data1, 'Dataset 1')
    plot_roc_curves(roc_data2, 'Dataset 2')
    plot_roc_curves(roc_data3, 'Dataset 3')
    
    all_results = {
        'dataset1': {'name': 'Indonesian Credit Data', 'results': results1},
        'dataset2': {'name': 'Australian Credit Approval', 'results': results2},
        'dataset3': {'name': 'German Credit Data', 'results': results3}
    }
    
    with open('experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n=== Results saved to experiment_results.json ===")

if __name__ == '__main__':
    main()