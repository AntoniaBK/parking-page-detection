import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split


import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('simple_data.csv')
y = data.parking

target = ['parking']
features = ['keyword_in_title_en', 'text_alpha_length', 'kword_domain']

X = data[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
def evaluate_model_cv(model, X, y, model_name="Model"):
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_macro',
        'recall': 'recall_macro',
        'f1': 'f1_macro'
    }
    # Perform cross-validation
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
    
    print(f"{model_name} Cross-Validation Results:")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.2f} (+/- {cv_results['test_accuracy'].std():.2f})")
    print(f"Precision: {cv_results['test_precision'].mean():.2f} (+/- {cv_results['test_precision'].std():.2f})")
    print(f"Recall: {cv_results['test_recall'].mean():.2f} (+/- {cv_results['test_recall'].std():.2f})")
    print(f"F1-Score: {cv_results['test_f1'].mean():.2f} (+/- {cv_results['test_f1'].std():.2f})\n")

modelRF = RandomForestClassifier(
    criterion='gini',  # Gini impurity
    max_depth=None,  # No limit on tree depth
    min_samples_split=2,  # Minimum samples required to split an internal node
    n_estimators=300,  # Number of trees in the forest
    random_state=42
)
evaluate_model_cv(modelRF, X, y, "Random Forest")