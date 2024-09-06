import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV


import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('simple_data.csv')
y = data.parking

target = ['parking']
features = [
            'kword_auction',
            'kword_domain',
            'kword_regist',
            'kword_price',
            'kword_offer',
            'kword_servic',
            'kword_host',
            'kword_websit',
            'kword_contact',
            'kword_site',
            'kword_transfer',
            'kword_héberge',
            'kword_internet',
            'kword_serveur',
            'kword_découvr',
            'kword_mainten',
            'number_domain_keywords_en',
            'number_parking_kewords_en',
            'number_together_in_line_keywords_en',
            'keyword_in_title_en',
            'stemmed_keyword_in_title',
            'presence_of_nav',
            'text_alpha_length',
            'number_frames',
            'number_images',
            'domain_in_title',
            'domain_in_text',
            'in_warninglist',
            'number_redirects',
            'different_final_domain',
            'number_links',
            'number_link_same_domain',
            'average_link_length_same_domain',
            'number_different_link_domains',
            'average_link_length',
            'maximum_link_length',
            'link_to_text_ratio',
            'number_non_link_characters',
            'third_party_requests_ratio',
            'third_party_data_ratio',
            'third_party_html_content_ratio',
            'initial_response_size',
            'initial_response_ratio'
        ]

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

def plot_importances(importances, name, file):
    plt.figure(figsize=(10, 6))
    plt.barh(importances['Feature'], importances['Importance'], color='skyblue', height=0.6)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'{name} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(file)
    plt.close()

# Random Forest with fine-tuned parameters
print("Random Forest\n")
modelRF = RandomForestClassifier(
    criterion='gini',  # Gini impurity
    max_depth=None,  # No limit on tree depth
    min_samples_split=2,  # Minimum samples required to split an internal node
    n_estimators=300,  # Number of trees in the forest
    random_state=42
)
evaluate_model_cv(modelRF, X, y, "Random Forest")

# Train the Random Forest model on the whole training data and get predictions
modelRF.fit(X, y)

# Feature importances for Random Forest
importances_rf = modelRF.feature_importances_
importance_df_rf = pd.DataFrame({'Feature': X.columns, 'Importance': importances_rf})
importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)

# Plot Random Forest Feature Importances
plot_importances(importance_df_rf, "Random Forest", 'rf_all_feature_importance.png')

# Gradient Boosting with fine-tuned parameters
print("\nGradient Boosting\n")
modelGB = GradientBoostingClassifier(
    learning_rate=0.2,  # Learning rate shrinks contribution of each tree
    max_depth=5,  # Maximum depth of the individual regression estimators
    n_estimators=300,  # Number of boosting stages
    random_state=42
)
evaluate_model_cv(modelGB, X, y, "Gradient Boosting")


modelGB.fit(X, y)

# Feature importances for Gradient Boosting
importances_gb = modelGB.feature_importances_
importance_df_gb = pd.DataFrame({'Feature': X.columns, 'Importance': importances_gb})
importance_df_gb = importance_df_gb.sort_values(by='Importance', ascending=False)

# Plot Gradient Boosting Feature Importances
plot_importances(importance_df_gb, "Gradient Boosting", 'gb_all_feature_importance.png')

numbers = [40, 35, 30, 25, 20, 15, 10, 5]
for n in numbers:
    gb_important_features = importance_df_gb.head(n)['Feature']
    rf_important_features = importance_df_rf.head(n)['Feature']
    common_important_features = list(set(rf_important_features) & set(gb_important_features))
    print("Common important features:")
    print(common_important_features)
    print(len(common_important_features))

    X_common = data[common_important_features]


    # Random Forest with selected features
    modelRF = RandomForestClassifier(
        criterion='gini',  # Gini impurity
        max_depth=None,  # No limit on tree depth
        min_samples_split=2,  # Minimum samples required to split an internal node
        n_estimators=300,  # Number of trees in the forest
        random_state=42
    )
    evaluate_model_cv(modelRF, X_common, y, "Random forest Common Features")
    modelRF.fit(X_common, y)
    # Feature importances for Random Forest
    importances_rf = modelRF.feature_importances_
    importance_df_rf = pd.DataFrame({'Feature': X_common.columns, 'Importance': importances_rf})
    importance_df_rf = importance_df_rf.sort_values(by='Importance', ascending=False)
    # Plot Random Forest Feature Importances
    plot_importances(importance_df_rf, "Random Forest", f'rf_{n}_feature_importance.png')

    
    # Gradient Boosting with selected features
    modelGB = GradientBoostingClassifier(
        learning_rate=0.2,  # Learning rate shrinks contribution of each tree
        max_depth=5,  # Maximum depth of the individual regression estimators
        n_estimators=300,  # Number of boosting stages
        random_state=42
    ) 
    evaluate_model_cv(modelGB, X_common, y, "Gradient Boosting Common Features")
    modelGB.fit(X_common, y)
    # Feature importance
    importances_gb = modelGB.feature_importances_
    importance_df_gb = pd.DataFrame({'Feature': X_common.columns, 'Importance': importances_gb})
    importance_df_gb = importance_df_gb.sort_values(by='Importance', ascending=False)
    # Plot Gradient Boosting Feature Importances
    plot_importances(importance_df_gb, "Gradient Boosting", f'gb_{n}_feature_importance.png')

    
