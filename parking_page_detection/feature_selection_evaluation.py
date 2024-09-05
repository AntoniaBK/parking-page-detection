# feature_selection_evaluation.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def load_data():
    """
    Loads the two datasets.
    """
    simple_data = pd.read_csv("simple_data.csv")
    duplicates = pd.read_csv("duplicated_data.csv")
    all_data = pd.concat([simple_data, duplicates], axis=0)

    return {'Simple': simple_data, 'Duplicates': all_data}

def get_feature_sets():
    """
    Defines different sets of features.
    """
    all_features = [
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
    manual_selection = [
        'kword_domain',
        'number_parking_kewords_en',
        'number_together_in_line_keywords_en',
        'stemmed_keyword_in_title',
        'text_alpha_length',
        'number_images',
        'domain_in_title',
        'number_links',
        'number_link_same_domain',
        'number_different_link_domains',
        'initial_response_size',
    ]
    keyword_features = [
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
    ]
    parking_sensores = [
            'average_link_length',
            'maximum_link_length',
            'link_to_text_ratio',
            'number_link_same_domain',
            'number_non_link_characters',
            'number_redirects',
            'different_final_domain',
            'third_party_requests_ratio',
            'third_party_data_ratio',
            'third_party_html_content_ratio',
            'initial_response_size',
            'initial_response_ratio',
            'number_links',
            'text_alpha_length',
            'number_frames',
            'number_images'
        ]
    # First with 'in_warninglist', second without 'in_warninglist'
    mutual_information1 = ['initial_response_size', 'stemmed_keyword_in_title', 'number_link_same_domain', 'keyword_in_title_en', 'third_party_data_ratio', 'maximum_link_length', 'number_redirects', 'number_links', 'initial_response_ratio', 'third_party_requests_ratio', 'in_warninglist', 'kword_domain', 'presence_of_nav', 'number_together_in_line_keywords_en', 'number_images', 'number_different_link_domains', 'text_alpha_length', 'average_link_length_same_domain', 'third_party_html_content_ratio', 'domain_in_title', 'number_non_link_characters', 'average_link_length']
    mutual_information2 = ['number_images', 'number_together_in_line_keywords_en', 'number_links', 'third_party_requests_ratio', 'number_non_link_characters', 'average_link_length_same_domain', 'number_redirects', 'initial_response_ratio', 'number_link_same_domain', 'initial_response_size', 'presence_of_nav', 'third_party_data_ratio', 'third_party_html_content_ratio', 'keyword_in_title_en', 'maximum_link_length', 'text_alpha_length', 'kword_offer', 'domain_in_title', 'average_link_length', 'stemmed_keyword_in_title', 'number_different_link_domains', 'kword_domain']
    rfe1 = ['stemmed_keyword_in_title', 'number_link_same_domain', 'keyword_in_title_en', 'third_party_data_ratio', 'number_redirects', 'number_links', 'in_warninglist', 'kword_domain', 'number_domain_keywords_en', 'kword_regist', 'number_together_in_line_keywords_en', 'number_images', 'kword_auction', 'number_different_link_domains', 'number_frames', 'average_link_length_same_domain', 'domain_in_title', 'number_parking_kewords_en', 'number_non_link_characters', 'kword_servic', 'kword_websit', 'kword_internet']
    rfe2 = ['number_images', 'number_together_in_line_keywords_en', 'kword_contact', 'number_parking_kewords_en', 'number_links', 'number_non_link_characters', 'kword_site', 'kword_auction', 'number_link_same_domain', 'initial_response_size', 'presence_of_nav', 'third_party_data_ratio', 'keyword_in_title_en', 'kword_regist', 'text_alpha_length', 'domain_in_title', 'average_link_length', 'number_frames', 'stemmed_keyword_in_title', 'number_different_link_domains', 'kword_servic', 'kword_domain']
    intersection1 = ['number_together_in_line_keywords_en', 'number_images', 'number_link_same_domain', 'stemmed_keyword_in_title', 'number_different_link_domains', 'average_link_length_same_domain', 'domain_in_title', 'number_non_link_characters', 'keyword_in_title_en', 'third_party_data_ratio', 'in_warninglist', 'number_redirects', 'kword_domain', 'number_links']
    intersection2 = ['number_images', 'domain_in_title', 'number_together_in_line_keywords_en', 'average_link_length', 'keyword_in_title_en', 'stemmed_keyword_in_title', 'number_different_link_domains', 'text_alpha_length', 'kword_domain', 'number_link_same_domain', 'presence_of_nav', 'number_links', 'initial_response_size', 'number_non_link_characters', 'third_party_data_ratio']
    return {
            'AllFeatures': all_features,
            'ManualSelection': manual_selection,
            'KeywordFeatures': keyword_features,
            'ParkingSensores': parking_sensores, 
            'MutualInformation1': mutual_information1, 
            'MutualInformation2': mutual_information2, 
            'RFE1': rfe1,
            'RFE2': rfe2,
            'Intersection1': intersection1,
            'Intersection2': intersection2
            }

def define_models():
    """
    Defines models and their hyperparameter grids for tuning.
    """
    models = {
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }
    return models

def evaluate_model(model, params, X, y):
    """
    Performs hyperparameter tuning and cross-validation for a given model.
    Returns a dictionary of evaluation metrics.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    
    # Cross-validation scores
    accuracies = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    precisions = cross_val_score(best_model, X, y, cv=skf, scoring='precision', n_jobs=-1)
    recalls = cross_val_score(best_model, X, y, cv=skf, scoring='recall', n_jobs=-1)
    f1s = cross_val_score(best_model, X, y, cv=skf, scoring='f1', n_jobs=-1)
    
    metrics = {
        'BestParams': grid_search.best_params_,
        'Accuracy_Mean': accuracies.mean(),
        'Accuracy_STD': accuracies.std(),
        'Precision_Mean': precisions.mean(),
        'Precision_STD': precisions.std(),
        'Recall_Mean': recalls.mean(),
        'Recall_STD': recalls.std(),
        'F1_Score_Mean': f1s.mean(),
        'F1_Score_STD': f1s.std()
    }
    return metrics

def compile_results(datasets, feature_sets, models):
    """
    Compiles evaluation results for all combinations into a DataFrame.
    """
    results = []
    for dataset_name, data in datasets.items():
        X_full = data.drop('parking', axis=1)
        y = data['parking']
        for feature_set_name, features in feature_sets.items():
            X = X_full[features]
            for model_name, model_info in models.items():
                print(f'Evaluating {model_name} on {dataset_name} with {feature_set_name}...')
                metrics = evaluate_model(model_info['model'], model_info['params'], X, y)
                result = {
                    'Dataset': dataset_name,
                    'FeatureSet': feature_set_name,
                    'Model': model_name,
                    **metrics
                }
                results.append(result)
    results_df = pd.DataFrame(results)
    return results_df

def export_to_latex(df, filename='evalutaion_results_table.tex'):
    """
    Exports the results DataFrame to a LaTeX table.
    """
    df.to_csv("evalutaion_results.csv")
    latex_table = df.to_latex(index=False, float_format="%.4f", bold_rows=True)
    with open(filename, 'w') as f:
        f.write(latex_table)
    print(f'LaTeX table saved to {filename}')

def main():
    datasets = load_data()
    feature_sets = get_feature_sets()
    models = define_models()
    results_df = compile_results(datasets, feature_sets, models)
    export_to_latex(results_df)

if __name__ == '__main__':
    main()
