import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, RFE, SelectKBest, chi2
import xgboost as xgb
data1 = pd.read_csv("data/data_selected_captures.csv")
data2 = pd.read_csv("data/corrected_additional_captures.csv")
data = pd.concat([data1, data2], axis=0)

#print(data.duplicated())
data.loc[data.duplicated(keep=False)]

y = data.parking
target = ['parking']
keywords = [
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
    'kword_mainten'
]
features = [
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
    #'in_warninglist',
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
ml_features = features + keywords 
X = data[ml_features]

feature_names = X.columns

# Results dictionary to store rankings/scores along with feature names
results = pd.DataFrame({'Feature': feature_names})

# 1. Mutual Information
print('# Mutual Information')
mi = mutual_info_classif(X, y)
results['Mutual Information'] = mi

# 2. Recursive Feature Elimination (RFE)
print('# Recursive Feature Elimination (RFE)')
model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
rfe = RFE(model, n_features_to_select=1)
rfe.fit(X, y)
results['RFE Ranking'] = rfe.ranking_
'''
# Save the DataFrame to a CSV file
results.to_csv('feature_selection_results.csv', index=False)'''

'''# Plot the feature selection results as bar charts
fig, axes = plt.subplots(nrows=len(results.columns)-1, ncols=1, figsize=(10, 15))

for i, method in enumerate(results.columns[1:]):  # Skip the 'Feature' column
    sorted_results = results.sort_values(by=method, ascending=True)
    axes[i].barh(sorted_results['Feature'], sorted_results[method])
    axes[i].set_title(f'{method} Feature Rankings/Scores')
    axes[i].set_xlabel('Ranking/Score')
    axes[i].set_ylabel('Features')

plt.tight_layout()
plt.savefig('feature_selection_results.png')'''

mi_results = pd.DataFrame({'Feature': ml_features, 'Mutual_Information': mi})
rfe_results = pd.DataFrame({'Feature': ml_features, 'RFE_Ranking': rfe.ranking_})


top_mi_features = set(mi_results.nlargest(22, 'Mutual_Information')['Feature'])
print("\nTop MI ")
print(top_mi_features)

top_rfe_features = set(rfe_results.nsmallest(22, 'RFE_Ranking')['Feature'])
print("\nTop RFE ")
print(top_rfe_features)

common_features = top_mi_features.intersection(top_rfe_features)
print("\nIntersection ")
print(common_features)