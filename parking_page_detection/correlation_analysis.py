import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

data1 = pd.read_csv("data/data_selected_captures.csv")
data2 = pd.read_csv("data/corrected_additional_captures.csv")
data = pd.concat([data1, data2], axis=0)
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
ml_features = target + parking_sensores
X = data[ml_features]

#spearman_corr ()
plt.figure(figsize=(25, 20))
sns.heatmap(X.corr(method='spearman'), annot=True, cmap='coolwarm', fmt='.2f')
plt.savefig('spearman_heatmap_parking_sensores.png')

# corr (linear relationships)
plt.figure(figsize=(25, 20))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.savefig('spearman_heatmap_parking_sensores.png')



