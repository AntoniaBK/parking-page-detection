from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def predictByWarninglist(X: pd.DataFrame) -> list[int]:
    predictions = []
    for _, x in X.iterrows():
        predictions.append(x['in_warninglist'])
    return predictions

'''False positives  (should be labelled positive)
0b3b8b0a-8fb4-4c7d-bea5-9e33c629d923 (wordpress)
05f7a59f-5c94-41e5-8e0c-22b2f05dcd96 (wordpress site privé)
c64ca8a6-c24a-4deb-834b-46b2233f0d43 (wordpress)
'''
# Import data
data1 = pd.read_csv("simple_data.csv")
data2 = pd.read_csv("duplicated_data.csv")

data = pd.concat([data1, data2], axis=0)
#data = data1

y = data.parking
target = ['parking']
keywords = [
    #'kword_auction',
    'kword_domain',
    #'kword_regist',
    'kword_price',
    #'kword_offer',
    'kword_servic',
    #'kword_host',
    #'kword_websit',
    'kword_contact',
    'kword_site',
    'kword_transfer',
    'kword_héberge',
    'kword_internet',
    #'kword_serveur',
    #'kword_découvr',
    #'kword_mainten'
]
features = [
    'number_domain_keywords_en',
    'number_parking_kewords_en',
    'number_together_in_line_keywords_en',
    #'keyword_in_title_en',
    'stemmed_keyword_in_title',
    'presence_of_nav',
    'text_alpha_length',
    'number_frames',
    'number_images',
    'domain_in_title',
    'domain_in_text',
    'in_warninglist',
    'number_redirects',
    #'different_final_domain',
    'number_links',
    'number_link_same_domain',
    'average_link_length_same_domain',
    'number_different_link_domains',
    'average_link_length',
    'maximum_link_length',
    'link_to_text_ratio',
    #'number_non_link_characters',
    'third_party_requests_ratio',
    #'third_party_data_ratio',
    'third_party_html_content_ratio',
    'initial_response_size',
    'initial_response_ratio'
]
ml_features = features  + keywords 
X = data[ml_features]
y_pred = predictByWarninglist(X)


accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Example confusion matrix
cm = confusion_matrix(y, y_pred)
# Labels for the axes
labels = ['Negative', 'Positive']

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Warninglist Approach')
plt.savefig('duplicated_cm_warninglist.png')
