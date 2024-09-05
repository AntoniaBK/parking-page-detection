import pandas as pd
# Import data
simple_data = pd.read_csv("simple_data.csv")
data3 = pd.read_csv("duplicated_data.csv")
all_data = pd.concat([simple_data, data3], axis=0)
y = 'parking'
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
    'domain_in_title',
    'domain_in_text',
    'in_warninglist',
    
    
    
    'average_link_length_same_domain',
    'number_different_link_domains'
    
    
    
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
structural_features = [
    'uuid',
    'tags',
    'url',
    'language',
    'ip',
    'structural_hash'
]
'''X = data[structural_features]
X.describe()
X.info()
print(X.nunique())

hashes = data['structural_hash'].values.tolist()
#print(len(set(hashes)))
i=0
number_per_hash = {}
for hash in hashes:
    if hash in number_per_hash.keys():
        number_per_hash[hash] += 1
    else:
        number_per_hash[hash] = 1
#print(number_per_hash)
for n in number_per_hash.keys():
    if number_per_hash[n] > 1:
        print(n)
        i+=1'''

ds = simple_data[simple_data['parking'] == True]
print("\nData without 1")
print(len(ds))
print(len(simple_data))

ds = data3[data3['parking'] == True]
print("\ndata3")
print(len(ds))
print(len(data3))

ds = all_data[all_data['parking'] == True]
print("\ndata3")
print(len(ds))
print(len(all_data))



