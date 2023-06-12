import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
corpus = [
    'I would like to check this tt',
    'How about one more ttt',
    'Aim is to capture the house xx from the corpus',
    'xx of xx in a documeyynt is called term xx'
]

X = tfidf.fit_transform(corpus)
feature_names = np.array(tfidf.get_feature_names_out())

z = tfidf.get_feature_names_out()
print(X.shape)
print("ffffffdsdfsdf")

print("ffeat name ", feature_names)


new_doc = ['can key words in this new document be identified?',
           'idf is the inverse document frequency caculcated for each of the words']
responses = tfidf.transform(new_doc)


def get_top_tf_idf_words(response, top_n=3):
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]]
  
print([get_top_tf_idf_words(response,2) for response in responses])

#[array(['key', 'words'], dtype='<U9'),
#array(['frequency', 'words'], dtype='<U9')