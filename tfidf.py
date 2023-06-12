import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

mylist = [
'This is the first document.',
'This document is the second document.',
'And this is the third one.',
'Is this the first document?',
'Is this the second cow?, why is it blue?']


df = pd.DataFrame({"texts": mylist})
tfidf_vectorizer = TfidfVectorizer(ngram_range=[1, 1])
tfidf_separate = tfidf_vectorizer.fit_transform(df["texts"])


word_lst = tfidf_vectorizer.get_feature_names()
count_lst = tfidf_separate.toarray().sum(axis=0)

vocab_df = pd.DataFrame((zip(word_lst,count_lst)),
                          columns= ["vocab","tfidf_value"])

vocab_df.sort_values(by="tfidf_value",ascending=False)
print(vocab_df)