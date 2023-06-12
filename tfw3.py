from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import tokenize
from operator import itemgetter
import math


tf_idf_vect = TfidfVectorizer(ngram_range=(1,1))
tf_idf_vect.fit(sample_data)
final_tf_idf = tf_idf_vect.transform(sample_data)

tf_idf_vect = TfidfVectorizer(ngram_range=(1,1), max_features=1000)


