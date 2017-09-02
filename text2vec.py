

import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords # stop words
from nltk.tokenize import wordpunct_tokenize,word_tokenize # splits sentences into words
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import re


from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

class TextProcessing:
    def __init__(self,rawdatatrain,rawdatatest):
        self.text=pd.DataFrame(pd.concat([rawdatatrain.Text,rawdatatest.Text]))
        self.text=self.text.reset_index()
        self.text.drop(["index"],axis=1,inplace=True)
        


    def tokenize_and_stem(self,text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems



    def tfidfvect(self,text):


        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000,min_df=0.3,
                                          stop_words='english',
                                         use_idf=True, tokenizer=self.tokenize_and_stem)

        tfidf_matrix = tfidf_vectorizer.fit_transform(self.text.Text) #fit the vectorizer to synopses')
        train_tfidf_matrix=tfidf_matrix[:3321]
        test_tfidf_matrix=tfidf_matrix[3321:]
        return train_tfidf_matrix,test_tfidf_matrix

if __name__=="__main__":
    train=pickle.load( open( "FinalTrain.pickle", "rb" ) )
    test=pickle.load( open( "FinalTest.pickle", "rb" ) )
    textProcessing=TextProcessing(train,test)
    train_tfidf,test_tfidf=textProcessing.tfidfvect(textProcessing.text)
    with open('train_tfidf.pickle','wb') as f:
        pickle.dump(train_tfidf,f)
    with open('test_tfidf.pickle','wb') as f:
        pickle.dump(test_tfidf,f)
        
