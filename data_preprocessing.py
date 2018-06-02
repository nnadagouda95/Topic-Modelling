# -*- coding: utf-8 -*-
"""
data_preprocessing.py
"""

import numpy as np # linear algebra
import pandas as pd #for io
import nltk # for text preprocessing
import re
  
def data_preprocess(path):

    english_stopwords=stopwords.words('english')
    english_punctuations=[',','.',':',';','?','(',')','[',']','&','!','*','@','#','$','%',\
                        '<','>','`','``',"''",'--','|']

    text=[]
    label=[]
    i=0
    #read files from disks
    for filelist in os.listdir(path):
        print(filelist)
        for filename in os.listdir(path+'/'+filelist):

            #clear all parameters
            data=[];data_tokenized=[];data_fliter_stopwords=[];data_flitered=[];data_stemmed=[];

            #read from the documents
            data=[open(path+'/'+filelist+'/'+filename).read()]


            #tokenized the lower words
            #nltk.download('punkt')
            data_tokenized=[[word.lower() for word in word_tokenize(document.decode('iso-8859-15'))] \
                              for document in data]

            #remove stopwords and punctuations from the text
            data_filter_stopwords=[[word for word in document if not word in english_stopwords]\
                                     for document in data_tokenized]

            data_filtered=[[word for word in document if not word in english_punctuations]\
                             for document in data_filter_stopwords]

            #stemming the text
            st=LancasterStemmer()
            data_stemmed=[[st.stem(word) for word in document] for document in data_flitered]


            text.append(data_stemmed[0])
            label.append(i)
        i=i+1
    return text, np.array(label)

def freq_terms(text,top_k):
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    # tf = tf_vectorizer.fit_transform(doc_train)
    tf = tf_vectorizer.fit_transform(text)

    return tf

if __name__ == "__main__":
    path_train='./bbcsport_train/'
    text,train_label = ps.data_preprocess(path_train)
    top_k = 2000
    tf = freq_terms(text, top_k)
    train_data =  dict(zip('data':tf, 'label':train_label))
    savemat('bbc_train_data.mat', train_data)
