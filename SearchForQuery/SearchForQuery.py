"""
Created on Thu Nov 28 11:58:22 2019

@author: Diana
"""

import tensorflow as tf
import tensorflow_hub as hub

import gensim
from gensim import models
from gensim import similarities

import pandas as pd
import re
import pickle

import wiki
import preprocessData as pData

class SearchForQuery:

    def __init__(self, noOfClusters):
        tf.compat.v1.disable_v2_behavior()

        self.__documents = []
        self.__documentsWithLabels = [[],[],[]]
        self.__preprocessedDocumentsList = []
        self.__noOfClusters = noOfClusters
        self.__preprocessedListForDictionary = [[],[],[]]
        self.__dictionary = []
        self.__lsi = [None] * self.__noOfClusters
        self.__indexList = [None] * self.__noOfClusters
        self.__embed = None
        self.__clf = None

        self.__processTranscripts()
        self.__LSI()
        self.__loadModel()

    def __extractData(self):
        for i in range(3):
            dataframe = pd.read_excel("LabeledTranscripts.xlsx", engine = "openpyxl", usecols = [i])
            yield dataframe.values
    
    #Process all transcripts
    def __processTranscripts(self):
        videosName,transcripts,labels = self.__extractData()

        nrOfTranscriptsToProcess = len(videosName)

        for i in range(0, nrOfTranscriptsToProcess):
            transcript = transcripts[i][0]
            videoName = videosName[i][0]
            label = labels[i][0]
            if transcript !=  "" and len(transcript)>6000:      #keep only valid transcripts for preprocessing
                preprocessedTranscript = pData.preprocess(transcript)   

                #keep all preprocessed transcripts in a container for each label
                self.__documentsWithLabels[label].append((preprocessedTranscript,videoName))
                
                #keep all preprocessed transcripts together
                self.__documents.append([preprocessedTranscript, videoName, label])

                #keep all words from transcripts
                self.__preprocessedDocumentsList.append(preprocessedTranscript.split())
        
        index = 0
        for transcript, videoName, label in self.__documents:
            self.__preprocessedListForDictionary[label].append(self.__preprocessedDocumentsList[index])
            index += 1

    def __LSI(self):
        for i in range(self.__noOfClusters):
            self.__dictionary.append(gensim.corpora.Dictionary(self.__preprocessedListForDictionary[i]))

        bow_corpus = [None] * self.__noOfClusters  

        for i in range(self.__noOfClusters):
            if( len(self.__dictionary[i]) != 0):
                #we have to create a bag of words( BoW )
                bow_corpus[i] = [self.__dictionary[i].doc2bow(doc) for doc in self.__preprocessedListForDictionary[i]]
                
                #we will transform it in a tf-idf vector
                tfidf = models.TfidfModel(bow_corpus[i]) 
                corpus_tfidf = tfidf[bow_corpus[i]]

                self.__lsi[i] = models.LsiModel(corpus = corpus_tfidf, id2word = self.__dictionary[i], num_topics = 5)
                #we will compute a similarity matrix, which it will help us later, for query
                self.__indexList[i] = similarities.MatrixSimilarity(self.__lsi[i][corpus_tfidf])
                
                #print(indexList[0])
                print(self.__lsi[i].print_topics(num_topics = 5, num_words = 10))
        

    def __loadModel(self):
        self.__embed = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")
        self.__clf = pickle.load(open("pretrained_model.sav", 'rb'))

    def resultForQuery(self, query):
        ans = wiki.wikipedia_search(query)     #search a sequence on wiki pages
        if(len(ans["itemList"]) != 0 ):             #if we have a result
                queryWiki =  (ans["itemList"][0]["description"])       #assign as query this sequence
        else:   
                queryWiki = query                  #else assign just the query
        
        
        
        with tf.compat.v1.Session() as session:
                tf.compat.v1.disable_eager_execution()
                session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
                embeddedQueryWiki = session.run(self.__embed([queryWiki]))

        #make a predict for the query for its cluster
        queryClusterWiki = self.__clf.predict(embeddedQueryWiki)[0]
            
        #search in assigned container
        cluster = queryClusterWiki
        
        #transform in a bow corpus the query
        vec_bow = self.__dictionary[cluster].doc2bow(pData.singularizeQuery(query))
        # convert the query to LSI space
        vec_lsi = self.__lsi[cluster][vec_bow]
        
        # perform a similarity query against the corpus
        sims = self.__indexList[cluster][vec_lsi]  
        
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        #print top 10 results
        for i, s in enumerate(sims[:10]):
            print(s, self.__documentsWithLabels[cluster][s[0]][1])

