# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:58:22 2019

@author: Diana
"""

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
import tensorflow_hub as hub


import gensim
import pandas as pd
import re
import pickle
import time

import wiki
import preprocessData as pData

print("Don't forget to start StanfordCoreNLP!")

#extract data bases of videos
def extractData():
    for i in range(3):
        dataframe = pd.read_excel("LabeledTranscripts.xlsx", engine = "openpyxl", usecols = [i+1])
        yield dataframe.values

videosName,transcripts,labels = extractData()

print(videosName)


documents = []
documentsWithLabels = [[],[],[]]
preprocessedDocumentsList = []
videoIdDictionary = {}


#Process all transcripts
nrOfTranscriptsToProcess = len(videosName)

print("initialized values!")

for i in range(0,nrOfTranscriptsToProcess):
    transcript = transcripts[i][0]
    videoName = videosName[i][0]
    label = labels[i][0]
    if transcript !=  "" and len(transcript)>6000:      #keep only valid transcripts for preprocessing
        preprocessedTranscript = pData.preprocess(transcript)   

        #keep all preprocessed transcripts in a container for each label
        documentsWithLabels[label].append((preprocessedTranscript,videoName))
        
        #keep all preprocessed transcripts together
        documents.append([preprocessedTranscript, videoName, label])

        #keep all words from transcripts
        preprocessedDocumentsList.append(preprocessedTranscript.split())

   
    
noOfClusters = 3
preprocessedListForDictionary = []


for i in range(noOfClusters):
    preprocessedListForDictionary.append([])


print("appended preprocessed list for dict")

#append transcripts words in a list
index = 0
for transcript, videoName, label in documents:
    preprocessedListForDictionary[label].append(preprocessedDocumentsList[index])
    index += 1
print("appende transcripts words in a list")


# Creating the dictionaries for the LSI models

dictionary = []
for i in range(noOfClusters):
    dictionary.append(gensim.corpora.Dictionary(preprocessedListForDictionary[i]))


bow_corpus = [None] * noOfClusters  
lsi = [None] * noOfClusters
indexList = [None] * noOfClusters


from gensim import models
from gensim import similarities

print("Created dictionary for the LSI models")

# Creating the LSI models

for i in range(noOfClusters):
    if( len(dictionary[i]) != 0):
        #we have to create a bag of words( BoW )
        bow_corpus[i] = [dictionary[i].doc2bow(doc) for doc in preprocessedListForDictionary[i]]
        
        #we will transform it in a tf-idf vector
        tfidf = models.TfidfModel(bow_corpus[i]) 
        corpus_tfidf = tfidf[bow_corpus[i]]

        lsi[i] = models.LsiModel(corpus = corpus_tfidf, id2word=dictionary[i], num_topics=5)
        #we will compute a similarity matrix, which it will help us later, for query
        indexList[i] = similarities.MatrixSimilarity(lsi[i][corpus_tfidf])
        
        #print(indexList[0])
        print(lsi[i].print_topics(num_topics= 5 , num_words=10))
        
print("Created the LSI models")
#load the model
embed = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")
print("embed loaded")
clf = pickle.load(open("pretrained_model.sav", 'rb'))

print("loaded model", flush=True)
#Function that returns the result for a query

def resultForQuery(query):
    ans = wiki.wikipedia_search(query)     #search a sequence on wiki pages
    if(len(ans["itemList"]) != 0 ):             #if we have a result
            queryWiki =  (ans["itemList"][0]["description"])       #assign as query this sequence
    else:
            queryWiki = query                  #else assign just the query
    
    
    
    with tf.compat.v1.Session() as session:
            tf.compat.v1.disable_eager_execution()
            session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
            embeddedQueryWiki = session.run(embed([queryWiki]))

    #make a predict for the query for its cluster
    queryClusterWiki = clf.predict(embeddedQueryWiki)[0]
        
    #search in assigned container
    cluster = queryClusterWiki
    
    print("cluster")
    print(cluster)
    
    #transform in a bow corpus the query
    vec_bow = dictionary[cluster].doc2bow(pData.singularizeQuery(query))
    # convert the query to LSI space
    vec_lsi = lsi[cluster][vec_bow]  
    
    # perform a similarity query against the corpus
    sims = indexList[cluster][vec_lsi]  
    
    
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    
    #print top 10 results
    for i, s in enumerate(sims[:10]):
        print(s, documentsWithLabels[cluster][s[0]][1])



while(True):
    query = input()
    millis = int(round(time.time() * 1000))
    resultForQuery(query)
    print('python query part took ', int(round(time.time() * 1000))-millis, ' seconds')
    print('eoq',flush=True)





