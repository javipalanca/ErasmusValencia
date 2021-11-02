# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 12:22:41 2019

@author: Diana
"""
import inflector

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP("..\\stanford-server", lang = 'es')
#need to run a java server

#preprocess the query
def singularizeQuery(query):
    query = query.lower().split()
    return [singularizator.singularize(word) for word in query]

#get all stopwords from an input file and put their in a list
def buildStopWords():
    with open('stopwords.txt', 'r', encoding='latin-1') as fileInput:
        spanish_stopwords = [ line.strip() for line in fileInput ]


def ifIsFromStopWords(word):
    return word in spanish_stopwords


#keep from transcripts just the nouns in singular form
def preprocess(text):
    result = " "
    
    for word, pos in nlp.pos_tag(text):
        if pos == 'NOUN' :
            if( ifIsFromStopWords(word) == False ):
                result += singularizator.singularize(word) + " "
                
    
    return result

spanish_stopwords = []
buildStopWords()
singularizator = inflector.Spanish()