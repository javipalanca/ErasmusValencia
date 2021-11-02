import os
import pandas as pd
from nltk import RegexpTokenizer

def processTranscript(transcript):
    regex = RegexpTokenizer(r"\b\w+\b");
    words = regex.tokenize(transcript)
    processedTranscript = ""
    for word in words:
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript


path=os.path.dirname(os.path.abspath(__file__))
pathFile=path+"\\data sets\\wikipediaArticles2.0.xlsx"

dataframe = pd.read_excel(pathFile, usecols=[1])
articles = dataframe.values
dataframe = pd.read_excel(pathFile, usecols=[2])
labels = dataframe.values

processedArticles = []
processedLabels = []

for i in range(0,len(articles)):
    art = str(articles[i][0])
    if len(art) < 50:
        continue
        
    processedArticles.append(processTranscript(art))
    processedLabels.append(labels[i])

from sklearn.model_selection import train_test_split    

wikiTrain, wikiEvalTrain, wikiTrainTest, wikiEvalTest = train_test_split(processedArticles,processedLabels , test_size=0.30, random_state=42)
wikiEvalTrain, wikiFinalTrain, wikiEvalTest, wikiFinalTest =  train_test_split(wikiEvalTrain,wikiEvalTest , test_size=0.50, random_state=42)
