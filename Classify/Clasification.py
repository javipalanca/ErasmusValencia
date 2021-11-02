import time
import numpy as np

from Matrix_Generated import X,Z,Q,T,K
from Process_Articles import wikiTrainTest,wikiEvalTest,wikiFinalTest
from Loading_Transcripts import transcripts
from Loading_Transcripts import dictTranscriptEmbed,dictTranscriptTitle

from sklearn import svm
from sklearn.model_selection import cross_val_score 
from collections import defaultdict 


dictTitleCluster = {} 
wikiTrainTest = np.array(wikiTrainTest)

validTranscriptsIndices = defaultdict(list)
wrongTranscriptsIndices = range(0,len(transcripts))

TranscriptsX = []
TranscriptsXLabels =[]


clf = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='rbf', C=60)
start = time.time()
while True:
    nr = 0
    #intre processed articles si labels
    scores = cross_val_score(clf, X,wikiTrainTest.ravel(), cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    
    #fit intre proccessed articles si labels
    clf.fit(X, wikiTrainTest)
    
    #prezice si pt articles de test 
    result = clf.predict(Z)
    
    from sklearn.metrics import classification_report
    
    #cat de bine prezice label pt articles de test
    print(classification_report( (np.asarray(wikiEvalTest)).ravel() , result))
    
    resultTrLabels = clf.predict(T)
    resultKeyLabels = clf.predict(K)
    
    validTranscriptsX = []
    validTranscriptsLabels = []
    
    invalidTranscriptsX = []
    invalidTranscriptsLabels = []
    invalidKeywordsX =[]
    
    wrongIndices =[]
    for i in range(0,len(resultTrLabels)):
        if(resultTrLabels[i] == resultKeyLabels[i]):
            validTranscriptsX.append(T[i])
            for key_transcript in dictTranscriptEmbed:
                ok = 1
                for k in range(0, len(dictTranscriptEmbed[key_transcript])):
                   if(dictTranscriptEmbed[key_transcript][k] != T[i][k]):
                       ok = 0
                       break
               
                if(ok == 1):
                    dictTitleCluster[dictTranscriptTitle[key_transcript]] = resultTrLabels[i]
                    
            TranscriptsX.append(T[i])
            TranscriptsXLabels.append(resultTrLabels[i])    
            validTranscriptsLabels.append(resultTrLabels[i])
            validTranscriptsIndices[resultTrLabels[i]].append(wrongTranscriptsIndices[i])
            nr+=1
        else:
            invalidTranscriptsX.append(T[i])
            invalidTranscriptsLabels.append(resultTrLabels[i])
            invalidKeywordsX.append(K[i])
            wrongIndices.append(wrongTranscriptsIndices[i])
    
    wrongTranscriptsIndices = wrongIndices
        

    print("Valid transcripts:" + str(nr) + " / " + str(len(resultTrLabels)) )
    
    if(nr < 20):
        for j in range(0,len(invalidTranscriptsX)):
              TranscriptsX.append(invalidTranscriptsX[j])
              TranscriptsXLabels.append(invalidTranscriptsLabels[j])
        break
   
    X = np.append(X, np.array(validTranscriptsX),axis = 0)
    wikiTrainTest = np.append(wikiTrainTest, np.array(validTranscriptsLabels))
    
    T = np.array(invalidTranscriptsX)
    K = np.array(invalidKeywordsX)
    
    if(nr == len(resultTrLabels)):
        print(nr)
        print(resultTrLabels)
        break

clf = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='rbf', C=60)
scores = cross_val_score(clf, np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel(), cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel())
result = clf.predict(Q)

end = time.time()   
print(classification_report((np.asarray(wikiFinalTest)).ravel(), result))

print(start - end)