import pickle
import pandas as pd
from Loading_Transcripts import dictTranscriptTitle
from Clasification import dictTitleCluster,clf

titles = []
file = open("rakeKeywordsceva.txt", "w", encoding="utf-8")
for title_key in dictTitleCluster:
    titles.append(title_key)
titles.sort()  
for title_key in titles:
    file.write(str(title_key) + " " + str(dictTitleCluster[title_key])+'\n')

file.close()

ldaDataFrame = pd.DataFrame(columns = ["VideoTitle", "ProcessedTranscript", "Cluster"])
for transcriptGabi_key in dictTranscriptTitle:
        if(dictTranscriptTitle[transcriptGabi_key] in dictTitleCluster):
            newLine = pd.DataFrame(
                                    [
                                        [
                                            dictTranscriptTitle[transcriptGabi_key], 
                                            transcriptGabi_key,
                                            dictTitleCluster[dictTranscriptTitle[transcriptGabi_key]]
                                        ]
                                    ],
                                    columns = ['VideoTitle','ProcessedTranscript', 'Cluster']
                                )
            ldaDataFrame = ldaDataFrame.append(newLine,ignore_index = True)

writer = pd.ExcelWriter('LdaData.xlsx', engine='xlsxwriter')

#Convert the dataframe to an XlsxWriter Excel object.
ldaDataFrame.to_excel(writer, sheet_name='Sheet1')

#Close the Pandas Excel writer and output the Excel file.
writer.save()

filename = 'alex_model.sav'
pickle.dump(clf, open(filename, 'wb'))
