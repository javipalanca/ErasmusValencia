from Process_Articles import path,processTranscript

import json
pathFile= path + "\\videos_upv.json"
with open(pathFile,"r", encoding='utf-8') as f:
    videos_json = json.load(f)

i = 0
counter = 0

transcripts = []
transcriptsKeywords = []

dictIndexTranscript = {}
dictTranscriptTitle = {}
dictTranscriptEmbed = {}

for videos in videos_json:
    transcript =  videos["transcription"]
    
    if("metadata" not in videos):
        continue
    if("keywords" not in videos["metadata"]):
        continue
    
    keywords_obj = videos["metadata"]["keywords"]
    keywords = ""

    if(type( keywords_obj) is list):
        for text in  keywords_obj:
            keywords+= text + " "
    else:
        keywords = keywords_obj
        
    if transcript != "" and keywords!= "":
        processedTr = processTranscript(transcript)
        processedKey = processTranscript(keywords)
        
       
        if(processedTr !="" and processedKey !=""):
            transcripts.append(processedTr)
            dictIndexTranscript[counter] = processedTr
            dictTranscriptTitle[processedTr] = videos["title"]
            
            processedKey += videos["title"] + " "
            transcriptsKeywords.append(processedKey)
            counter = counter + 1
    i = i + 1