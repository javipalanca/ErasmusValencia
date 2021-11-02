# -*- coding: utf-8 -*-

from SearchForQuery import SearchForQuery

searchForQuery = SearchForQuery(3)

while(True):
    print("Ready for input")
    query = input()
    if(query == "STOP"):
        nlp.close()
        break
    searchForQuery.resultForQuery(query)