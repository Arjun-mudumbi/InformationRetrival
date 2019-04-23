#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:52:31 2019
TeamName: A team has no name
@author: VidhyaSri , Arjun Mudumbi Srinivasan, Rahul Pandey
Usage: python precision_recall.py cranqrel cran.output.txt
Information : This python file is used to determine the Precision and recall of the Information retrival system
This takes in the cranqrel file which is the gold standard and the result file for which the performance is to be tested 
"""

# Importing the required libraries  
import numpy as np
import pandas as pd
import re
import os
import sys
from collections import defaultdict
# opening the gold standard file
querydict=defaultdict(list)# Intializing a defaultdict of list 
with open(sys.argv[1]) as c:
    for line in c:
        queryid=re.findall(r'^([0-9]+) ',line)# from each line we are finding the query ID 
        docsid=re.findall(r' ([0-9]+) ',line)# from each line we are finding the dco ID
        if int(queryid[0]) not in querydict.keys():# checking if the query ID is present in the default dict keys
            querydict[int(queryid[0])]=[int(docsid[0])]# if the default dict doesnt have the key then create a key value pair
        else:
            querydict[int(queryid[0])].append(int(docsid[0]))#if the defaultdict has the key then append the corresponding doc Id to the value of the dict
#opening the result file 
resultdict=defaultdict(list)#Intitializing a defaultdict of list for the results
with open(sys.argv[2]) as c:
    for line in c:
        queryid=re.findall(r'^([0-9]+) ',line)# finding the query id in the results
        docsid=re.findall(r' ([0-9]+) ',line)# finding the doc id in the result
        if int(queryid[0]) not in resultdict.keys():
            resultdict[int(queryid[0])]=[int(docsid[0])]
        else:
            resultdict[int(queryid[0])].append(int(docsid[0]))
#%%%%%%%%%%%%%55
precision=[]# Initializing the list of Precisions
recall=[]# Initializing the list of Recalls
for query in resultdict.keys():# Since each query must have a relevant 
    resultlist=resultdict[query]
    querylist=querydict[query]
    relevant=[]# creating a list of relevant docs
    for result in resultlist:#for every result in the result list we compare against the query list
        if result in querylist:# if the result is present in the querylist
            relevant.append(result)# We append to the relevant list
        precisionscore=len(relevant)/len(resultlist)# precision score = total relevant retrieved docs/total docs retrieved
        recallscore=len(relevant)/len(querylist)# recall score = total relevant retrieved docs/ total relevant docs
    precision.append(precisionscore)# appending the scores to the precision list
    recall.append(recallscore)# appending the scores to recall list
#%%%%%%%%%55
precision=np.array(precision)
recall=np.array(recall)
print("Precision=%.4f" %precision.mean())# calculating the mean precision for all the queries 
print("Recall=%.4f"%recall.mean())# calculating the mean recall for all the queries

        