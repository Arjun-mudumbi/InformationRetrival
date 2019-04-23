"""
Course: AIT 690
Assignment: Programming Assignment 4 - Information Retrieval
Date: 17/04/2019
Team Name: ' A team has no name'
Members:  1. Rahul Pandey
          2. Arjun Mudumbi Srinivasan
          3. Vidhyasri Ganapathi
          
ir-system.py program implements a system that returns ranked documents relevant to the given
query.

The program learns all queries from cran.qry and the documents from cran.all.1400.
Then they are converted into vector representation by calculating TF-IDF score for document 
and queries. Next, cosine similarity is calculated of all query vectors for each document vectors
and stored in cran.output.txt. Then precision and recall is calculated for the system by comparing
against gold-standard list.


Files used:

1. cran.qry contains queries. Each query is assigned an ID number marked with a 
    line that starts with .I and a number. .W indicates the body of text of the query. 
2. cran.all.1400 contains 1400 documents in one file.  These documents are each assigned an ID 
    number marked with a line that starts with .I and a number. In the next line .T indicates 
    that the title of the article follows. .A marks the author name. .B gives publication 
    information. .W indicates the body of text of the article. 
3. cranqrel contains three columns of information, where the first column corresponds to 
    query IDs and the second corresponds to document IDs.
4. precision_recall.py computes the precision and recall value of the system by comparing the
    reults from cran.output.txt file with gold-standard list in cranqrel file.



Our performance = 
    Precision=0.3571
    Recall=0.4373


Baseline performance:
    Precision=17.1
    Recall=18.4


Improvement over the bag of words feature:
1) By removing the non-important features such as Punctutions and stop words
2) By using unigrams and bigrams over the traditional unigrams when generating the Bag of words Models
3) By Varying the threshold for returning the related documents .We have developed a system which
    will use the minimum cosine score and maximum cosine score for each query and return only the documents which
    are atleast more min score +0.5(max score-min score). By this threshold we are able to generate good precision 
    and recall score
4) By changing new line ("\n") to space (" ")

Run the ir-system file:
$ python ir-system.py cran.all.1400 cran.qry> cran.output.txt
Run the precision-recall.py file:
$ python precision-recall.py cranqrel cran.output.txt> mylogfile.txt

"""

#%%%%%%%%%%%55
import numpy as np
import pandas as pd
import re
import string
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

#Method to calculate the cosine similarity
def get_cosine_similarity(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(-1,1)
    dot = np.dot(a, b)#returns dot product of the values
    norma = np.linalg.norm(a)#computing the norm of the document vector array
    normb = np.linalg.norm(b) #computing the norm of the query vector array
    cos = dot / (norma * normb)
    return cos


remove_punct_regex = re.compile('[%s]' % re.escape(string.punctuation))
all_stopwords = set(stopwords.words("english"))


def clean_str(string):
    string = string.replace("\n", " ")# replace new line with space
    string = remove_punct_regex.sub(' ', string)# replace punctuations
    string = re.sub(r"\s{2,}", " ", string)#replace multiple spaces with a single space
    string = re.sub(' +', ' ', string)#replace + with space
    string = string.strip()
    string = " ".join([wrd for wrd in word_tokenize(string) if wrd not in all_stopwords])
    return string


#%%%%%%%%%%%%%5555
doc = ""
title = ""
with open('cran.all.1400') as c:
    for j in c:
        doc =  doc + j
    docs = re.findall(r'.W\n([^I]*)',doc)#extracting all the title and the body of text of the article
    docs = [clean_str(doc) for doc in docs]# applying the cleaning function for the docs
    title = re.findall(r'.T\n([^A]*)',doc)
    title = [clean_str(tl) for tl in title]
#%%%%%%%5
query = ""
with open('cran.qry') as c:
    for j in c:
        query =  query + j
    queries = re.findall(r'.W\n([^I]*)',query)#extracting the body of text of the query
    queries = [clean_str(query) for query in queries]#applying the cleaning function for the queries

# for j in range(len(docs)):
#     docs[j] = docs[j].replace("\n", " ")
# #    docs[i] = doc[i].replace("/","")
# #    docs[i] = docs[i].replace(")","")
#     docs[j] = docs[j].replace("-", " ")
#     docs[j] =  docs[j][:-1]
# for j in range(len(queries)):
#     queries[j] = queries[j][:-1]
#%%%%%%%%%55
#adding each element to the list which is split by whitespace
docwordlist = []
for doc in docs:
    docwordlist.extend(word_tokenize(doc))
for qr in queries:
    docwordlist.extend(word_tokenize(qr))
docwordlist = list(set(docwordlist))
#%%
#Calculate the TFIDF 
vectorizer = TfidfVectorizer(vocabulary = docwordlist, ngram_range=(1,2))
doc_query = [x for x in docs]
doc_query.extend(queries)
vectorizer.fit(doc_query)#fitting the vectorizer and converting document vector to an array
representation_document = vectorizer.transform(docs)
representation_document = pd.DataFrame(representation_document.toarray(),columns = vectorizer.get_feature_names())
#%%%%%%%%%%%%%55
query_wordlist = []
for query in queries:
    query_wordlist.append(query.split(" "))#adding each element to the list which is split by whitespace
#%%%%%%%%%%%%55
queries_tfidf = []
for query in queries:
    fit = vectorizer.transform([query])#fitting the vectorizer and converting query vector to an array
    queries_tfidf.append(fit.toarray()[0])
#%%%%%%%%%%55
#calling the cosine_similarity function to calculate cosine of all query vectors 
#for each document vector
result = defaultdict(list)
for i in range(len(queries_tfidf)):
    temp = []
    for j in range(len(representation_document)):
        p = representation_document.iloc[j].values#retrieve each doc vector
        q = queries_tfidf[i]#retrieve each query vector
        p = p.reshape(1,-1)# reshape the vectors
        q = q.reshape(1,-1)
        cosine_score = get_cosine_similarity(p, q)#get the cosine score
        if cosine_score > 0.0:#neglecting the 0 cosine score vectors
            temp.append((j+1, cosine_score))
    min_cosine_score = np.array(temp)[:, 1].min()#take the min cosine score for the query
    max_cosine_score = np.array(temp)[:, 1].max()#take the max cosine score for the query
    th = min_cosine_score + 0.5*(max_cosine_score-min_cosine_score)# Calculate the threshold value
    # print("Min: %.6f | Max: %.6f | Th: %.6f" % (min_cosine_score, max_cosine_score, th))
    # cnt = 0
    for dc_id, score in temp:
        if score > th:
            result[i+1].append((dc_id, score))# append only scores getter than the threshold value
            # cnt += 1
    # print("%d: %d: %d" % (i+1, len(temp), cnt))
for qr_id in result.keys():
    result[qr_id] = sorted(result[qr_id], key = lambda x: x[1], reverse = True)# sort the values

for qr_id in result.keys():
    for dc_id, score in result[qr_id]:
        print("%d %d %d" % (qr_id, dc_id, int(score*100)))# print the scores