import numpy as np
import matplotlib.pyplot as plt
import time

from nltk.corpus import brown

np.set_printoptions(threshold = np.inf)

st = time.time()

x = brown.tagged_sents(tagset = 'universal')
sentenceCount = len(x)         #Total number of sentences

tags = [ j[1] for i in x for j in i ]
words = [ j[0] for i in x for j in i ]

N = len(tags)
print("Total number of words :", len(tags))

uniqueWords = set(words)
uniqueWords = list(uniqueWords)
print("Number of unique words :", len(uniqueWords))

uniqueTags = set(tags)
uniqueTags = list(uniqueTags)

print("Unique tags :", uniqueTags)

wordCount = len(uniqueWords)    #Number of unique words
tagCount = len(uniqueTags)      #Number of unique tags

mapWords = {}
for i in range(len(uniqueWords)):
    mapWords[uniqueWords[i]] = i

mapTags = {}
for i in range(len(uniqueTags)):
    mapTags[uniqueTags[i]] = i

lenOfFold = int(np.floor(sentenceCount/5))
foldIndices = [[0, lenOfFold], 
               [lenOfFold, 2*lenOfFold], 
               [2*lenOfFold, 3*lenOfFold], 
               [3*lenOfFold, 4*lenOfFold], 
               [4*lenOfFold, sentenceCount]]

perFoldTagsToWords = np.zeros((5, tagCount, wordCount), dtype = np.int64)
perFoldTagsToTags = np.zeros((5, tagCount, tagCount), dtype = np.int64)
perFoldStartWords = np.zeros((5, tagCount))

for i in range(5):
    for j in range(foldIndices[i][0], foldIndices[i][1]):
        curSentence = x[j]
        perFoldStartWords[i][mapTags[curSentence[0][1]]] += 1
        for k in x[j]:
            indexOfWord = mapWords[k[0]]
            indexOfTag = mapTags[k[1]]
            perFoldTagsToWords[i][indexOfTag][indexOfWord] += 1
        
        for l in range(1, len(x[j])):
            curTag = curSentence[l][1]
            prevTag = curSentence[l-1][1]

            perFoldTagsToTags[i][mapTags[prevTag]][mapTags[curTag]] += 1

for i in range(5):
    tagsToWords = np.zeros((tagCount, wordCount), dtype = np.int64)
    tagsToTags = np.zeros((tagCount, tagCount), dtype = np.int64)
    for j in range(5):
        if (j != i):
            tagsToWords += perFoldTagsToWords[j]
            tagsToTags += perFoldTagsToTags[j]
    dum = np.sum(tagsToWords, axis = 0)
    print(dum)
    probTagsToWords = tagsToWords/tagsToWords.sum(axis = 1, keepdims = True)
    probTagsToTags = tagsToTags/tagsToTags.sum(axis = 1, keepdims = True)


et = time.time()
print(et - st)