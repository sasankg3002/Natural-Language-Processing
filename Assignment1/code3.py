from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import time

from nltk.corpus import brown

np.set_printoptions(threshold = np.inf)

emissionMat = []
transitionMat = []
startWords = []

alpha = 0.4

def viterbi(sentences):
    #pt = 0
    global emissionMat, transitionMat, startWords, alpha, uniqueWords, uniqueTags, mapWords, mapTags
    wordCount = len(uniqueWords)
    tagCount = len(uniqueTags)
    tags = []
    tagList = []
    a = time.time()
    for sentence in sentences:
        #print(pt)
        #pt += 1
        #print("***** ", time.time()-a)
        for i in range(0, len(sentence)):
            #a = time.time()
            tag = None
            max = -100
            maxtag = None  
            curTag = None    
            for tag in uniqueTags:
                num = alpha
                wordin = mapWords.get(sentence[i])
                tagin = mapTags[tag]
                emissionProb = 1/wordCount
                if (wordin):
                    emissionProb = emissionMat[tagin][wordin]
                    #num += emissionMat[tagin][wordin]
                # den = np.sum(emissionMat[mapTags[tag]]) + alpha*wordCount
                # emissionProb = num/den
                
                transProb = 0
                # num = alpha
                # den = alpha*tagCount
                num = 0
                den = 0
                if (i >= 1):
                    # num += transitionMat[mapTags[tags[i-1]]][tagin]
                    # den += np.sum(transitionMat[tagin])
                    # transProb = num/den
                    transProb = transitionMat[mapTags[tags[i-1]]][tagin]
                else:
                    num += startWords[tagin]
                    den += np.sum(startWords)
                    transProb = num/den
                if (np.log(emissionProb) + np.log(transProb) > max):
                    max = np.log(emissionProb) + np.log(transProb)
                    curTag = tag
            #print(time.time()-a)
        
            tags.append(curTag)  
        tagList.append(tags)
    return tagList 
    
def checkMatch(trueTags, predictedTags):
    n = len(trueTags)
    count = 0
    for i in range(n):
        if (trueTags[i] == predictedTags[i]):
            count += 1
    return count, n

st = time.time()

x = brown.tagged_sents(tagset = 'universal')
sentenceCount = len(x)         #Total number of sentences

tags = [ j[1] for i in x for j in i ]
words = [ j[0] for i in x for j in i ]

lens = np.array([ len(x) for x in words ])

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

        columnSum = np.sum(perFoldTagsToTags, axis = 0)
        
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

    probTagsToWords = (tagsToWords + alpha)/(tagsToWords.sum(axis = 1, keepdims = True) + alpha*wordCount)
    probTagsToTags = (tagsToTags + alpha)/(tagsToTags.sum(axis = 1, keepdims = True) + alpha*tagCount)

    emissionMat = probTagsToWords
    transitionMat = probTagsToTags
    startWords = perFoldStartWords[i]

    matches = 0
    total = 0
    print("------------------------------------------------------------------------------------------------------------------------")
    print(time.time() - st)

    sentences = []
    for i in x[foldIndices[i][0] : foldIndices[i][1]]:
        sentence = [i[0] for j in i]
        sentences.append(sentence)

    viterbi(sentences)

et = time.time()
print(et - st)