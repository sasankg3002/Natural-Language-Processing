from pickle import NONE
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sn

import nltk
from nltk.corpus import brown

emissionMat = []
transitionMat = []
startWords = []

alpha = 0.4

nltk.download('brown')
nltk.download('universal_tagset')

def viterbi(sentences):
    global emissionMat, transitionMat, startWords, alpha, uniqueWords, uniqueTags, mapWords, mapTags
    wordCount = len(uniqueWords)
    tagCount = len(uniqueTags)
    tagList = []
    a = time.time()
    probTagsToWords = (tagsToWords + alpha)/(tagsToWords.sum(axis=1, keepdims=True) + alpha*wordCount)
    probTagsToTags = (tagsToTags + alpha)/(tagsToTags.sum(axis=1, keepdims=True) + alpha*tagCount)

    tagCounts = tagsToTags.sum(axis = 1)

    for sentence in sentences:
        L = len(sentence)
        probTagSeq = np.zeros((L, tagCount))
        pred = np.zeros((L, tagCount))
        sentenceTags = np.zeros(L, dtype = np.int64)

        for i in range(0, L):
            wordin = mapWords.get(sentence[i])
            if (i == 0):
                for tag in uniqueTags:
                    tagin = mapTags[tag]
                    emissionProb = 0
                    transitionProb = 0
                    if (wordin):
                        emissionProb = probTagsToWords[tagin][wordin]
                    else:
                        emissionProb = alpha/(tagCounts[tagin] + alpha*wordCount)
                    transitionProb = startWords[tagin]/np.sum(startWords)
                    probTagSeq[i][tagin] = np.log(emissionProb) + np.log(transitionProb)
            else:
                for tag in uniqueTags:
                    tagin = mapTags[tag]
                    maxTag = 0
                    maxProb = -np.inf
                    for prevTag in uniqueTags:
                        tagind = mapTags[prevTag]
                        emissionProb = 0
                        transitionProb = 0
                        if (wordin):
                            emissionProb = probTagsToWords[tagin][wordin]
                        else:
                            emissionProb = alpha/(tagCounts[tagin] + alpha*wordCount)
                        transitionProb = probTagsToTags[tagind][tagin]
                        if (np.log(emissionProb) + np.log(transitionProb) + probTagSeq[i-1][tagind] > maxProb):
                            maxProb = np.log(emissionProb) + np.log(transitionProb) + probTagSeq[i-1][tagind]
                            maxTag = tagind
                    pred[i][tagin] = maxTag
                    probTagSeq[i][tagin] = maxProb

        lastTag = 0
        maxProbability = -np.inf

        for tag in uniqueTags:
            tagin = mapTags[tag]
            if (probTagSeq[L-1][tagin] > maxProbability):
                maxProbability = probTagSeq[L-1][tagin]
                lastTag = tagin
                lastTag = int(lastTag)
        sentenceTags[L-1] = lastTag
        for j in range(L-1):
            lastTag = int(lastTag)
            lastTag = pred[L-1-j][lastTag]
            sentenceTags[L-2-j] = lastTag
        sentenceTags = list(sentenceTags)
        tagList.append(sentenceTags)
    return tagList

def FScore(precision, recall, beta):
    fscore = ((1+beta*beta)*precision*recall)/(beta*beta*precision + recall)
    return fscore

st = time.time()

x = brown.tagged_sents(tagset='universal')
sentenceCount = len(x)  # Total number of sentences

tags = [j[1] for i in x for j in i]
words = [j[0] for i in x for j in i]

lens = np.array([len(x) for x in words])

N = len(tags)
print("Total number of words :", len(tags))

uniqueWords = set(words)
uniqueWords = list(uniqueWords)
print("Number of unique words :", len(uniqueWords))

uniqueTags = set(tags)
uniqueTags = list(uniqueTags)
print("Unique tags :", uniqueTags)

wordCount = len(uniqueWords)  # Number of unique words
tagCount = len(uniqueTags)  # Number of unique tags

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

perFoldTagsToWords = np.zeros((5, tagCount, wordCount), dtype=np.int64)
perFoldTagsToTags = np.zeros((5, tagCount, tagCount), dtype=np.int64)
perFoldStartWords = np.zeros((5, tagCount), dtype=np.int64)

for i in range(5):
    for j in range(foldIndices[i][0], foldIndices[i][1]):
        curSentence = x[j]
        perFoldStartWords[i][mapTags[curSentence[0][1]]] += 1
        for k in x[j]:
            indexOfWord = mapWords[k[0]]
            indexOfTag = mapTags[k[1]]
            perFoldTagsToWords[i][indexOfTag][indexOfWord] += 1

        columnSum = np.sum(perFoldTagsToTags, axis=0)

        for l in range(1, len(x[j])):
            curTag = curSentence[l][1]
            prevTag = curSentence[l-1][1]

            perFoldTagsToTags[i][mapTags[prevTag]][mapTags[curTag]] += 1

for i in range(5):
    tagsToWords = np.zeros((tagCount, wordCount), dtype=np.int64)
    tagsToTags = np.zeros((tagCount, tagCount), dtype=np.int64)
    startWords = np.zeros(tagCount, dtype = np.int64)
    for j in range(5):
        if (j != i):
            tagsToWords += perFoldTagsToWords[j]
            tagsToTags += perFoldTagsToTags[j]
            startWords += perFoldStartWords[j]

    probTagsToWords = (tagsToWords + alpha)/(tagsToWords.sum(axis=1, keepdims=True) + alpha*wordCount)
    probTagsToTags = (tagsToTags + alpha)/(tagsToTags.sum(axis=1, keepdims=True) + alpha*tagCount)

    emissionMat = tagsToWords
    transitionMat = tagsToTags

    matches = 0
    total = 0

    sentences = []
    trueTagList = []
    for j in x[foldIndices[i][0]: foldIndices[i][1]]:
        sentence = [k[0] for k in j]
        trueTags = [k[1] for k in j]
        sentences.append(sentence)
        trueTagList.append(trueTags)

    predictedTagList = viterbi(sentences)

    confusionMatrix = np.zeros((tagCount, tagCount), dtype=np.int64)

    for j in range(len(trueTagList)):
        for k in range(len(trueTagList[j])):
            trueTagIn = mapTags[trueTagList[j][k]]
            predictedTagIn = predictedTagList[j][k]

            confusionMatrix[trueTagIn][predictedTagIn] += 1

    TP = np.zeros((tagCount, 1))
    FP = np.zeros((tagCount, 1))
    FN = np.zeros((tagCount, 1))
    for j in range(tagCount):
        TP[j] = confusionMatrix[j][j]
        FN[j] = np.sum(confusionMatrix[j]) - TP[j]
        FP[j] = np.sum(confusionMatrix[:, j]) - TP[j]
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)


    print("Precision :")
    print(precision)
    print("Recall :")
    print(recall)
    print("F1-score :")
    print(FScore(precision, recall, 1.0))
    print("F0.5-score :")
    print(FScore(precision, recall, 0.5))
    print("F2-score :")
    print(FScore(precision, recall, 2))

    print(confusionMatrix)
    df_cm = pd.DataFrame(confusionMatrix, index = uniqueTags, columns = uniqueTags)
    plt.figure(figsize = (15,10))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})
    plt.show()


et = time.time()

print("Total time taken for execution :", et - st, "seconds")