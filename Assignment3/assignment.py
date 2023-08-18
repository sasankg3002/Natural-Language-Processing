import nltk
from nltk import pos_tag

from nltk.corpus import semcor
from nltk.corpus import wordnet as wn

import numpy as np

def similarity(synset1, synset2):
    meaning1 = synset1.definition()
    meaning2 = synset2.definition()
    set1 = meaning1.lower()
    set2 = meaning2.lower()
    s1List = set1.split(" ")
    s2List = set2.split(" ")
    return len(list(set(s1List)&set(s2List)))

# tagged_semcor_sents = semcor.tagged_sents(tag = 'sem')
# semcor_sents = semcor.sents()   

# #Number of sentences in semcor dataset
# L = 37176

# #Take sentence as input from user
# sentence = "Sainath is an Indian"

# sentence_words = sentence.split()
# num_synsets = []
# synset_to_index_maps = []
# all_synsets = []

# N = 0

# for i in sentence_words:
#     synsets = wn.synsets(i)
#     for synset in synsets:
#         all_synsets.append(synset)
#     wordmap = {}
#     for i_dx, synset in enumerate(synsets):
#         wordmap[synset] = i_dx
#     synset_to_index_maps.append(wordmap)

#     num_synsets.append(len(synsets))
#     N += len(synsets)

# print(all_synsets)

# print(num_synsets)
# print(synset_to_index_maps)

# P = np.zeros((N, N))
# print(P.shape)

# count = 0
# for i in range(N):
#     for j in range(N):
#         if (j != i):
#             P[i][j] = similarity(all_synsets[i], all_synsets[j])

# for i in range(N):
#     for j in range(N):
#         if (np.sum(P[j]) > 0):
#             P[i][j] = P[i][j]/np.sum(P[j])

# num_iter = 100

# original_prob = np.ones((N, 1))/N
# final_prob = original_prob
# for i in range(num_iter):
#     final_prob = np.matmul(P, final_prob)
#     final_prob = final_prob/np.linalg.norm(final_prob)

# final_prob = np.reshape(final_prob, (N, ))

# count = 0
# for i in range(len(sentence_words)):
#     if (num_synsets[i] == 0):
#         print(sentence_words[i], "No other synsets")
#     else :
#         idx = np.argmax(final_prob[count : count + num_synsets[i]])
#         sense_synset = all_synsets[count + idx]
#         print(sentence_words[i], sense_synset, sense_synset.definition())
#         count += num_synsets[i]

sem_data = semcor.tagged_sents(tag = "sem")
pos_data = semcor.tagged_sents(tag = "pos")

finalWords = []
finalTags = []
finalSenses = []

totalSentences = 37176

for i in range(totalSentences):
    if (i%500 == 0):
        print(i)
    sent = sem_data[i]
    wordListInSentence = []
    posTagsInSentence = []
    sensesInSentence = []
    for j in range(len(sent)):
        if (isinstance(sent[j], nltk.tree.Tree) and isinstance(sent[j].label(), nltk.corpus.reader.wordnet.Lemma)):
            if (len(pos_data[i][j]) == 1):
                word = pos_data[i][j][0]
                wordListInSentence.append(word)
                posTagsInSentence.append(pos_data[i][j].label())
                sensesInSentence.append(sent[j].label().synset())
        
        finalWords.append(wordListInSentence)
        finalTags.append(posTagsInSentence)

print(finalWords)
print(finalTags)
print(sensesInSentence)
