import re
from nltk.corpus import wordnet as wn
import nltk

sense_key_regex = r"(.*)\%(.*):(.*):(.*):(.*):(.*)"
synset_types = {1:'n', 2:'v', 3:'a', 4:'r', 5:'s'}

def synset_from_sense_key(sense_key):
    lemma, ss_type, lex_num, lex_id, head_word, head_id = re.match(sense_key_regex, sense_key).groups()
    ss_idx = '.'.join([lemma, synset_types[int(ss_type)], lex_id])
    return wn.synset(ss_idx)

x = "pass%2:38:03::"

sense = synset_from_sense_key(x)
synsets = wn.synsets("pass", pos='v')
print("Total number of possible synsets :", len(synsets))
for i in synsets:
    print(i, i.definition(), i.examples())
print(sense, sense.definition(), sense.examples())