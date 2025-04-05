from conllu import parse
import numpy as np
from collections import Counter

#FUNZIONA
#crea la matrice con tutte le parole e il corrispondente tag
def create_mat(data):
    sentences = parse(data)
    mat = [[token["form"], token["upostag"]]    #aggiungi a mat la coppia
           for s in sentences                   #per ogni frase
           for token in s                       #per ogni token nella frase
           if token["upostag"] != '_']          #se il token è diverso da _
    return np.array(mat)


#FUNZIONA
#conta e calcola le probabilità di comparsa di parole/tag
def calc_prob(data):
    data_counts = Counter(data)
    np_data_counts = np.array([[word, count] for word, count in data_counts.items()])
    counts = np_data_counts[:, 1].astype(int)
    freq = (counts / np.sum(counts)).reshape(-1, 1)
    return np.hstack((np_data_counts, freq))

#FUNZIONA
#conta quante volte compare un tag e ne calcola la probabilità a priori
def count_tags(data):
    tags = [x[1] for x in data]
    return calc_prob(tags)

#FUNZIONA
#conta quante volte compare una parola e ne calcola la probabilità a priori
def count_words(data):
    words = [x[0] for x in data]
    return calc_prob(words)


#SBAGLIATA
#conta quante volte una parola è associata ad un tag
#calcola la probabilità che una parola
def count_word_tag_number(data):
    pairs = [tuple(x) for x in data]  # (word, tag)
    pairs_counts = Counter(pairs)
    np_pairs_counts = np.array([[word, tag, count] for (word, tag), count in pairs_counts.items()])
    counts = np_pairs_counts[:, 2].astype(int)
    freq = (counts / np.sum(counts)).reshape(-1, 1)
    return np.hstack((np_pairs_counts, freq))








#FUNZIONA
#leggi i file
#with open("UD_Italian-VIT-master/it_vit-ud-train.conllu", "r", encoding="utf-8") as f:
#    data_vit_train = f.read()
## vit_train = create_mat(data_vit_train)
#
#with open("UD_Italian-VIT-master/it_vit-ud-dev.conllu", "r", encoding="utf-8") as f:
#    data_vit_dev = f.read()
## vit_dev = create_mat(data_vit_dev)
#
#with open("UD_Italian-VIT-master/it_vit-ud-test.conllu", "r", encoding="utf-8") as f:
#    data_vit_test = f.read()
## vit_test = create_mat(data_vit_test)
#
#with open("UD_Italian-Old-master/it_old-ud-train.conllu", "r", encoding="utf-8") as f:
#    data_old_train = f.read()
## old_train = create_mat(data_old_train)
#
#with open("UD_Italian-Old-master/it_old-ud-dev.conllu", "r", encoding="utf-8") as f:
#    data_old_dev = f.read()
## old_dev = create_mat(data_old_dev)

with open("UD_Italian-Old-master/it_old-ud-test.conllu", "r", encoding="utf-8") as f:
    data_old_test = f.read()
old_test = create_mat(data_old_test)

#--------------------------------------------------------------------------------------------------------------------------------------



test = count_word_tag_number(old_test)

[print(x) for x in test]










