from conllu import parse
import numpy as np
from collections import Counter
import itertools



#FUNZIONA
#crea la matrice con tutte le parole e il corrispondente tag e crea il vettore con tutte le frasi
#[['Non' 'ADV']
# .....
# ['stelle' 'NOUN']]

#['ADV AUX VERB NOUN ADJ ADP DET NOUN ADP DET NOUN ADP NOUN PUNCT',
# .....
#'PRON VERB PRON ADV ADP ADV PUNCT SCONJ ADV PRON ADP ADJ AUX VERB ADP DET NOUN PRON VERB ADP VERB PRON PUNCT']
def create_mat(data):
    sentences = parse(data)
    
    mat = [[token["form"], token["upostag"]]    #aggiungi a mat la coppia
           for s in sentences                   #per ogni frase
           for token in s                       #per ogni token nella frase
           if token["upostag"] != '_']          #se il token è diverso da _
    
    
    frasi = [' '.join([token['upostag'] 
                for token in sentence 
                if token["upostag"] != "_"]) 
                for sentence in sentences]

    frasi = ["S0 " + s for s in frasi]

    return np.array(mat), np.array(frasi)



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
#[['ADV' '1039' '0.0855848434925865']
# .....
#['PART' '7' '0.0005766062602965404']]
def count_tags(data):
    tags = [x[1] for x in data]
    return calc_prob(tags)



#FUNZIONA
#conta quante volte compare una parola e ne calcola la probabilità a priori
#[['Non' '10' '0.0008237232289950577']
# .....
#'velle' '1' '8.237232289950576e-05']]
def count_words(data):
    words = [x[0] for x in data]
    return calc_prob(words)



#FUNZIONA
#calcola le probabilità di ogni tag di essere associato ad una certa parola
#[['ADV' 'Non' '10' '0.009624639076034648']
#['ADV' 'qui' '17' '0.016361886429258902']
# .....
#['PART' 'Oh' '3' '0.42857142857142855']
#['PART' 'O' '4' '0.5714285714285714']]
def calc_prob_emissione(data, tags):
    result = np.empty((0, 4))

    for tag in tags[:, 0]: # ciclo su tutti i tag

        row_tag = data[data[:, 1] == tag] # prende tutte le occorrenze di tag
        pairs = [tuple(x) for x in row_tag] # crea coppie (word, tag)
        pairs_counts = Counter(pairs) # conta le occorrenze di ogni coppia
        np_pairs_counts = np.array([[word, tag, count] for (tag, word), count in pairs_counts.items()]) # le converte in una lista numpy fatta: [word, tag, count]

        total_tag = count_tags[count_tags[:, 0] == tag][0, 1].astype(int) # cerca in count_tags il tag e ritorna il numero totale di volte che compare

        probability = np.array([[word, tag, int(count), int(count) / total_tag] for word, tag, count in np_pairs_counts]) #calcola la probabilità di tag di essere una certa parola e crea la lista: [word, tag, count, prob]
        result = np.vstack((result, probability)) # aggiunge la lista al risultato

    return result



#calcola la probabilità che un tag occorra dato il tag precedente
#se è la prima parola di una frase allora il tag prima sarà S0
#[['ADV', 'ADV', 78, 0.06200317965023847],
# .....
#['SYM', 'SYM', 0, 0.0]]
def calc_prob_transizione(tags, sentences):
    tags = np.vstack((tags, ["S0", len(sentences), '0']))

    print(tags)
    pairs = list(itertools.product([item[0] for item in tags], repeat = 2)) #crea tutte le possibili coppie di tag
    pairs = [[p[0], p[1], 0] for p in pairs]

    for s in sentences: #stringa di tag
        for element in s.split(): #tag singolo
            if element != "S0": #se non è il primo elemento
                #(before, element) per questa coppia devo fare +1 in pairs
                for p in pairs: #conta quante volte è presente la coppia x, y nel corpus
                    if p[0] == before and p[1] == element:
                        p[2] += 1
                        break
            before = element
    result = []
    for pair in pairs:
        total_tag = tags[tags[:, 0] == pair[0]][0, 1].astype(int)
        prob = pair[2] / total_tag
        result.append(prob)

    for pair, prob in zip(pairs, result):
        pair.append(prob)
    return pairs














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
with open("UD_Italian-VIT-master/it_vit-ud-test.conllu", "r", encoding="utf-8") as f:
    data_vit_test = f.read()
vit_test, vit_test_sentences = create_mat(data_vit_test)
#
#with open("UD_Italian-Old-master/it_old-ud-train.conllu", "r", encoding="utf-8") as f:
#    data_old_train = f.read()
## old_train = create_mat(data_old_train)
#
#with open("UD_Italian-Old-master/it_old-ud-dev.conllu", "r", encoding="utf-8") as f:
#    data_old_dev = f.read()
## old_dev = create_mat(data_old_dev)

#with open("UD_Italian-Old-master/it_old-ud-test.conllu", "r", encoding="utf-8") as f:
#    data_old_test = f.read()
#old_test = create_mat(data_old_test)

#[print(x) for x in old_test]
#--------------------------------------------------------------------------------------------------------------------------------------

count_tags = count_tags(vit_test)
count_words = count_words(vit_test)

prob_emissione = calc_prob_emissione(vit_test, count_tags)

prob_transizione = calc_prob_transizione(count_tags, vit_test_sentences)
print(prob_transizione)












