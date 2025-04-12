from conllu import parse
import numpy as np
from collections import Counter, defaultdict
import itertools
import pandas as pd
import hmmlearn
import hidden_markov
import pprint
from hmmlearn import hmm


# FUNZIONA
# crea la matrice con tutte le parole e il corrispondente tag e crea il vettore con tutte le frasi
# [['Non' 'ADV']
# .....
# ['stelle' 'NOUN']]

# ['ADV AUX VERB NOUN ADJ ADP DET NOUN ADP DET NOUN ADP NOUN PUNCT',
# .....
# 'PRON VERB PRON ADV ADP ADV PUNCT SCONJ ADV PRON ADP ADJ AUX VERB ADP DET NOUN PRON VERB ADP VERB PRON PUNCT']
def create_mat(data):
    sentences = parse(data)

    mat = [[token["form"], token["upostag"]]  # aggiungi a mat la coppia
           for s in sentences  # per ogni frase
           for token in s  # per ogni token nella frase
           if token["upostag"] != '_']  # se il token è diverso da _

    frasi_tag = [' '.join([token['upostag']
                           for token in sentence
                           if token["upostag"] != "_"])
                 for sentence in sentences]

    frasi_tag = ["S0 " + s for s in frasi_tag]

    frasi_parole = [' '.join([token['form']
                              for token in sentence
                              if token["upostag"] != "_"])
                    for sentence in sentences]

    return np.array(mat), np.array(frasi_tag), np.array(frasi_parole)


# FUNZIONA
# conta e calcola le probabilità di comparsa di parole/tag
def calc_prob(data):
    data_counts = Counter(data)
    np_data_counts = np.array([[word, count] for word, count in data_counts.items()])
    counts = np_data_counts[:, 1].astype(int)
    freq = (counts / np.sum(counts)).reshape(-1, 1)
    return np.hstack((np_data_counts, freq))


# FUNZIONA
# conta quante volte compare un tag e ne calcola la probabilità a priori
# [['ADV' '1039' '0.0855848434925865']
# .....
# ['PART' '7' '0.0005766062602965404']]
def count_tags(data):
    tags = [x[1] for x in data]
    return calc_prob(tags)


# FUNZIONA
# conta quante volte compare una parola e ne calcola la probabilità a priori
# [['Non' '10' '0.0008237232289950577']
# .....
# 'velle' '1' '8.237232289950576e-05']]
def count_words(data):
    words = [x[0] for x in data]
    return calc_prob(words)


# FUNZIONA
# calcola le probabilità di ogni tag di essere associato ad una certa parola
# [['ADV' 'Non' '10' '0.009624639076034648']
# ['ADV' 'qui' '17' '0.016361886429258902']
# .....
# ['PART' 'Oh' '3' '0.42857142857142855']
# ['PART' 'O' '4' '0.5714285714285714']]
#def calc_prob_emissione(data, tags):
#    result = np.empty((0, 4))
#
#    for tag in tags[:, 0]:  # ciclo su tutti i tag
#
#        row_tag = data[data[:, 1] == tag]  # prende tutte le occorrenze di tag
#        pairs = [tuple(x) for x in row_tag]  # crea coppie (word, tag)
#        pairs_counts = Counter(pairs)  # conta le occorrenze di ogni coppia
#        np_pairs_counts = np.array([[word, tag, count] for (tag, word), count in
#                                    pairs_counts.items()])  # le converte in una lista numpy fatta: [word, tag, count]
#
#        total_tag = count_tags[count_tags[:, 0] == tag][0, 1].astype(
#            int)  # cerca in count_tags il tag e ritorna il numero totale di volte che compare
#
#        probability = np.array([[word, tag, int(count), int(count) / total_tag] for word, tag, count in
#                                np_pairs_counts])  # calcola la probabilità di tag di essere una certa parola e crea la lista: [word, tag, count, prob]
#        result = np.vstack((result, probability))  # aggiunge la lista al risultato
#    return result



def emission_matrix(data, count_tags):
    tags = count_tags[:, 0]  # tutti i tag
    parole = np.unique(data[:, 0])  # tutte le parole uniche
    emission_matrix = pd.DataFrame(index=tags, columns=parole, data=0.0)

    for tag in tags:
        row_tag = data[data[:, 1] == tag]  # tutte le righe con quel tag
        pairs = [tuple(x) for x in row_tag]  # crea (word, tag)
        pairs_counts = Counter(pairs)

        total_tag = int(count_tags[count_tags[:, 0] == tag][0, 1])  # numero totale di volte che compare il tag

        for (word, _), count in pairs_counts.items():
            prob = count / total_tag if total_tag > 0 else 0.0
            emission_matrix.loc[tag, word] = prob

    return emission_matrix


# FUNZIONA
# calcola la probabilità che un tag occorra dato il tag precedente
# se è la prima parola di una frase allora il tag prima sarà S0
# [['ADV', 'ADV', 78, 0.06200317965023847],
# .....
# ['SYM', 'SYM', 0, 0.0]]
#def calc_prob_transizione(tags, sentences):
#    tags = np.vstack((tags, ["S0", len(sentences), '0']))
#
#    # print(tags)
#    pairs = list(itertools.product([item[0] for item in tags], repeat=2))  # crea tutte le possibili coppie di tag
#    pairs = [[p[0], p[1], 0] for p in pairs]
#
#    for s in sentences:  # stringa di tag
#        for element in s.split():  # tag singolo
#            if element != "S0":  # se non è il primo elemento
#                # (before, element) per questa coppia devo fare +1 in pairs
#                for p in pairs:  # conta quante volte è presente la coppia x, y nel corpus
#                    if p[0] == before and p[1] == element:
#                        p[2] += 1
#                        break
#            before = element
#    result = []
#    for pair in pairs:
#        total_tag = tags[tags[:, 0] == pair[0]][0, 1].astype(int)
#        prob = pair[2] / total_tag
#        result.append(prob)
#
#    for pair, prob in zip(pairs, result):
#        pair.append(prob)
#    return np.array(pairs)

def transition_matrix(tags, sentences):
    tags = np.vstack((tags, ["S0", len(sentences), '0']))

    tag_list = [item[0] for item in tags]
    pairs = list(itertools.product(tag_list, repeat=2))  # tutte le possibili coppie (prev_tag, next_tag)
    counts = { (p[0], p[1]): 0 for p in pairs }

    for s in sentences:
        before = "S0"  # prima parola della frase parte da "S0"
        for element in s.split():
            counts[(before, element)] += 1
            before = element

    # Calcola le probabilità di transizione
    transition_matrix = pd.DataFrame(index=tag_list, columns=tag_list, data=0.0)

    for (prev_tag, next_tag), count in counts.items():
        total_prev = int(tags[tags[:, 0] == prev_tag][0, 1])  # frequenza totale del tag precedente
        prob = count / total_prev if total_prev > 0 else 0.0
        transition_matrix.loc[next_tag, prev_tag] = prob

    return transition_matrix




def start_tag(transition_matrix):
    last_col = transition_matrix.columns[-1]  # es. "S0"
    start = transition_matrix[last_col].values.copy()  # copia per sicurezza
    start[-1] = 0.0  # azzera l'ultimo valore
    return start


# FUNZIONA
# leggi i file
# with open("UD_Italian-VIT-master/it_vit-ud-train.conllu", "r", encoding="utf-8") as f:
#    data_vit_train = f.read()
## vit_train = create_mat(data_vit_train)
#
# with open("UD_Italian-VIT-master/it_vit-ud-dev.conllu", "r", encoding="utf-8") as f:
#    data_vit_dev = f.read()
## vit_dev = create_mat(data_vit_dev)
#
with open("UD_Italian-VIT-master/it_vit-ud-test.conllu", "r", encoding="utf-8") as f:
    data_vit_test = f.read()
vit_test, vit_test_tags, vit_test_words = create_mat(data_vit_test)

#
# with open("UD_Italian-Old-master/it_old-ud-train.conllu", "r", encoding="utf-8") as f:
#    data_old_train = f.read()
## old_train = create_mat(data_old_train)
#
# with open("UD_Italian-Old-master/it_old-ud-dev.conllu", "r", encoding="utf-8") as f:
#    data_old_dev = f.read()
## old_dev = create_mat(data_old_dev)

# with open("UD_Italian-Old-master/it_old-ud-test.conllu", "r", encoding="utf-8") as f:
#    data_old_test = f.read()
# old_test = create_mat(data_old_test)

# --------------------------------------------------------------------------------------------------------------------------------------

# conta tag e parole
count_tags = count_tags(vit_test)
count_words = count_words(vit_test)

# trasforma i tag in un lista di tag
tags_list = count_tags[:, 0].tolist()
tags_list.append('S0')  # aggiunge S0 ai tag (17 in totale)

# calcola le probabilità di emissione
emission_matrix = emission_matrix(vit_test, count_tags)
#display(pd.DataFrame(emission_matrix, columns = list(tags_list), index=list(count_words[:, 0])))

# calcola le probabilità di transizione
transition_matrix = transition_matrix(count_tags, vit_test_tags)
#display(pd.DataFrame(transition_matrix, columns = list(tags_list), index=list(tags_list)))

start_tag = start_tag(transition_matrix)  # dimensione 17
#print(start_tag)

print(emission_matrix)




                #sentence, vit_test, transition_matri, emission_matrix
def Viterbi_common(words, train_bag, tags_df, emission_matrix):





    state = []
    T = list(set([pair[1] for pair in train_bag]))  # tutti i tag possibili

    for key, word in enumerate(words):
        p = []
        for tag in T:
            # Probabilità di transizione
            if key == 0:
                transition_p = tags_df.loc['S0', tag]  # "S0" è lo stato iniziale
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # Probabilità di emissione: usa la matrice precomputata
            if word in emission_matrix.columns:
                emission_prob = emission_matrix.at[tag, word] if word in emission_matrix.columns else 0.0
            else:
                emission_prob = 1.0  # smoothing: parola non vista

            # Probabilità totale
            state_probability = emission_prob * transition_p
            p.append(state_probability)

        # Scelta del tag più probabile
        pmax = max(p)
        if pmax == 0.0:
            state_word = 'NOUN'  # fallback tag
        else:
            state_word = T[p.index(pmax)]

        state.append(state_word)
    return list(zip(words, state))



for sentence in vit_test_words:
    sentence = sentence.split()

tagged_seq_vanilla = Viterbi_common(sentence, vit_test, transition_matrix, emission_matrix)
#print(tagged_seq_vanilla)
#
#check = [1 for (w1, t1), (w2, t2) in zip(tagged_seq_vanilla, vit_test) if w1 == w2 and t1 == t2]
#accuracy = len(check) / len(tagged_seq_vanilla)
#print(accuracy)




def calculate_accuracy(predicted_seq, true_seq):
    # Filtriamo il tag S0 e estraiamo solo i tag
    predicted_tags = [tag for _, tag in predicted_seq if tag != 'S0']
    true_seq = true_seq.split()
    true_tags = [tag for tag in true_seq if tag != 'S0']

    print(predicted_tags)
    print(true_tags)

    # Confrontiamo tag corrispondenti
    correct_tags = [1 for p, t in zip(predicted_tags, true_tags) if p == t]

    # Calcoliamo l'accuratezza
    accuracy = len(correct_tags) / len(true_tags)
    return accuracy




accuracy = calculate_accuracy(tagged_seq_vanilla, vit_test_tags[-1])
print("Accuracy:", accuracy)






