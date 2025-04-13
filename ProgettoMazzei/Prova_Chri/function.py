from conllu import parse
import numpy as np
from collections import Counter
import pandas as pd
import itertools

# FUNZIONA
# crea la matrice con tutte le parole e il corrispondente tag e crea il vettore con tutte le frasi

# mat
# [['Non' 'ADV']
# .....
# ['stelle' 'NOUN']]

# frasi_tag
# ['ADV AUX VERB NOUN ADJ ADP DET NOUN ADP DET NOUN ADP NOUN PUNCT',
# .....
# 'PRON VERB PRON ADV ADP ADV PUNCT SCONJ ADV PRON ADP ADJ AUX VERB ADP DET NOUN PRON VERB ADP VERB PRON PUNCT']

#frasi_parole
# ["Non sono consentite assegnazioni provvisorie in l' ambito di il comune di titolarità .",
# .....
# 'Questo mettiamo lo bene in sodo , se no niente di maraviglioso potrà scaturire da la storia che son per narrar vi .']
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



# calcola la matrice di emissione
# input data: elenco parole taggate
# input tags: elenco dei possibili tag
# output emission_matrix: tag x all_words
def emission_matrix(data, tags):
    tags_list = tags[:, 0]  # tutti i tag
    parole = np.unique(data[:, 0]) # tutte le parole uniche

    mat = pd.DataFrame(index = tags_list, columns = parole, data = 0.0)

    for tag in tags_list:
        row_tag = data[data[:, 1] == tag] # tutte le righe con quel tag
        pairs = [tuple(x) for x in row_tag] # crea (word, tag)
        pairs_counts = Counter(pairs)

        total_tag = int(tags[tags[:, 0] == tag][0, 1]) # numero totale di volte che compare il tag

        for (word, _), count in pairs_counts.items():
            prob = count / total_tag if total_tag > 0 else 1e-6
            mat.loc[tag, word] = prob

    return mat



# calcola la matrice di transizione
# input sentences: elenco delle frasi con i tag al posto delle parole
# input tags: elenco dei possibili tag
# output transition_matrix: tag x tag
def transition_matrix(sentences, tags):
    tags = np.vstack((tags, ["S0", len(sentences), '0']))

    tag_list = [item[0] for item in tags]
    pairs = list(itertools.product(tag_list, repeat = 2))  # tutte le possibili coppie (prev_tag, next_tag)
    counts = { (p[0], p[1]): 0 for p in pairs }

    for s in sentences:
        before = "S0"  # prima parola della frase parte da "S0"
        for element in s.split():
            counts[(before, element)] += 1
            before = element

    # Calcola le probabilità di transizione
    transition_matrix = pd.DataFrame(index = tag_list, columns = tag_list, data = 0.0)

    for (prev_tag, next_tag), count in counts.items():
        total_prev = int(tags[tags[:, 0] == prev_tag][0, 1])  # frequenza totale del tag precedente
        prob = count / total_prev if total_prev > 0 else 0.0
        transition_matrix.loc[next_tag, prev_tag] = prob

    return transition_matrix



# FUNZIONA MA NON SERVE
# calcola la probabilità del primo tag
# input transition_matrix
# output start: array con le probabilità dei tag
def start_tag(transition_matrix):
    last_col = transition_matrix.columns[-1]  # es. "S0"
    start = transition_matrix[last_col].values.copy()  # copia per sicurezza
    start[-1] = 0.0  # azzera l'ultimo valore (se non si fa rimane 1)
    return start



# TESTARE
# input words: elenco delle parole di una singola frase da taggare
# input train_set: training set (serve solo per ricavare tutti i possibili tag) (modificare?)
# input tm: transition matrix
# input em: emission matrix
def viterbi(words, tags, tm, em):

    state = [] # lista dei tag assegnati parola per parola
    #tag_list = list(set([pair[1] for pair in tags]))  # tutti i tag possibili
    tags = tags[:, 0].tolist()

    for key, word in enumerate(words): # ciclo su tutte le parole da taggare
        p = [] # lista delle probabilità di ogni tag per la parola

        for tag in tags: # cicla su tutti i possibili tag

            # probabilità di transizione
            if key == 0:
                tr_prob = tm.loc[tag, 'S0']  # "S0" è lo stato iniziale
            else:
                tr_prob = tm.loc[tag, state[-1]]

            # probabilità di emissione
            if word in em.columns:
                em_prob = em.at[tag, word] # se la parola è nella matrice di emissione
            else:
                em_prob = 1e-6  # smoothing: parola non vista

            # probabilità totale
            final_prob = em_prob * tr_prob
            p.append(final_prob)

        max_prob = max(p) # sceglie il tag con probabilità maggiore
        #print(word, max_prob)
        state_word = 'NOUN' if max_prob == 0.0 else tags[p.index(max_prob)]
        state.append(state_word)

    return list(zip(words, state))



# calcola l'accuratezza dei tag
# input predict_seq: sequenza di tutti i tag predetti
# input true_seq: sequenza di tutti i veri tag
def calc_accuracy(predict_seq, true_seq):

    flat_result = []
    for sentence in predict_seq:
        for word, tag in sentence:
            # Se il tag è di tipo np.str_, estrai il valore con .item(), altrimenti prendilo diretto
            if hasattr(tag, "item"):
                flat_result.append(tag.item())
            else:
                flat_result.append(tag)



    flat_result = [tag for tag in flat_result if tag != 'S0'] # rimozione dei tag S0 (non dovrebbero essercene)

    all_tags = []
    for s in true_seq:
        all_tags.extend(s.item().split())

    true_tags = [tag for tag in all_tags if tag != 'S0'] # rimozione dei tag S0

    correct = sum(1 for p, t in zip(flat_result, true_tags) if p == t) # somma 1 per ogni tag corretto

    accuracy = correct / len(true_tags)

    return accuracy


















































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