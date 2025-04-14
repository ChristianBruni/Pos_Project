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
def fn_count_tags(data):
    tags = [x[1] for x in data]
    return calc_prob(tags)



# FUNZIONA
# conta quante volte compare una parola e ne calcola la probabilità a priori
# [['Non' '10' '0.0008237232289950577']
# .....
# 'velle' '1' '8.237232289950576e-05']]
def fn_count_words(data):
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
            prob = count / total_tag if total_tag > 0 else 0
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
# output: accuratezza
def calc_accuracy(predict_seq, true_seq):

    predict = [str(tag) for sentence in predict_seq for _, tag in sentence] # estrae dalla sequenza predetta tutti i tag e li mette in un vettore

    true = [tag for s in true_seq for tag in s.item().split() if tag != 'S0'] # estrae dalla vera sequenza tutti i tag e li mette in un vettore

    correct = sum(1 for p, t in zip(predict, true) if p == t) # somma 1 per ogni tag corretto

    return correct / len(true) # calcola l'accuratezza (# tag corretti / # totale di tag)


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

def tagging(train, train_tags, test_tags, test_words):

    # conta i tag
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()
    tags_list.append('S0')  # aggiunge S0

    # conta le parole
    #count_words = fn_count_words(train)
    #words_list = count_words[:, 0].tolist()

    # calcola le probabilità di emissione
    em = emission_matrix(train, count_tags)

    # calcola le probabilità di transizione
    tm = transition_matrix(train_tags, count_tags)
    # print(pd.DataFrame(transition_matrix, columns = list(tags_list), index = list(tags_list)))

    predicted_tags = []
    for i, sentence in enumerate(test_words):
        tagged_seq = viterbi(sentence.split(), count_tags, tm, em)
        acc = calculate_accuracy(tagged_seq, test_tags[i])
        print(acc)
        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)