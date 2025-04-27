from conllu import parse
from collections import Counter, defaultdict
import itertools
import numpy as np
import pandas as pd
import viterbi as viterbi
import math









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


#FUNZIONA
#conta e calcola le probabilità di comparsa di parole/tag
def calc_prob(data):
    data_counts = Counter(data)
    np_data_counts = np.array([[word, count] for word, count in data_counts.items()])
    counts = np_data_counts[:, 1].astype(int)
    freq = (counts / np.sum(counts)).reshape(-1, 1)
    return np.hstack((np_data_counts, freq))




#FUNZIONA
#conta quante volte compare una parola e ne calcola la probabilità a priori
#[['Non' '10' '0.0008237232289950577']
# .....
#'velle' '1' '8.237232289950576e-05']]
def count_words(data):
    words = [x[0] for x in data]
    return calc_prob(words)




def prob_dev_distribution(dev, tags):
    result = {}
    # 1. Conta le occorrenze di ogni parola
    word_counts = Counter(dev[:, 0])

    # 2. Trova le parole che compaiono una sola volta
    single_words = {word for word, count in word_counts.items() if count == 1}
    total_single = len(single_words)
    # 3. Conta i tag associati a queste parole singole
    tag_counts = {}

    for word, tag in dev:
        if word in single_words:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # 4. Calcola la distribuzione: P(tag | parola singola)
    for tag in tags:
        prob = tag_counts.get(tag, 0) / total_single if total_single > 0 else 0
        result[tag] = prob

    return result








#FUNZIONA
#conta quante volte compare un tag e ne calcola la probabilità a priori
#[['ADV' '1039' '0.0855848434925865']
# .....
#['PART' '7' '0.0005766062602965404']]
def fn_count_tags(data):
    tags = [x[1] for x in data]
    return calc_prob(tags)



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

def prob_words_tag(data, tags):
    parole = np.unique(data[:, 0])   # tutte le parole uniche
    tags_list = tags[:, 0]  # tutti i tag

    # Inizializza la matrice: righe = parole, colonne = tag
    mat = pd.DataFrame(index=parole, columns=tags_list, data=0.0)

    # Conta tutte le (parola, tag)
    pairs = [tuple(x) for x in data]
    pairs_counts = Counter(pairs)

    # Conta quante volte compare ogni parola
    word_counts = Counter(data[:, 0])

    for (word, tag), count in pairs_counts.items():
        total_word = word_counts[word]
        prob = count / total_word if total_word > 0 else 0
        mat.loc[word, tag] = prob

    return mat



def baseline_tagger(words, dict):

    state = []  # lista dei tag assegnati parola per parola

    for word in words:  # per ogni parola da taggare
        if word in dict.index:
            tag_probs = dict.loc[word]  # Serie con P(tag | parola)
            best_tag = tag_probs.idxmax()  # tag con la probabilità più alta
        else:
            best_tag = 'NOUN'  # smoothing: parola mai vista, default a NOUN
        state.append(best_tag)

    return list(zip(words, state))



def evaluate_baseline(train, train_tags, test_tags, test_words):

    # Conta i tag
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()
    tags_list.append('S0')  # aggiunge S0

    # Calcola le probabilità di emissione
    dict_word_tag = prob_words_tag(train, count_tags)
    predicted_tags = []
    for sentence in test_words:
        tagged_seq = baseline_tagger(sentence.split(), dict_word_tag)
        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)


























# Calcola le probabilità di ogni tag di essere associato ad una certa parola
def calc_emission_probability(train, tags):

    # Tutte le parole uniche
    all_words = np.unique(train[:, 0])

    # Totale occorrenze per ogni tag
    total_tags = {tag[0]: int(tag[1]) for tag in tags}

    # Conta tutte le coppie possibili (word, tag)
    all_pairs_counts = Counter({(word, tag[0]): 0 for word in all_words for tag in tags})
    for word, tag in zip(train[:, 0], train[:, 1]):
        all_pairs_counts[(word, tag)] += 1

    # Crea direttamente il dizionario emissione
    emission_dict = defaultdict(dict)
    for tag in total_tags:
        total_tag = total_tags[tag]
        for word in all_words:
            count = all_pairs_counts[(word, tag)]
            prob = count / total_tag if total_tag > 0 else 0.0
            emission_dict[tag][word] = prob

    return dict(emission_dict)



# Calcola la probabilità che un tag occorra dato il tag precedente
# Se è la prima parola di una frase allora il tag prima sarà S0
def calc_transition_probability(train_tags, tags):

    # Aggiungiamo "S0" come tag iniziale
    tag_counts = np.vstack((tags, ["S0", len(train_tags), '0']))
    all_tags = [tag[0] for tag in tag_counts]

    # Inizializza dizionario delle transizioni
    trans_counts = defaultdict(lambda: defaultdict(int))

    # Conta le occorrenze delle coppie (prev_tag -> curr_tag)
    for sentence in train_tags:
        before = "S0"
        for tag in sentence.split():
            trans_counts[before][tag] += 1
            before = tag

    # Calcola le probabilità direttamente
    trans_dict = defaultdict(dict)
    for prev_tag in all_tags:
        total_prev_tag = int(tag_counts[tag_counts[:, 0] == prev_tag][0, 1])
        for curr_tag in all_tags:
            count = trans_counts[prev_tag].get(curr_tag, 0)
            prob = count / total_prev_tag if total_prev_tag > 0 else 0.0
            trans_dict[prev_tag][curr_tag] = prob

    return dict(trans_dict)



# Calcola le probabilità ........................................................................................
def start_prob(numpy_dict):
    converted_dict = {}

    for outer_key, inner_dict in numpy_dict.items():
        # Converti le chiavi da np.str_ a stringhe
        outer_key_str = str(outer_key)
        inner_dict_converted = {str(inner_key): float(value) for inner_key, value in inner_dict.items()}

        # Assegna il dizionario interno al tag esterno
        converted_dict[outer_key_str] = inner_dict_converted
    # Ora estrai solo i valori associati a 'S0', rimuovendo 'S0' stesso se è presente nei tag
    start_probs = {tag: prob for tag, prob in converted_dict['S0'].items() if tag != 'S0'}

    return start_probs













# Calcola l'accuratezza dei tag
# input predict_seq: sequenza di tutti i tag predetti
# input true_seq: sequenza di tutti i veri tag
# output: accuratezza
def calc_accuracy(predict_seq, true_seq):

    predict = [str(tag) for sentence in predict_seq for _, tag in sentence] # estrae dalla sequenza predetta tutti i tag e li mette in un vettore

    true = [tag for s in true_seq for tag in s.item().split() if tag != 'S0'] # estrae dalla vera sequenza tutti i tag e li mette in un vettore

    correct = sum(1 for p, t in zip(predict, true) if p == t) # somma 1 per ogni tag corretto

    return correct / len(true) # calcola l'accuratezza (# tag corretti / # totale di tag)



def tagging_n(train, train_tags, test_tags, test_words):

    # Conta quanti tag singoli compaiono nel training set
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()

    # Calcola probabilità di emissione e transizione
    em_prob = calc_emission_probability(train, count_tags)
    tr_prob = calc_transition_probability(train_tags, count_tags)

    # Calcola probabilità della prima parola della frase
    start = start_prob(tr_prob)

    predicted_tags = []

    for i, sentence in enumerate(test_words):

        tagged_seq = viterbi.viterbi_n(sentence.split(), tags_list, start, tr_prob, em_prob)

        #acc = calculate_accuracy(tagged_seq, test_tags[i])
        #print(acc)

        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)



def tagging_nv(train, train_tags, test_tags, test_words):

    # Conta quanti tag singoli compaiono nel training set
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()

    # Calcola probabilità di emissione e transizione
    em_prob = calc_emission_probability(train, count_tags)
    tr_prob = calc_transition_probability(train_tags, count_tags)

    # Calcola probabilità della prima parola della frase
    start = start_prob(tr_prob)

    predicted_tags = []

    for i, sentence in enumerate(test_words):

        tagged_seq = viterbi.viterbi_vn(sentence.split(), tags_list, start, tr_prob, em_prob)

        #acc = calculate_accuracy(tagged_seq, test_tags[i])
        #print(acc)

        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)



def tagging_dev(train, train_tags, test_tags, test_words, dev):

    # Conta quanti tag singoli compaiono nel training set
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()

    # Calcola probabilità di emissione e transizione
    em_prob = calc_emission_probability(train, count_tags)
    tr_prob = calc_transition_probability(train_tags, count_tags)

    # Calcola probabilità della prima parola della frase
    start = start_prob(tr_prob)

    # Calcola la distribuzione delle parole che compaiono una sola volta nel development set
    dev_dist = prob_dev_distribution(dev, tags_list)

    predicted_tags = []

    for i, sentence in enumerate(test_words):
        tagged_seq = viterbi.viterbi_dev(sentence.split(), tags_list, start, tr_prob, em_prob, dev_dist)

        # acc = calculate_accuracy(tagged_seq, test_tags[i])
        # print(acc)

        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)



def tagging_uniform(train, train_tags, test_tags, test_words):

    # Conta quanti tag singoli compaiono nel training set
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()

    # Calcola probabilità di emissione e transizione
    em_prob = calc_emission_probability(train, count_tags)
    tr_prob = calc_transition_probability(train_tags, count_tags)

    # Calcola probabilità della prima parola della frase
    start = start_prob(tr_prob)

    predicted_tags = []

    for i, sentence in enumerate(test_words):

        tagged_seq = viterbi.viterbi_uniform(sentence.split(), tags_list, start, tr_prob, em_prob)

        #acc = calculate_accuracy(tagged_seq, test_tags[i])
        #print(acc)

        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)



def tagging_syntax(train, train_tags, test_tags, test_words):

    # Conta quanti tag singoli compaiono nel training set
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()

    # Calcola probabilità di emissione e transizione
    em_prob = calc_emission_probability(train, count_tags)
    tr_prob = calc_transition_probability(train_tags, count_tags)

    # Calcola probabilità della prima parola della frase
    start = start_prob(tr_prob)

    predicted_tags = []

    for i, sentence in enumerate(test_words):

        tagged_seq = viterbi.viterbi_sintax(sentence.split(), tags_list, start, tr_prob, em_prob)

        #acc = calculate_accuracy(tagged_seq, test_tags[i])
        #print(acc)

        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)