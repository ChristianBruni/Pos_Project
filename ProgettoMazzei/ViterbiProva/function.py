from conllu import parse
from collections import Counter, defaultdict
import itertools
import numpy as np
import pandas as pd
import math


def guess_emission_prob(word, state):
    """
    Stima la probabilità di emissione per parole non viste, basandosi sul suffisso.
    """

    dict = {
        'ADJ': [
            'oso', 'osa', 'ile', 'ente', 'ante', 'ivo', 'iva', 'esco', 'esca',
            'ario', 'aria', 'aceo', 'acea', 'ico', 'ica', 'ino', 'ina'
        ],
        'ADP': [
            'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra'
        ],
        'ADV': [
            'mente', 'qui', 'lì', 'là', 'via', 'giù', 'sù', 'ora', 'sempre', 'spesso', 'mai'
        ],
        'AUX': [
            'sti', 'rà', 'rò', 'rebbe', 'fui', 'sarà', 'sia'
        ],
        'CCONJ': [
            'ma', 'oppure', 'però', 'bensì', 'infatti', 'cioè'
        ],
        'DET': [
            'il', 'lo', 'la', 'gli', 'le', 'un', 'una', 'uno', 'questo', 'quella', 'quel', 'questi'
        ],
        'INTJ': [
            'oh', 'eh', 'ah', 'uff', 'bah', 'mah', 'wow', 'oops', 'accidenti', 'evviva'
        ],
        'NOUN': [
            'zione', 'tà', 'mento', 'ore', 'ista', 'tore', 'trice', 'aggio', 'ezza',
            'ismo', 'icità', 'ità', 'anza', 'enza', 'itudine', 'logia'
        ],
        'NUM': [
            'uno', 'due', 'tre', 'quattro', 'cinque', 'sei', 'sette', 'otto', 'nove',
            'dieci', 'cento', 'mille', 'milione', 'miliardo'
        ],
        'PART': [
            'ci', 'mi', 'ti', 'vi', 'si'
        ],
        'PRON': [
            'noi', 'voi', 'loro', 'egli', 'ella', 'esso'
        ],
        'PROPN': [
            'ini', 'etti', 'one', 'oni', 'ani', 'elli', 'ardo'
        ],
        'PUNCT': [
            '.', ',', ';', ':', '!', '?', '"', '«', '»', '…', '(', ')'
        ],
        'SCONJ': [
            'che', 'perché', 'quando', 'mentre', 'sebbene', 'poiché', 'se', 'nonostante', 'finché', 'dopo che'
        ],
        'VERB': [
            'are', 'ere', 'ire', 'ando', 'endo', 'ato', 'uto', 'ito', 'erei', 'erà', 'avo', 'ono', 'ano', 'iamo'
        ],
        'X': [
            'xxx', '???', '###', '!!!', 'null'
        ]
    }

    suffissi = dict.get(state, [])

    # Capitalization heuristic: parola maiuscola = PROPN
    if word[0].isupper():
        word = word.lower()
        if word in suffissi:
                return 1
        elif any(word.endswith(suf) for suf in suffissi):
            return 1

        if state == 'PROPN':
            return 1
        else:
            return 0

    word = word.lower()
    suffissi = dict.get(state, [])

    # Se la parola combacia con un suffisso noto
    if word in suffissi:
        return 1
    elif any(word.endswith(suf) for suf in suffissi):
        return 1

    # Fallback: probabilità medie o basse
    if state in ['NOUN', 'VERB']:
        return 0.5
    else:
        return 0


def viterbi_suffisso(observations, states, start_p, trans_p, emit_p):

    # Inizializzazione delle strutture dati
    viterbi = {state: [0.0] * len(observations) for state in states}
    backpointer = {state: [None] * len(observations) for state in states}

    ## Inizializzazione (primo passo)
    arr_emit = []
    for state in states:
        word = observations[0]
        if word not in emit_p[state]:
            emit_prob = guess_emission_prob(word, state)
        else:
            emit_prob = emit_p[state][word]
        arr_emit.append(emit_prob)

    sum_arr_emit = sum(arr_emit)
    if sum_arr_emit > 1:
        for emit, state in zip(arr_emit, states):
            viterbi[state][0] = start_p.get(state, 0) * (emit / sum_arr_emit)
            backpointer[state][0] = None
    else:
        for emit, state in zip(arr_emit, states):
            viterbi[state][0] = start_p.get(state, 0) * emit
            backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(observations)):
        word = observations[t]
        arr_emit2 = []
        for current_state in states:
            if word not in emit_p[current_state]:
                emit_prob = guess_emission_prob(word, current_state)
            else:
                emit_prob = emit_p[current_state][word]
            arr_emit2.append(emit_prob)

        sum_arr_emit = sum(arr_emit2)

        if sum_arr_emit > 1:
            for emit, current_state in zip(arr_emit2, states):
                max_prob = -1.0
                best_prev_state = None
                for prev_state in states:
                    trans_prob = trans_p[prev_state].get(current_state, 0)
                    prob = viterbi[prev_state][t-1] * trans_prob * (emit / sum_arr_emit)

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_state

                viterbi[current_state][t] = max_prob
                backpointer[current_state][t] = best_prev_state
        else:
            for emit, current_state in zip(arr_emit2, states):
                max_prob = -1.0
                best_prev_state = None
                for prev_state in states:
                    trans_prob = trans_p[prev_state].get(current_state, 0)
                    prob = viterbi[prev_state][t-1] * trans_prob * emit

                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_state

                viterbi[current_state][t] = max_prob
                backpointer[current_state][t] = best_prev_state

    # Terminazione
    best_last_state = max(states, key=lambda s: viterbi[s][-1])
    best_path = [best_last_state]
    for t in range(len(observations) - 1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(observations, best_path))





def viterbi_n(observations, states, start_p, trans_p, emit_p):
    # Inizializzazione delle strutture dati
    viterbi = {state: [0.0] * len(observations) for state in states}
    backpointer = {state: [None] * len(observations) for state in states}

    # Inizializzazione (primo passo)
    for state in states:
        word = observations[0]
        if word not in emit_p[state]:
            emit_prob = emit_p[state].get(word, 1 if state in 'NOUN' else 0.0)
        else:
            emit_prob = emit_p[state][word]
        viterbi[state][0] = start_p.get(state, 0) * emit_prob
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(observations)):
        word = observations[t]
        for current_state in states:
            max_prob = -1.0
            best_prev_state = None
            if word not in emit_p[current_state]:
                emit_prob = emit_p[current_state].get(word, 1 if current_state in 'NOUN' else 0.0)
            else:
                emit_prob = emit_p[current_state][word]

            for prev_state in states:
                trans_prob = trans_p[prev_state].get(current_state, 0)
                prob = viterbi[prev_state][t-1] * trans_prob * emit_prob

                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state

    # Terminazione (trova lo stato finale migliore)
    best_last_state = max(states, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]
    for t in range(len(observations)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(observations, best_path))



def viterbi_vn(observations, states, start_p, trans_p, emit_p):
    epsilon= 1e-6
    # Inizializzazione delle strutture dati
    viterbi = {state: [float('inf')] * len(observations) for state in states}
    backpointer = {state: [None] * len(observations) for state in states}

    # Inizializzazione (primo passo)
    for state in states:
        word = observations[0]
        if word not in emit_p[state]:
            emit_prob = emit_p[state].get(word, 0.5 if state in ['NOUN', 'VERB'] else epsilon)
        else:
            emit_prob = emit_p[state][word]
        emit_prob= max(emit_prob,epsilon)
        start_prob = start_p.get(state, 0.0)
        start_prob = max(start_prob, epsilon)
        viterbi[state][0] = (- math.log(start_prob)) + (- math.log(emit_prob))
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(observations)):
        word = observations[t]
        for current_state in states:
            max_prob = float('inf')
            best_prev_state = None
            if word not in emit_p[current_state]:
                emit_prob = emit_p[current_state].get(word, 0.5 if current_state in ['NOUN', 'VERB'] else epsilon)
            else:
                emit_prob = emit_p[current_state][word]
            emit_prob= max(emit_prob,epsilon)
            for prev_state in states:
                trans_prob = trans_p[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) + (- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state

    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(states, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]
    for t in range(len(observations)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(observations, best_path))

def viterbi_with_develop(observations, states, start_p, trans_p, emit_p, distribution_prob):
    # Inizializzazione delle strutture dati
    epsilon= 1e-6
    viterbi = {state: [float('inf')] * len(observations) for state in states}
    backpointer = {state: [None] * len(observations) for state in states}

    # Inizializzazione (primo passo)
    for state in states:
        word = observations[0]
        if word not in emit_p[state]:
            emit_prob = distribution_prob[state]
        else:
            emit_prob = emit_p[state][word]

        emit_prob= max(emit_prob,epsilon)
        start_prob = start_p.get(state, 0.0)
        start_prob = max(start_prob, epsilon)
        viterbi[state][0] = (- math.log(start_prob)) + (- math.log(emit_prob))
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(observations)):
        word = observations[t]
        for current_state in states:
            max_prob = float('inf')
            best_prev_state = None
            if word not in emit_p[current_state]:
                emit_prob = distribution_prob[current_state]
            else:
                emit_prob = emit_p[current_state][word]
            emit_prob= max(emit_prob,epsilon)
            for prev_state in states:
                trans_prob = trans_p[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) + (- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state

    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(states, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]
    for t in range(len(observations)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(observations, best_path))


def viterbi_uniform(observations, states, start_p, trans_p, emit_p):
    # Inizializzazione delle strutture dati
    epsilon= 1e-6
    viterbi = {state: [float('inf')] * len(observations) for state in states}
    backpointer = {state: [None] * len(observations) for state in states}

    # Inizializzazione (primo passo)
    for state in states:
        word = observations[0]
        if word not in emit_p[state]:
            emit_prob = 1/len(states)
        else:
            emit_prob = emit_p[state][word]

        emit_prob = max(emit_prob, epsilon)
        start_prob = start_p.get(state, 0.0)

        start_prob = max(start_prob, epsilon)
        viterbi[state][0] = math.log(start_prob) + math.log(emit_prob)
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(observations)):
        word = observations[t]
        for current_state in states:
            max_prob = float('inf')
            best_prev_state = None
            if word not in emit_p[current_state]:
                emit_prob = 1/len(states)
            else:
                emit_prob = emit_p[current_state][word]

            emit_prob = max(emit_prob, epsilon)

            for prev_state in states:
                trans_prob = trans_p[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) +(- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state
            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state
    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(states, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]
    for t in range(len(observations)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(observations, best_path))






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



#FUNZIONA
#calcola le probabilità di ogni tag di essere associato ad una certa parola
#[['ADV' 'Non' '10' '0.009624639076034648']
#['ADV' 'qui' '17' '0.016361886429258902']
# .....
#['PART' 'Oh' '3' '0.42857142857142855']
#['PART' 'O' '4' '0.5714285714285714']]
def calc_prob_emissione(data, tags):
    # Lista che conterrà i risultati finali
    result = []

    # Tutte le parole uniche nel dataset
    all_words = np.unique(data[:, 0])
    # Creiamo un dizionario con i tag e il loro numero totale di occorrenze
    total_tags = {tag[0]: int(tag[1]) for tag in tags}

    # Inizializza un contatore per tutte le possibili coppie (word, tag)
    all_pairs_counts = Counter({(word, tag[0]): 0 for word in all_words for tag in tags})

    # Aggiorna il contatore con le occorrenze effettive
    for word, tag in zip(data[:, 0], data[:, 1]):
        all_pairs_counts[(word, tag)] += 1

    # Ora calcoliamo la probabilità per tutte le coppie
    for tag in total_tags:  # ciclo su tutti i tag
        total_tag = total_tags[tag]  # Numero totale di occorrenze del tag corrente

        # Per ogni parola, calcoliamo la probabilità di quella parola rispetto al tag corrente
        for word in all_words:
            count = all_pairs_counts[(word, tag)]  # Conta le occorrenze della coppia (word, tag)
            prob = count / total_tag if total_tag > 0 else 0.0  # Calcola la probabilità

            # Aggiungi la riga al risultato
            result.append([word, tag, count, prob])

    # Converti la lista di risultati in un array NumPy
    result_array = np.array(result, dtype=object)

    return result_array





#FUNZIONA
#calcola la probabilità che un tag occorra dato il tag precedente
#se è la prima parola di una frase allora il tag prima sarà S0
#[['ADV', 'ADV', 78, 0.06200317965023847],
# .....
#['SYM', 'SYM', 0, 0.0]]
def calc_prob_transizione(sentences, tags):
    tags = np.vstack((tags, ["S0", len(sentences), '0']))

    #print(tags)
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
    return np.array(pairs)


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


def convert_table(table):
    trans_p = defaultdict(dict)
    #print(table)
    for x, y, count, prob in table:
        #print(x, y,count,prob)
        trans_p[x][y] = prob
    return dict(trans_p)

def convert_table_emissione(table):
    trans_p = defaultdict(dict)
    for x, y, count, prob in table:
        trans_p[y][x] = prob
    return dict(trans_p)

# modifica il formato dei dati
def format_dict(data):
    converted_dict = {}

    for outer_key, inner_dict in data.items():

        # Converti le chiavi da np.str_ a stringhe
        outer_key_str = str(outer_key)
        inner_dict_converted = {str(inner_key): float(value) for inner_key, value in inner_dict.items()}

        # Assegna il dizionario interno al tag esterno
        converted_dict[outer_key_str] = inner_dict_converted

    return converted_dict



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


#FUNZIONA
#conta quante volte compare un tag e ne calcola la probabilità a priori
#[['ADV' '1039' '0.0855848434925865']
# .....
#['PART' '7' '0.0005766062602965404']]
def fn_count_tags(data):
    tags = [x[1] for x in data]
    return calc_prob(tags)

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


def baseline_tagger(words, tags, dict):
    state = []  # lista dei tag assegnati parola per parola
    tags_list = tags[:, 0]  # tutti i tag  # i tag sono le colonne nella matrice

    for word in words:  # per ogni parola da taggare
        if word in dict.index:
            tag_probs = dict.loc[word]  # Serie con P(tag | parola)
            best_tag = tag_probs.idxmax()  # tag con la probabilità più alta
        else:
            best_tag = 'NOUN'  # smoothing: parola mai vista, default a NOUN
        state.append(best_tag)

    return list(zip(words, state))



def tagging(train, train_tags, test_tags, test_words):

    count_tags = fn_count_tags(train)
    #trasforma i tag in un lista di tag
    tags_list = count_tags[:, 0].tolist()
    #tags_list.append('S0')     #aggiunge S0 ai tag (17 in totale)

    #calcola le probabilità di emissione
    prob_emissione = calc_prob_emissione(train, count_tags)

    #calcola le probabilità di transizione
    prob_transizione = calc_prob_transizione(train_tags,count_tags)
    #print(prob_transizione)

    #scrivere meglio
    prob_transizione_after = convert_table(prob_transizione)
    transition_dict = format_dict(prob_transizione_after)

    prob_emissione_after = convert_table_emissione(prob_emissione)
    emission_dict = format_dict(prob_emissione_after)


    start = start_prob(prob_transizione_after)
    tags_list_new = count_tags[:, 0].tolist()

    predicted_tags = []

    for i, sentence in enumerate(test_words):
        tagged_seq = viterbi_vn(sentence.split(), tags_list_new, start, transition_dict, emission_dict)
        print(sentence.split())
        acc = calculate_accuracy(tagged_seq, test_tags[i])
        print(acc)
        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)

def tagging_with_develop(train, train_tags, test_tags, test_words,dev):
    count_tags = fn_count_tags(train)
    #trasforma i tag in un lista di tag
    tags_list = count_tags[:, 0].tolist()
    #tags_list.append('S0')     #aggiunge S0 ai tag (17 in totale)

    #calcola le probabilità di emissione
    prob_emissione = calc_prob_emissione(train, count_tags)

    #calcola le probabilità di transizione
    prob_transizione = calc_prob_transizione(train_tags,count_tags)

    #scrivere meglio
    prob_transizione_after = convert_table(prob_transizione)
    transition_dict = format_dict(prob_transizione_after)

    prob_emissione_after = convert_table_emissione(prob_emissione)
    emission_dict = format_dict(prob_emissione_after)
    start = start_prob(prob_transizione_after)
    tags_list_new = count_tags[:, 0].tolist()


    distribution_prob_unknown= prob_dev_distribution(dev, tags_list)
    predicted_tags = []

    for i, sentence in enumerate(test_words):
        tagged_seq = viterbi_with_develop(sentence.split(), tags_list_new, start, transition_dict, emission_dict, distribution_prob_unknown)
        #acc = calculate_accuracy(tagged_seq, test_tags[i])
        #print(acc)
        predicted_tags.append(tagged_seq)


    return calc_accuracy(predicted_tags, test_tags)

def evaluate_baseline(train, train_tags, test_tags, test_words):

    # conta i tag
    count_tags = fn_count_tags(train)
    tags_list = count_tags[:, 0].tolist()
    tags_list.append('S0')  # aggiunge S0

    # calcola le probabilità di emissione
    dict_word_tag = prob_words_tag(train, count_tags)
    predicted_tags = []
    for sentence in test_words:
        tagged_seq = baseline_tagger(sentence.split(), count_tags, dict_word_tag)
        predicted_tags.append(tagged_seq)

    return calc_accuracy(predicted_tags, test_tags)