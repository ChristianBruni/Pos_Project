from conllu import parse
import numpy as np
from collections import Counter, defaultdict
import itertools
import hmmlearn
import hidden_markov
import pprint
from hmmlearn import hmm


def viterbi(obs, states, start_p, trans_p, emit_p):
    V=[{}]
    for i in states:
        V[0][i]=start_p[i]*emit_p[i][obs[0]]
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
        for i in dptable(V):
            print (i)
        opt=[]
        for j in V:
            for x,y in j.items():
                if j[x]==max(j.values()):
                    opt.append(x)
    #the highest probability
    h=max(V[-1].values())
    print ('The steps of states are '+' '.join(opt)+' with highest probability of %s'%h)
    #it prints a table of steps from dictionary

def dptable(V):
    yield " ".join(("%10d" % i) for i in range(len(V)))
    for y in V[0]:
        yield "%.7s: " % y+" ".join("%.7s" % ("%f" % v[y]) for v in V)

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
import numpy as np
from collections import Counter

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
def calc_prob_transizione(tags, sentences):
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
vit_test, vit_test_tags, vit_test_words = create_mat(data_vit_test)

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

#--------------------------------------------------------------------------------------------------------------------------------------

#conta tag e parole
count_tags = count_tags(vit_test)
count_words = count_words(vit_test)
#trasforma i tag in un lista di tag
tags_list = count_tags[:, 0].tolist()
tags_list.append('S0')     #aggiunge S0 ai tag (17 in totale)

#calcola le probabilità di emissione
prob_emissione = calc_prob_emissione(vit_test, count_tags)
#print(prob_emissione)
#calcola le probabilità di transizione
prob_transizione = calc_prob_transizione(count_tags, vit_test_tags)
#print(prob_transizione)

#scrivere meglio
prob_transizione_after = convert_table(prob_transizione)
transition_dict = format_dict(prob_transizione_after)

prob_emissione_after = convert_table_emissione(prob_emissione)
emission_dict = format_dict(prob_emissione_after)
print(emission_dict['ADV'])
# Conta gli elementi per ogni chiave
for chiave, valore in emission_dict.items():
    print(f"La chiave '{chiave}' contiene {len(valore)} elementi.")

start = start_prob(prob_transizione_after)  #dimensione 17
states = [word for sentence in vit_test_words for word in sentence.split()]
#viterbi(tags_list,states,start,transition_dict,emission_dict)






#tags_df = pd.DataFrame(transition_dict, columns = list(tags_list), index=list(tags_list))
#display(tags_df)
