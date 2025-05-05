import math

from fontTools.misc.bezierTools import epsilon


# Algoritmo di viterbi con smoothing NOUN
def viterbi_n(sentence, tags, start_p, tr, em):

    epsilon = 1e-10

    # Inizializzazione delle strutture dati
    viterbi = {state: [0.0] * len(sentence) for state in tags}
    backpointer = {state: [None] * len(sentence) for state in tags}

    # Inizializzazione (primo passo)
    for state in tags:
        word = sentence[0]

        if word not in em[state]:
            emit_prob = em[state].get(word, 1 if state in 'NOUN' else 0.0)
        else:
            emit_prob = em[state][word]

        emit_prob = max(emit_prob, epsilon)
        start_prob = max(start_p.get(state, 0.0), epsilon)
        viterbi[state][0] = (- math.log(start_prob)) + (- math.log(emit_prob))
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(sentence)):
        word = sentence[t]

        for current_state in tags:
            max_prob = float('inf')
            best_prev_state = None

            if word not in em[current_state]:
                emit_prob = em[current_state].get(word, 1 if current_state in 'NOUN' else 0.0)
            else:
                emit_prob = em[current_state][word]

            emit_prob = max(emit_prob, epsilon)

            for prev_state in tags:
                trans_prob = tr[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) + (- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state

    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(tags, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]

    for t in range(len(sentence)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(sentence, best_path))




# Algoritmo di viterbi con smoothing NOUN + VERB
def viterbi_vn(sentence, tags, start_p, tr, em):

    epsilon  = 1e-10

    # Inizializzazione delle strutture dati
    viterbi = {state: [float('inf')] * len(sentence) for state in tags}
    backpointer = {state: [None] * len(sentence) for state in tags}

    # Inizializzazione (primo passo)
    for state in tags:
        word = sentence[0]

        if word not in em[state]:
            emit_prob = em[state].get(word, 0.5 if state in ['NOUN', 'VERB'] else epsilon)
        else:
            emit_prob = em[state][word]

        emit_prob = max(emit_prob,epsilon)
        start_prob = start_p.get(state, 0.0)
        start_prob = max(start_prob, epsilon)
        viterbi[state][0] = (- math.log(start_prob)) + (- math.log(emit_prob))
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(sentence)):
        word = sentence[t]

        for current_state in tags:
            max_prob = float('inf')
            best_prev_state = None

            if word not in em[current_state]:
                emit_prob = em[current_state].get(word, 0.5 if current_state in ['NOUN', 'VERB'] else epsilon)
            else:
                emit_prob = em[current_state][word]
            emit_prob= max(emit_prob,epsilon)

            for prev_state in tags:
                trans_prob = tr[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) + (- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state

    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(tags, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]

    for t in range(len(sentence)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(sentence, best_path))




# Algoritmo di viterbi con smoothing development set
def viterbi_dev(sentence, tags, start_p, tr, em, dev_dist):

    epsilon = 1e-10

    # Inizializzazione delle strutture dati
    viterbi = {state: [float('inf')] * len(sentence) for state in tags}
    backpointer = {state: [None] * len(sentence) for state in tags}

    # Inizializzazione (primo passo)
    for state in tags:
        word = sentence[0]
        if word not in em[state]:
            emit_prob = dev_dist[state]
        else:
            emit_prob = em[state][word]

        emit_prob= max(emit_prob,epsilon)
        start_prob = start_p.get(state, 0.0)
        start_prob = max(start_prob, epsilon)
        viterbi[state][0] = (- math.log(start_prob)) + (- math.log(emit_prob))
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(sentence)):
        word = sentence[t]

        for current_state in tags:
            max_prob = float('inf')
            best_prev_state = None

            if word not in em[current_state]:
                emit_prob = dev_dist[current_state]
            else:
                emit_prob = em[current_state][word]
            emit_prob= max(emit_prob,epsilon)

            for prev_state in tags:
                trans_prob = tr[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) + (- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state

    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(tags, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]

    for t in range(len(sentence)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(sentence, best_path))




# Algoritmo di viterbi con smoothing uniform distribution
def viterbi_uniform(sentence, tags, start_p, tr, em):

    epsilon = 1e-10

    # Inizializzazione delle strutture dati
    viterbi = {state: [float('inf')] * len(sentence) for state in tags}
    backpointer = {state: [None] * len(sentence) for state in tags}

    # Inizializzazione (primo passo)
    for state in tags:
        word = sentence[0]

        if word not in em[state]:
            emit_prob = 1/len(tags)
        else:
            emit_prob = em[state][word]

        emit_prob = max(emit_prob, epsilon)
        start_prob = start_p.get(state, 0.0)

        start_prob = max(start_prob, epsilon)
        viterbi[state][0] = math.log(start_prob) + math.log(emit_prob)
        backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(sentence)):
        word = sentence[t]

        for current_state in tags:
            max_prob = float('inf')
            best_prev_state = None

            if word not in em[current_state]:
                emit_prob = 1/len(tags)
            else:
                emit_prob = em[current_state][word]

            emit_prob = max(emit_prob, epsilon)

            for prev_state in tags:
                trans_prob = tr[prev_state].get(current_state, 0)
                trans_prob = max(trans_prob, epsilon)
                prob = viterbi[prev_state][t-1] + (- math.log(trans_prob)) +(- math.log(emit_prob))

                if prob < max_prob:
                    max_prob = prob
                    best_prev_state = prev_state
            viterbi[current_state][t] = max_prob
            backpointer[current_state][t] = best_prev_state
    # Terminazione (trova lo stato finale migliore)
    best_last_state = min(tags, key=lambda s: viterbi[s][-1])

    # Ricostruzione del percorso all'indietro
    best_path = [best_last_state]

    for t in range(len(sentence)-1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(sentence, best_path))





def check_syntax(word, tag):


    #dict = {
    #    'ADJ': [
    #        'oso', 'osa', 'ile', 'ente', 'ante', 'ivo', 'iva', 'esco', 'esca', 'ario', 'aria', 'aceo', 'acea', 'ico', 'ica', 'ino', 'ina'
    #    ],
    #    'ADP': [
    #        'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra'
    #    ],
    #    'ADV': [
    #        'mente', 'qui', 'lì', 'là', 'via', 'giù', 'sù', 'ora', 'sempre', 'spesso', 'mai'
    #    ],
    #    'AUX': [
    #        'sti', 'rà', 'rò', 'rebbe', 'fui', 'sarà', 'sia'
    #    ],
    #    'CCONJ': [
    #        'ma', 'oppure', 'però', 'bensì', 'infatti', 'cioè'
    #    ],
    #    'DET': [
    #        'il', 'lo', 'la', 'gli', 'le', 'un', 'una', 'uno', 'questo', 'quella', 'quel', 'questi'
    #    ],
    #    'INTJ': [
    #        'oh', 'eh', 'ah', 'uff', 'bah', 'mah', 'wow', 'oops', 'accidenti', 'evviva'
    #    ],
    #    'NOUN': [
    #        'zione', 'tà', 'mento', 'ore', 'ista', 'tore', 'trice', 'aggio', 'ezza', 'ismo', 'icità', 'ità', 'anza', 'enza', 'itudine', 'logia'
    #    ],
    #    'NUM': [
    #        'uno', 'due', 'tre', 'quattro', 'cinque', 'sei', 'sette', 'otto', 'nove', 'dieci', 'cento', 'mille', 'milione', 'miliardo'
    #    ],
    #    'PART': [
    #        'ci', 'mi', 'ti', 'vi', 'si'
    #    ],
    #    'PRON': [
    #        'noi', 'voi', 'loro', 'egli', 'ella', 'esso'
    #    ],
    #    'PROPN': [
    #        'ini', 'etti', 'one', 'oni', 'ani', 'elli', 'ardo'
    #    ],
    #    'PUNCT': [
    #        '.', ',', ';', ':', '!', '?', '"', '«', '»', '…', '(', ')'
    #    ],
    #    'SCONJ': [
    #        'che', 'perché', 'quando', 'mentre', 'sebbene', 'poiché', 'se', 'nonostante', 'finché', 'dopo che'
    #    ],
    #    'VERB': [
    #        'are', 'ere', 'ire', 'ando', 'endo', 'ato', 'uto', 'ito', 'erei', 'erà', 'avo', 'ono', 'ano', 'iamo'
    #    ],
    #    'X': [
    #        'xxx', '???', '###', '!!!', 'null'
    #    ]
    #}

    dict = {
        'ADJ': [
            'ale', 'ambulo', 'ario', 'bile', 'errimo', 'ese', 'evole', 'frago', 'fugo',
            'ico', 'issimo', 'oso', 'ota', 'oto', 'vago', 'tivo', 'torio'
        ],
        'ADP': [
            'di', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra'
        ],
        'ADV': [
            'mente', 'oni'
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
            'accio', 'aggine', 'aggio', 'aglia', 'aio', 'ame', 'ano', 'anza', 'asco', 'astro',
            'ata', 'ato', 'azzo', 'enza', 'eria', 'esco', 'ese', 'età', 'etto', 'ezza', 'fero',
            'icciolo', 'iere', 'ino', 'iolo', 'ismo', 'ista', 'ità', 'legio', 'mento', 'oide',
            'one', 'osi', 'ota', 'oto', 'otto', 'tore', 'tura', 'uccio', 'ucolo', 'ume', 'uto',
            'uzzo', 'zione'
        ],
        'NUM': [
            'uno', 'due', 'tre', 'quattro', 'cinque', 'sei', 'sette', 'otto', 'nove', 'dieci', 'cento', 'mille', 'milione', 'miliardo'
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
            'are', 'arsi', 'ere', 'ersi', 'arre', 'orre', 'orsi', 'urre', 'ursi', 'ire', 'irsi'
        ],
         'X': [
            'xxx', '???', '###', '!!!', 'null'
        ]
    }

    suffissi = dict.get(tag, [])

    if word[0].isupper():
        word = word.lower()
        if word in suffissi:
                return 1
        elif any(word.endswith(suf) for suf in suffissi):
            return 1

        return 1 if tag == 'PROPN' else 0

    word = word.lower()
    suffissi = dict.get(tag, [])

    if word in suffissi:
        return 1
    elif any(word.endswith(suf) for suf in suffissi):
        return 1

    return 0.5 if tag in ['NOUN', 'VERB'] else 0


# Algoritmo di viterbi con smoothing basato sulla sintassi
def viterbi_sintax(sentence, tags, start_p, tr, em):

    epsilon = 1e-10

    # Inizializzazione delle strutture dati
    viterbi = {state: [float('inf')] * len(sentence) for state in tags}
    backpointer = {state: [None] * len(sentence) for state in tags}

    # Inizializzazione (primo passo)
    arr_emit = []

    for state in tags:
        word = sentence[0]
        if word not in em[state]:
            emit_prob = check_syntax(word, state)
        else:
            emit_prob = em[state][word]
        arr_emit.append(max(emit_prob, epsilon))

    sum_arr_emit = sum(arr_emit)



    if sum_arr_emit > 1:
        for emit, state in zip(arr_emit, tags):
            viterbi[state][0] = -math.log(max(start_p.get(state, 0), epsilon)) + (-math.log((emit / sum_arr_emit)))
            backpointer[state][0] = None
    else:
        for emit, state in zip(arr_emit, tags):
            #viterbi[state][0] = start_p.get(state, 0) * emit
            viterbi[state][0] = -math.log(max(start_p.get(state, 0), epsilon)) + (-math.log(emit))
            backpointer[state][0] = None

    # Ricorsione (passi successivi)
    for t in range(1, len(sentence)):
        word = sentence[t]
        arr_emit2 = []

        for current_state in tags:
            if word not in em[current_state]:
                emit_prob = check_syntax(word, current_state)
            else:
                emit_prob = em[current_state][word]
            arr_emit2.append(max(emit_prob, epsilon))

        sum_arr_emit = sum(arr_emit2)

        if sum_arr_emit > 1:

            for emit, current_state in zip(arr_emit2, tags):
                max_prob = float('inf')
                best_prev_state = None

                for prev_state in tags:
                    trans_prob = tr[prev_state].get(current_state, 0)
                    prob = viterbi[prev_state][t-1] + (-math.log(max(trans_prob, epsilon))) + (-math.log((emit / sum_arr_emit)))

                    if prob < max_prob:
                        max_prob = prob
                        best_prev_state = prev_state

                viterbi[current_state][t] = max_prob
                backpointer[current_state][t] = best_prev_state
        else:

            for emit, current_state in zip(arr_emit2, tags):
                max_prob = float('inf')
                best_prev_state = None

                for prev_state in tags:
                    trans_prob = tr[prev_state].get(current_state, 0)
                    prob = viterbi[prev_state][t-1] + (-math.log(max(trans_prob, epsilon))) + (-math.log(emit))

                    if prob < max_prob:
                        max_prob = prob
                        best_prev_state = prev_state

                viterbi[current_state][t] = max_prob
                backpointer[current_state][t] = best_prev_state

    # Terminazione
    best_last_state = min(tags, key=lambda s: viterbi[s][-1])
    best_path = [best_last_state]

    for t in range(len(sentence) - 1, 0, -1):
        best_last_state = backpointer[best_last_state][t]
        best_path.insert(0, best_last_state)

    return list(zip(sentence, best_path))