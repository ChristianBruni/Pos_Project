import function as fn
import read_file as rf
import pandas as pd






#letture dei file
vit_train,  vit_train_tags, vit_train_words = rf.load_data_vit_train()
vit_dev,    vit_dev_tags,   vit_dev_words   = rf.load_data_vit_dev()
vit_test,   vit_test_tags,  vit_test_words  = rf.load_data_vit_test()
#old_train,  old_train_tags, old_train_words = rf.load_data_old_train()
#old_dev,    old_train_dev,  old_train_dev   = rf.load_data_old_dev()
#old_test,   old_test_tags,  old_test_words  = rf.load_data_old_test()


print(len(vit_train))
print(vit_train)
#-----------------------------------------------------------------------------------------------------------------------

# conta i tag
count_tags = fn.count_tags(vit_train)
tags_list = count_tags[:, 0].tolist()
tags_list.append('S0')  # aggiunge S0

# conta le parole
count_words = fn.count_words(vit_train)
words_list = count_words[:, 0].tolist()

# calcola le probabilità di emissione
emission_matrix = fn.emission_matrix(vit_train, count_tags)
#print(emission_matrix.shape)
#print(emission_matrix)
#display(pd.DataFrame(emission_matrix, columns = list(tags_list), index = words_list))

# calcola le probabilità di transizione
transition_matrix = fn.transition_matrix(vit_train_tags, count_tags)
#display(pd.DataFrame(transition_matrix, columns = list(tags_list), index=list(tags_list)))

start_tag = fn.start_tag(transition_matrix)  # dimensione 17
#print(start_tag)

predicted_tags = []
for sentence in vit_dev_words:
    tagged_seq = fn.viterbi(sentence.split(), vit_train, transition_matrix, emission_matrix)
    predicted_tags.append(tagged_seq)

accuracy = fn.calc_accuracy(predicted_tags, vit_dev_tags)
print("Accuracy:", accuracy)




#aggiungere i controlli per quando non ci sono le parole
#modificare come viene calcolata la emission table
#provare ad allenare il modello con un dataset diverso
#modificare le probabilità con i logaritmi
