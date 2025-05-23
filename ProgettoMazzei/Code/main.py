#screen del codice da fare
#create_mat per far vedeere come parsifichiamo il dataset
#viterbi_n il primo ciclo per far vedere il primo tag
#viterbi_n il secondo ciclo
#viterbi_syntax
#calc_emission_probability
#calc_transition_probability
#tagging_n per var vedere  che calcoliamo le prob e poi facciamo il ciclo su ogni frase


import function as fn
import read_file as rf
import numpy as np

print("Scegli il tipo di smoothing utilizzare:")
print("1. Probabilità di NOUN a 1")
print("2. Probabilità di NOUN e VERB a 0.5")
print("3. Distribuzione del development set")
print("4. Distribuzione uniforme")
print("5. Suffissi e capitalizzazione")

scelta = input("\nInserisci il numero dell'opzione: ")

#-----------------------------------------------------------------------------------------------------------------------

# Lettura dei file
vit_train,  vit_train_tags, vit_train_words = rf.load_data_vit_train()
vit_dev,    vit_dev_tags,   vit_dev_words   = rf.load_data_vit_dev()
vit_test,   vit_test_tags,  vit_test_words  = rf.load_data_vit_test()
old_train,  old_train_tags, old_train_words = rf.load_data_old_train()
old_dev,    old_dev_tags,   old_dev_words   = rf.load_data_old_dev()
old_test,   old_test_tags,  old_test_words  = rf.load_data_old_test()

match scelta:
    case "1":
        print("Smoothing NOUN")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_n(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_n(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, old_test_tags, old_test_words), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_n(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, old_test_tags, old_test_words), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_n(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy Viterbi:", fn.tagging_n(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0),  np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.tagging_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))


    case "2":
        print("Smoothing NOUN + VERB")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_nv(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_nv(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, old_test_tags, old_test_words), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_nv(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, old_test_tags, old_test_words), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_nv(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy Viterbi:", fn.tagging_nv(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.tagging_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))

    case "3":
        print("Smoothing con development")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_dev(vit_train, vit_train_tags, vit_test_tags, vit_test_words, vit_dev))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_dev(old_train, old_train_tags, old_test_tags, old_test_words, old_dev))
        print("Baseline semplice:", fn.tagging_baseline(old_train, old_test_tags, old_test_words), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_dev(vit_train, vit_train_tags, old_test_tags, old_test_words, vit_dev))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, old_test_tags, old_test_words), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_dev(old_train, old_train_tags, vit_test_tags, vit_test_words, old_dev))
        print("Baseline semplice:", fn.tagging_baseline(old_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy Viterbi:", fn.tagging_dev(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_dev, old_dev), axis=0)))
        print("Baseline semplice:", fn.tagging_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))

    case "4":
        print("Smoothing con distribuzione uniforme")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_uniform(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_uniform(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, old_test_tags, old_test_words), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_uniform(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, old_test_tags, old_test_words), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_uniform(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy Viterbi:", fn.tagging_uniform(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.tagging_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))



    case "5":
        print("Smoothing con suffissi e capitalizzazione")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_syntax(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_syntax(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, old_test_tags, old_test_words), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy Viterbi:", fn.tagging_syntax(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.tagging_baseline(vit_train, old_test_tags, old_test_words), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy Viterbi:", fn.tagging_syntax(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.tagging_baseline(old_train, vit_test_tags, vit_test_words), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy Viterbi:", fn.tagging_syntax(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.tagging_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
