import function as fn
import read_file as rf
import memm as mt
import numpy as np

# Lettura dei file
vit_train,  vit_train_tags, vit_train_words = rf.load_data_vit_train()
vit_dev,    vit_dev_tags,   vit_dev_words   = rf.load_data_vit_dev()
vit_test,   vit_test_tags,  vit_test_words  = rf.load_data_vit_test()
old_train,  old_train_tags, old_train_words = rf.load_data_old_train()
old_dev,    old_dev_tags,   old_dev_words   = rf.load_data_old_dev()
old_test,   old_test_tags,  old_test_words  = rf.load_data_old_test()

#-----------------------------------------------------------------------------------------------------------------------

print("Scegli il tipo di smoothing utilizzare:")
print("1. Probabilità di NOUN a 1")
print("2. Probabilità di NOUN e VERB a 0.5")
print("3. Distribuzione del development set")
print("4. Distribuzione uniforme")
print("5. Suffissi e capitalizzazione")

scelta = input("\nInserisci il numero dell'opzione: ")

match scelta:


    case "1":
        print("Smoothing NOUN")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_n(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_n(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_n(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_n(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_n(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0),  np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        #print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "2":
        print("Smoothing NOUN + VERB")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_nv(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_nv(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_nv(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_nv(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_nv(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        #print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "3":
        print("Smoothing con development")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_dev(vit_train, vit_train_tags, vit_test_tags, vit_test_words, vit_dev))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_dev(old_train, old_train_tags, old_test_tags, old_test_words, old_dev))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_dev(vit_train, vit_train_tags, old_test_tags, old_test_words, vit_dev))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_dev(old_train, old_train_tags, vit_test_tags, vit_test_words, old_dev))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_dev(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_dev, old_dev), axis=0)))
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        #print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "4":
        print("Smoothing con distribuzione uniforme")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_uniform(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_uniform(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_uniform(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_uniform(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_uniform(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        #print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "5":
        print("Smoothing con suffissi e capitalizzazione")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_syntax(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_syntax(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_syntax(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        #print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_syntax(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        #print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_syntax(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        #print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")