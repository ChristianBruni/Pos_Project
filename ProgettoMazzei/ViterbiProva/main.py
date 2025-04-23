import function as fn
import read_file as rf
import memm as mt
import numpy as np

#letture dei file
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
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_nv(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_nv(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_nv(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_nv(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "3":
        print("Smoothing con development")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_dev(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_dev(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_dev(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_dev(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_dev(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "4":
        print("Smoothing con distribuzione uniforme")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_uniform(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_uniform(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_uniform(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_uniform(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_uniform(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")

    case "5":
        print("Smoothing con suffissi e capitalizzazione")
        print("Training set: vit_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_syntax(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: old_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_syntax(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: vit_train  -  Test set: old_test")
        print("Accuracy:", fn.tagging_syntax(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))
        print("Baseline MEMM:", mt.run_process(vit_train_words, vit_train_tags, old_test_words, old_test_tags), "\n")

        print("Training set: old_train  -  Test set: vit_test")
        print("Accuracy:", fn.tagging_syntax(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))
        print("Baseline MEMM:", mt.run_process(old_train_words, old_train_tags, vit_test_words, vit_test_tags), "\n")

        print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
        print("Accuracy:", fn.tagging_syntax(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)), "\n")
        print("Baseline semplice:", fn.evaluate_baseline(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)))
        print("Baseline MEMM:", mt.run_process(np.concatenate((vit_train_words, old_train_words), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0)), "\n")





















#print("Smoothing NOUN + VERB")
#print("Training set: vit_train  -  Test set: vit_test")
#print("Accuracy:", fn.tagging_vn(vit_train, vit_train_tags, vit_test_tags, vit_test_words), "\n")
#
#print("Training set: old_train  -  Test set: old_test")
#print("Accuracy:", fn.tagging_vn(old_train, old_train_tags, old_test_tags, old_test_words), "\n")
#
#print("Training set: vit_train  -  Test set: old_test")
#print("Accuracy:", fn.tagging_vn(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")
#
#print("Training set: old_train  -  Test set: vit_test")
#print("Accuracy:", fn.tagging_vn(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")
#
#print("Training set: vit_train + old_train  -  Test set: vit_test + old_test")
#print("Accuracy:", fn.tagging_vn(np.concatenate((vit_train, old_train), axis=0), np.concatenate((vit_train_tags, old_train_tags), axis=0), np.concatenate((vit_test_tags, old_test_tags), axis=0), np.concatenate((vit_test_words, old_test_words), axis=0)), "\n")
#
#print("Baseline semplice")
#print("Accuracy:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))
#
#print("Baseline MEMM")
#print("Accuracy:", mt.run_process(old_train_words, old_train_tags, old_test_words,old_test_tags ))
#
#
#
#
#
#
#
#
#
#
#
#
#
#


# Training set: old_train
# Test set: old_test
#print("Training set: old_train \nTest set: old_test")
#print("Accuracy:", fn.tagging(old_train, old_train_tags, old_test_tags, old_test_words), "\n")
#print("Accuracy dev:", fn.tagging_with_develop(old_train, old_train_tags, old_test_tags, old_test_words, old_dev), "\n")
#print("Accuracy baseline easy:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))
#print("Accuracy baseline difficile:", mt.run_process(old_train_words, old_train_tags, old_test_words,old_test_tags ))


# 2) Insieme

# Training set: old_train + vit_train
# Test set: old_test + vit_test
#print("Training set: old_train + vit_train \nTest set: old_test + vit_test")
#train = np.concatenate((vit_train, old_train), axis=0)
#dev = np.concatenate((vit_dev, old_dev), axis=0)
#train_tags = np.concatenate((vit_train_tags, old_train_tags), axis=0)
#test_tags = np.concatenate((vit_test_tags, old_test_tags), axis=0)
#test_words = np.concatenate((vit_test_words, old_test_words), axis=0)
#print("Accuracy:", fn.tagging(train, train_tags, test_tags, test_words), "\n")
#print("Accuracy dev:", fn.tagging_with_develop(train, train_tags, test_tags, test_words, dev), "\n")
#print("Accuracy baseline easy:", fn.evaluate_baseline(train, train_tags, test_tags, test_words))

# 3) OUT of DOMAIN

# Training set: vit_train
# Test set: old_test
#print("Training set: vit_train \nTest set: old_test")
#print("Accuracy:", fn.tagging(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")
#print("Accuracy:", fn.tagging_with_develop(vit_train, vit_train_tags, old_test_tags, old_test_words, vit_dev), "\n")
#print("Accuracy baseline easy:", fn.evaluate_baseline(vit_train, vit_train_tags, old_test_tags, old_test_words))

# Training set: old_train
# Test set: vit_test
#print("Training set: old_train \nTest set: vit_test")
#print("Accuracy:", fn.tagging(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")
#print("Accuracy:", fn.tagging_with_develop(old_train, old_train_tags, vit_test_tags, vit_test_words, old_dev), "\n")
#print("Accuracy baseline easy:", fn.evaluate_baseline(old_train, old_train_tags, vit_test_tags, vit_test_words))


