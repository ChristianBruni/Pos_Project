import function as fn
import read_file as rf
import pandas as pd

from ProgettoMazzei.Prova.function import baseline_tagger

#letture dei file
vit_train,  vit_train_tags, vit_train_words = rf.load_data_vit_train()
vit_dev,    vit_dev_tags,   vit_dev_words   = rf.load_data_vit_dev()
vit_test,   vit_test_tags,  vit_test_words  = rf.load_data_vit_test()
old_train,  old_train_tags, old_train_words = rf.load_data_old_train()
old_dev,    old_dev_tags,   old_dev_words   = rf.load_data_old_dev()
old_test,   old_test_tags,  old_test_words  = rf.load_data_old_test()

#-----------------------------------------------------------------------------------------------------------------------

# Training set: vit_train
# Test set: vit_dev
#print("Training set: vit_train \nTest set: vit_dev")
#print("Accuracy:", fn.tagging(vit_train, vit_train_tags, vit_dev_tags, vit_dev_words), "\n")
print("Accuracy baseline easy:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_dev_tags, vit_dev_words))


# Training set: vit_train
# Test set: vit_test
#print("Training set: vit_train \nTest set: vit_test")
#print("Accuracy:", fn.tagging(vit_train, vit_train_tags, vit_test_tags, vit_test_words), "\n")
#print("Accuracy baseline easy:", fn.evaluate_baseline(vit_train, vit_train_tags, vit_test_tags, vit_test_words))


# Training set: old_train
# Test set: old_dev
#print("Training set: old_train \nTest set: old_dev")
#print("Accuracy:", fn.tagging(old_train, old_train_tags, old_dev_tags, old_dev_words), "\n")



# Training set: old_train
# Test set: old_test
#print("Training set: old_train \nTest set: old_test")
#print("Accuracy:", fn.tagging(old_train, old_train_tags, old_test_tags, old_test_words), "\n")
#print("Accuracy baseline easy:", fn.evaluate_baseline(old_train, old_train_tags, old_test_tags, old_test_words))


# Training set: vit_train
# Test set: old_test
#print("Training set: vit_train \nTest set: old_test")
#print("Accuracy:", fn.tagging(vit_train, vit_train_tags, old_test_tags, old_test_words), "\n")



# Training set: old_train
# Test set: vit_test
#print("Training set: old_train \nTest set: vit_test")
#print("Accuracy:", fn.tagging(old_train, old_train_tags, vit_test_tags, vit_test_words), "\n")




#aggiungere i controlli per quando non ci sono le parole
#modificare come viene calcolata la emission table
#provare ad allenare il modello con un dataset diverso
#modificare le probabilità con i logaritmi

