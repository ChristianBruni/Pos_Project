import os
import function as fn

def load_data_vit_train():
    path = os.path.dirname(os.path.abspath(__file__))
    path_vit_train = os.path.join(path, "..", "UD_Italian-VIT-master", "it_vit-ud-train.conllu")

    with open(path_vit_train, "r", encoding="utf-8") as f:
        data_vit_train = f.read()
    vit_train, vit_train_tags, vit_train_words = fn.create_mat(data_vit_train)

    return vit_train, vit_train_tags, vit_train_words

def load_data_vit_dev():
    path = os.path.dirname(os.path.abspath(__file__))
    path_vit_dev = os.path.join(path, "..", "UD_Italian-VIT-master", "it_vit-ud-dev.conllu")

    with open(path_vit_dev, "r", encoding="utf-8") as f:
        data_vit_dev = f.read()
    vit_dev, vit_dev_tags, vit_dev_words = fn.create_mat(data_vit_dev)

    return vit_dev, vit_dev_tags, vit_dev_words

def load_data_vit_test():
    path = os.path.dirname(os.path.abspath(__file__))
    path_vit_test = os.path.join(path, "..", "UD_Italian-VIT-master", "it_vit-ud-test.conllu")

    with open(path_vit_test, "r", encoding="utf-8") as f:
        data_vit_test = f.read()
    vit_test, vit_test_tags, vit_test_words = fn.create_mat(data_vit_test)

    return vit_test, vit_test_tags, vit_test_words

def load_data_old_train():
    path = os.path.dirname(os.path.abspath(__file__))
    path_old_train = os.path.join(path, "..", "UD_Italian-OLD-master", "it_old-ud-train.conllu")

    with open(path_old_train, "r", encoding="utf-8") as f:
        data_old_train = f.read()
    old_train, old_train_tags, old_train_words = fn.create_mat(data_old_train)

    return old_train, old_train_tags, old_train_words

def load_data_old_dev():
    path = os.path.dirname(os.path.abspath(__file__))
    path_old_dev = os.path.join(path, "..", "UD_Italian-OLD-master", "it_old-ud-dev.conllu")

    with open(path_old_dev, "r", encoding="utf-8") as f:
        data_old_dev = f.read()
    old_dev, old_train_tags, old_train_words = fn.create_mat(data_old_dev)

    return old_dev, old_train_tags, old_train_words

def load_data_old_test():
    path = os.path.dirname(os.path.abspath(__file__))
    path_old_test = os.path.join(path, "..", "UD_Italian-OLD-master", "it_old-ud-test.conllu")

    with open(path_old_test, "r", encoding="utf-8") as f:
        data_old_test = f.read()
    old_test, old_test_tags, old_test_words = fn.create_mat(data_old_test)

    return old_test, old_test_tags, old_test_words