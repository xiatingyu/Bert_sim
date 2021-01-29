import numpy as np
import pandas as pd
import string
from os.path import expanduser
from nltk.corpus import wordnet as wn
from sklearn.utils import shuffle


def get_synsets(input_lemma):
    synsets = []
    for syn in wn.synsets(input_lemma):
        for lemma in syn.lemmas():
            synsets.append(lemma.name())

    return synsets

def get_similarity(left_lsent, right_lsent):
    sim = []
    for i in range(len(left_lsent)):
        word = left_lsent[i]
        tmp = []
        for j in range(len(right_lsent)):
            targ = right_lsent[j]
            a = word.lower().translate(str.maketrans('', '', string.punctuation))
            b = targ.lower().translate(str.maketrans('', '', string.punctuation))
            left_syn, left_ant = get_synsets(a)
            right_syn, right_ant = get_synsets(b)
            left = wn.synsets(a)
            right = wn.synsets(b)
            if a != 'oov' and b != 'oov':
                if left != [] and right != []:
                    if b in left_syn or a in right_syn or b in left_ant or a in right_ant:
                        tmp.append(1.0)
                    else:
                        count1, count2= 0, 0
                        ScoreList1, ScoreList2 = 0, 0
                        for word1 in left:
                            for word2 in right:
                                try:
                                    score1 = word1.wup_similarity(word2)
                                except:
                                    score1 = 0.0
                                try:
                                    score2 = word2.wup_similarity(word1)
                                except:
                                    score2 = 0.0
                                #score1 = word1.stop_similarity(word2)
                                if score1 is not None:
                                    ScoreList1 += score1
                                    count1 += 1
                                if score2 is not None:
                                    ScoreList2 += score2
                                    count2 += 1

                        if count1 + count2 != 0:
                            similarity = (ScoreList1 + ScoreList2)/(count1 + count2)
                            tmp.append(similarity)
                        else:
                            if word.lower() == targ.lower():
                                tmp.append(1)
                            else:
                                tmp.append(0)
                else:
                    if word.lower() == targ.lower():
                        tmp.append(1)
                    else:
                        tmp.append(0)
            else:
                tmp.append(0)

        sim.append(tmp)

    return sim



def load_data():
    test_data = pd.read_csv('../../data/dataset/twitterURL/test.tsv', sep='\t')
    text_a, text_b, label, similarity = test_data['text_a'], test_data['text_b'], test_data['labels'], []
    for i in range(len(label)):
        left = ['oov'] + text_a[i].split() + ['oov']
        right = ['oov'] + text_b[i].split() + ['oov']
        sim = get_similarity(left, right)
        similarity.append(sim)
        print(i)
    test = pd.DataFrame(columns=['text_a', 'text_b', 'labels', 'similarity'])
    test["text_a"] = text_a
    test["text_b"] = text_b
    test["labels"] = label
    test["similarity"] = similarity
    test.to_csv('../../data/dataset/twitterURL/test_similarit.tsv', sep='\t', encoding='utf-8')

    dev_data = pd.read_csv('../../data/dataset/twitterURL/dev.tsv', sep='\t')
    text_a, text_b, label, similarity = dev_data['text_a'], dev_data['text_b'], dev_data['labels'], []
    for i in range(len(label)):
        left = ['oov'] + text_a[i].split() + ['oov']
        right = ['oov'] + text_b[i].split() + ['oov']
        sim = get_similarity(left, right)
        similarity.append(sim)
        print(i)
    dev = pd.DataFrame(columns=['text_a', 'text_b', 'labels', 'similarity'])
    dev["text_a"] = text_a
    dev["text_b"] = text_b
    dev["labels"] = label
    dev["similarity"] = similarity
    dev.to_csv('../../data/dataset/twitterURL/dev_similarity.tsv', sep='\t', encoding='utf-8')

    train_data = pd.read_csv('../../data/dataset/twitterURL/train.tsv', sep='\t')
    text_a, text_b, label, similarity = train_data['text_a'], train_data['text_b'], train_data['labels'], []
    for i in range(len(label)):
        left = ['oov'] + text_a[i].split() + ['oov']
        right = ['oov'] + text_b[i].split() + ['oov']
        sim = get_similarity(left, right)
        similarity.append(sim)
        print(i)
    train = pd.DataFrame(columns=['text_a', 'text_b', 'labels', 'similarity'])
    train["text_a"] = text_a
    train["text_b"] = text_b
    train["labels"] = label
    train["similarity"] = similarity
    train.to_csv('../../data/dataset/twitterURL/train_similarity.csv', sep='\t', encoding='utf-8')


def shuffle_data():
    base_path = '../../data/dataset/twitterURL/'
    text_a, text_b, label, similarity = [], [], [], []


    f = open(base_path + 'Twitter_URL_Corpus_test.txt', 'r', encoding='UTF-8')
    for line in f.readlines():
        line = line.strip().split('\t')
        if int(line[2][1]) >= 4:
           label.append(1)
           text_a.append(line[0])
           text_b.append(line[1])
        elif int(line[2][1]) <= 2:
           label.append(0)
           text_a.append(line[0])
           text_b.append(line[1])

    test = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
    test["text_a"] = text_a
    test["text_b"] = text_b
    test["labels"] = label
    test.to_csv(base_path + 'test.tsv', sep='\t', encoding='utf-8')

    text_a, text_b, label, similarity = [], [], [], []
    f = open(base_path + 'Twitter_URL_Corpus_train.txt', 'r', encoding='UTF-8')
    for line in f.readlines():
        line = line.strip().split('\t')
        if int(line[2][1]) >= 4:
           label.append(1)
           text_a.append(line[0])
           text_b.append(line[1])
        elif int(line[2][1]) <= 2:
           label.append(0)
           text_a.append(line[0])
           text_b.append(line[1])

    train = pd.DataFrame(columns = ['text_a','text_b','labels', 'similarity'])
    train["text_a"] = text_a
    train["text_b"] = text_b
    train["labels"] = label

    num = len(train)
    data = shuffle(train)
    data.reset_index(drop=True, inplace=True)

    dev = data.loc[:int(num * 0.1)]
    dev.reset_index(drop=True, inplace=True)
    dev.to_csv('../../data/dataset/twitterURL/dev.tsv', sep='\t', encoding='utf-8')

    test = data.loc[int(num * 0.1) + 1:]
    test.reset_index(drop=True, inplace=True)
    test.to_csv('../../data/dataset/twitterURL/train.tsv', sep='\t', encoding='utf-8')


load_data()

#