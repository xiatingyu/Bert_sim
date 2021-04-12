import numpy as np
import pandas as pd
import string
from sklearn.utils import shuffle
from os.path import expanduser
from nltk.corpus import wordnet
import pp


def load_Quora_data():
    for task in ['train', 'dev', 'test']:
        f = open('../../data/dataset/quora/Quora_question_pair_partition/{}.tsv'.format(task), 'r', encoding='utf-8')
        text_a, text_b, labels = [], [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            text_a.append(line[1])
            text_b.append(line[2])
            labels.append(line[0])

        train = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
        train["text_a"] = text_a
        train["text_b"] = text_b
        train["labels"] = labels
        train.to_csv('../../data/dataset/quora/{}.tsv'.format(task), sep='\t', index=False)

    return


def get_synsets(input_lemma):
    synsets = []
    for syn in wordnet.synsets(input_lemma):
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
            left_syn = get_synsets(word)
            right_syn = get_synsets(targ)
            left = wordnet.synsets(word)
            right = wordnet.synsets(targ)
            if word != 'oov' and targ != 'oov':
                if left != [] and right != []:
                    if targ in left_syn or word in right_syn:
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
                            if word == targ:
                                tmp.append(1)
                            else:
                                tmp.append(0)
                else:
                    if word == targ:
                        tmp.append(1)
                    else:
                        tmp.append(0)
            else:
                tmp.append(0)

        sim.append(tmp)

    return sim


def data_pre_quora(task):
    test_data = pd.read_csv('../../data/dataset/quora/{}.tsv'.format(task), sep='\t')
    test_data = test_data.fillna('none')
    text_a, text_b, label, similarity = test_data['text_a'], test_data['text_b'], test_data['labels'], []

    #this dataset use pp will cause memory error
    for i in range(len(label)):
        left = ['oov'] + text_a[i].lower().translate(str.maketrans('', '', string.punctuation)).split() + ['oov']
        right = ['oov'] + text_b[i].lower().translate(str.maketrans('', '', string.punctuation)).split() + ['oov']
        sim = get_similarity(left, right)
        similarity.append(sim)
        print(i)
    data = pd.DataFrame(columns=['text_a', 'text_b', 'labels', 'similarity'])
    data["text_a"] = text_a
    data["text_b"] = text_b
    data["labels"] = label
    data["similarity"] = similarity
    data.to_csv('../../data/dataset/quora/{}_similarity.tsv'.format(task), sep='\t', encoding='utf-8', index = False)


#load_Quora_data()

task = ['train']
for t in task:
    data_pre_quora(t)

