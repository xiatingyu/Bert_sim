import numpy as np
import pandas as pd
import string
from os.path import expanduser
from nltk.corpus import wordnet as wn



def load_data():
    df = pd.read_csv('../../data/dataset/quora/quora_duplicate_questions.tsv', sep='\t')
    data = df.loc[:, ['question1', 'question2','is_duplicate']]
    num = int(len(data))
    dev_num = int(num * 0.05)
    print(dev_num)


    dev = data.loc[:dev_num]
    dev.to_csv('../../data/dataset/quora/dev.tsv', sep='\t', encoding='utf-8')
    print(len(dev))

    test = data.loc[dev_num+1:dev_num*2+1]
    test.to_csv('../../data/dataset/quora/test.tsv', sep='\t', encoding='utf-8')
    print(len(test))

    train = data.loc[dev_num*2+2:]
    train.to_csv('../../data/dataset/quora/train.tsv', sep='\t', encoding='utf-8')
    print(len(train))

    return


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
            left_syn = get_synsets(a)
            right_syn = get_synsets(b)
            left = wn.synsets(a)
            right = wn.synsets(b)
            if a != 'oov' and b != 'oov':
                if left != [] and right != []:
                    if b in left_syn or a in right_syn:
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

def data_pre():
    dev_data = pd.read_csv('../../data/dataset/quora/dev.tsv', sep='\t')
    text_a, text_b, label, similarity = dev_data['question1'], dev_data['question2'], dev_data['is_duplicate'], []
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
    dev.to_csv('../../data/dataset/quora/dev_similarity.tsv', sep='\t', encoding='utf-8')



    test_data = pd.read_csv('../../data/dataset/quora/test.tsv', sep='\t')
    text_a, text_b, label, similarity = test_data['question1'], test_data['question2'], test_data['is_duplicate'], []
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
    test.to_csv('../../data/dataset/quora/test_similarity.tsv', sep='\t', encoding='utf-8')



    train_data = pd.read_csv('../../data/dataset/quora/train.tsv', sep='\t')
    text_a, text_b, label, similarity = train_data['question1'], train_data['question2'], train_data['is_duplicate'], []
    ids = []
    for i in range(len(label)):
        if type(text_a[i]) == float or type(text_b[i]) == float:
            similarity.append(['null'])
            ids.append(i)
            continue
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
    print(ids)
    train.drop(ids, inplace=True)
    data = train.reset_index(drop=True)
    data.to_csv('../../data/dataset/quora/train_similarity.csv', sep='\t', encoding='utf-8')

data_pre()