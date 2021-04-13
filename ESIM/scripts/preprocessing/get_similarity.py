import numpy as np
import pandas as pd
import string
from sklearn.utils import shuffle
from os.path import expanduser
from nltk.corpus import wordnet
import pp

def shuffle_URL_data():
    base_path = '../../data/dataset/url/'
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

    dev = data.loc[:int(num * 0.1)+3]
    dev.reset_index(drop=True, inplace=True)
    dev.to_csv('../../data/dataset/url/dev.tsv', sep='\t', encoding='utf-8')

    test = data.loc[int(num * 0.1) + 4:]
    test.reset_index(drop=True, inplace=True)
    test.to_csv('../../data/dataset/url/train.tsv', sep='\t', encoding='utf-8')

def load_STS_data():
    for task in ['train', 'dev', 'test']:
        f = open('../../data/dataset/sts/original/sts-{}.tsv'.format(task), 'r', encoding='utf-8')
        text_a, text_b, score = [], [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            score.append(line[4])
            text_a.append(line[5])
            text_b.append(line[6])

        test = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
        test["text_a"] = text_a
        test["text_b"] = text_b
        test["labels"] = score
        test.to_csv('../../data/dataset/sts/{}.tsv'.format(task), sep='\t', encoding='utf-8')
        print(len(test))

    return


def load_MRPC_data():
    for task in ['train', 'dev']:
        f = open('../../data/dataset/msrp/MRPC/{}.tsv'.format(task), 'r', encoding='utf-8')
        text_a, text_b, labels = [], [], []
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip().split('\t')
            if i > 0 and len(line)==5:
                text_a.append(line[3])
                text_b.append(line[4])
                labels.append(line[0])

        train = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
        train["text_a"] = text_a
        train["text_b"] = text_b
        train["labels"] = labels
        train.to_csv('../../data/dataset/msrp/{}.tsv'.format(task), sep='\t', index=False)

    f = open('../../data/dataset/msrp/MRPC/msr_paraphrase_test.txt', 'r', encoding='utf-8')
    text_a, text_b, labels = [], [], []
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip().split('\t')
        if i > 0 and len(line) == 5:
            text_a.append(line[3])
            text_b.append(line[4])
            labels.append(line[0])
    train = pd.DataFrame(columns=['text_a', 'text_b', 'labels'])
    train["text_a"] = text_a
    train["text_b"] = text_b
    train["labels"] = labels
    train.to_csv('../../data/dataset/msrp/test.tsv', sep='\t', index=False)


    return



def get_synsets(input_lemma):
    wordnet = nltk.corpus.wordnet
    synsets = []
    for syn in wordnet.synsets(input_lemma):
        for lemma in syn.lemmas():
            synsets.append(lemma.name())

    return synsets


def get_similarity(text_a, text_b, k):
    wordnet = nltk.corpus.wordnet
    left_lsent = ['oov'] + text_a[k].lower().translate(str.maketrans('', '', string.punctuation)).split() + ['oov']
    right_lsent = ['oov'] + text_b[k].lower().translate(str.maketrans('', '', string.punctuation)).split() + ['oov']
    print(k)
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

def data_pre(dataset, task):
    test_data = pd.read_csv('../../data/dataset/{}/{}.tsv'.format(dataset,task), sep='\t')
    text_a, text_b, label, similarity = test_data['text_a'], test_data['text_b'], test_data['labels'], []

    ppservers = ()  # ppservers = ("localhost",)
    job_server = pp.Server(ppservers=ppservers)
    modules = ('nltk.corpus', 'string', )
    jobs = [job_server.submit(get_similarity, (text_a, text_b, i), (get_synsets,), modules=modules ) \
        for i in range(len(label))]

    for job in jobs:
        similarity.append(job())

    # for i in range(len(label)):
    #     left = ['oov'] + text_a[i].lower().translate(str.maketrans('', '', string.punctuation)).split() + ['oov']
    #     right = ['oov'] + text_b[i].lower().translate(str.maketrans('', '', string.punctuation)).split() + ['oov']
    #     sim = get_similarity(left, right)
    #     similarity.append(sim)
    #     print(i)
    data = pd.DataFrame(columns=['text_a', 'text_b', 'labels', 'similarity'])
    data["text_a"] = text_a
    data["text_b"] = text_b
    data["labels"] = label
    data["similarity"] = similarity
    data.to_csv('../../data/dataset/{}/{}_similarity.tsv'.format(dataset, task), sep='\t', encoding='utf-8', index = False)


shuffle_URL_data()
load_MRPC_data()
load_STS_data()

dataset = ['msrp','sts','url']
task = ['train', 'test', 'dev']
for d in dataset:
    for t in task:
        data_pre(d,t)


