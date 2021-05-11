from nltk.corpus import wordnet as wn
import re
import math
import numpy as np
import pandas as pd
import pickle as pickle
import json
import string
import argparse
from os.path import expanduser
from uer.utils.tokenizer import *
#from nltk.corpus import wordnet_ic
#brown_ic = wordnet_ic.ic('ic-brown.dat')
#semcor_ic = wordnet_ic.ic('ic-semcor.dat')

def get_antonyms(input_lemma):
    antonyms = []
    synsets = []
    for syn in wn.synsets(input_lemma):
        for lemma in syn.lemmas():
            synsets.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return synsets, antonyms


def get_similarity(left_lsent, right_lsent):
    stopWords = {"a", "an", "and", "are", "as", "at", "be", "but", "by",

                 "for", "if", "in", "into", "is", "it",

                 "no", "not", "of", "on", "or", "such",

                 "that", "the", "their", "then", "there", "these",

                 "they", "this", "to", "was", "will", "with", "am", 'cls', 'sep'}

    sim = []
    for i in range(len(left_lsent)):
        word = left_lsent[i]
        tmp = []
        for j in range(len(right_lsent)):
            targ = right_lsent[j]
            a = word.lower().translate(str.maketrans('', '', string.punctuation))
            b = targ.lower().translate(str.maketrans('', '', string.punctuation))
            left_syn, left_ant = get_antonyms(a)
            right_syn, right_ant = get_antonyms(b)
            left = wn.synsets(a)
            right = wn.synsets(b)
            if a not in stopWords and b not in stopWords:
                if left != [] and right != []:
                    if b in left_ant or b in left_syn or a in right_ant or a in right_syn:
                        tmp.append(1.0)
                    else:
                        count1, count2 = 0, 0
                        ScoreList1, ScoreList2 = 0, 0
                        for word1 in left:
                            for word2 in right:
                                score1 = word1.wup_similarity(word2)
                                score2 = word2.wup_similarity(word1)
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


def output(todir, test):

    test_a, test_b, test_similarity = test['text_a'], test['text_b'], test['similarity']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )
    parser.add_argument("--vocab_path", type=str, default='./models/google_uncased_en_vocab.txt')
    args = parser.parse_args()
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)


    test['sim'] = [''] * len(test_a)
    ids = []
    # f = open('./datasets/quora/tmp/dev_um_similarity.txt', 'w', encoding='utf-8')
    for idx in range(len(test_a)):
        text_a = test_a[idx]
        text_b = test_b[idx]
        if type(text_a) == float or type(text_b) == float or text_a == 'N/A' or \
                text_a == 'N\A' or text_b == 'N/A' or text_b == 'N\A':
            print(text_a, text_b)
            test['sim'][idx] = ''
            ids.append(idx)
            continue
        text_a = text_a.lower().translate(str.maketrans('', '', string.punctuation)).split()
        text_b = text_b.lower().translate(str.maketrans('', '', string.punctuation)).split()
        text = ['[CLS]'] + text_a + ['[SEP]'] + text_b + ['[SEP]']

        tmp_sim = np.ones((len(text), len(text)))
        #sim = test_similarity[idx].replace(' 0,', ' 0.05,').replace(' 0]', ' 0.05]')
        #similarity = np.array(eval(sim))
        similarity = np.array(eval(test_similarity[idx]))
        tmp_sim[:len(text_a) + 2, :len(text_b) + 2] = similarity
        tmp_sim[len(text_a) + 2:, len(text_b) + 2:] = similarity.T[1:, 1:]

        tag_a = []
        for i in range(len(text)):
            word = text[i]
            token = args.tokenizer.tokenize(word)
            tag = [i] * len(token)
            tag_a = tag_a + tag

        tag = np.zeros((len(tag_a), len(tag_a)))
        for i in range(len(tag_a)):
            for j in range(len(tag_a)):
                tag[i][j] = tmp_sim[tag_a[i]][tag_a[j]]

        # f.write(str(tag) + '\n')
        tag = tag.tolist()
        test['sim'][idx] = tag


    test.drop(ids, inplace=True)
    data = test.reset_index(drop=True)
    del data['similarity']
    text = data.rename(columns={'sim':'similarity'})

    text.to_csv(todir, sep='\t', index=None)


if __name__ == '__main__':
    task = 'msrp'

    if task=='quora':
        '''test = pd.read_csv('./datasets/quora/train_sim.csv', sep='\t', encoding="utf-8", dtype=str)
        print(test.iloc[16343], test.iloc[257826], test.iloc[302378])
        ids = [16343, 257826, 302378]
        test.drop(ids, inplace=True)
        data = test.reset_index(drop=True)
        print(test.iloc[16343], test.iloc[257825], test.iloc[302376])
        print(len(data['text_b']))
        data.to_csv('./datasets/quora/train_sim.csv', sep = '\t', index=None)'''
        train=pd.read_csv('./datasets/quora/train_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        dev = pd.read_csv('./datasets/quora/dev_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        test = pd.read_csv('./datasets/quora/test_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        output('./datasets/quora/train_sim.csv', train)
        output('./datasets/quora/dev_sim.csv', dev)
        output('./datasets/quora/test_sim.csv', test)
    if task == 'url':
        train = pd.read_csv('./datasets/url/train_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        dev = pd.read_csv('./datasets/url/dev_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        test = pd.read_csv('./datasets/url/test_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        output('./datasets/url/train_sim.csv', train)
        output('./datasets/url/dev_sim.csv', dev)
        output('./datasets/url/test_sim.csv', test)

    if task == 'msrp':
        train = pd.read_csv('./datasets/msrp/train_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        dev = pd.read_csv('./datasets/msrp/dev_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)
        test = pd.read_csv('./datasets/msrp/test_similarity.tsv', sep='\t', encoding="utf-8", dtype=str)

        
        output('./datasets/msrp/dev_sim.tsv', dev)
        output('./datasets/msrp/test_sim.tsv', test)
        output('./datasets/msrp/train_sim.tsv', train)




