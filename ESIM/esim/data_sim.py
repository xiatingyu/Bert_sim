"""
Preprocessor and dataset definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np
import sys
import random
import pandas as pd
import pickle
from sklearn.utils import shuffle
sys.path.insert(0, "../../")
from collections import Counter
from torch.utils.data import Dataset

class Tokenizer():
    def get_elmo_tokenization(self, sentence):
        tokenization = [w for w in sentence.rstrip().split()]
        return tokenization, len(tokenization)

    def get_batched_elmo_tokenization(self, sentences):
        tokenized_text = []
        tokenized_length = []
        for sentence in sentences:
            # sentence[0] to pick the non-distmult representation of sentence. Refer to read_data of PreProcessor
            # text, length = self.get_elmo_tokenization(sentence[0])
            text, length = sentence, len(sentence)
            tokenized_text.append(text)
            tokenized_length.append(length)
        return tokenized_text, tokenized_length

class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.bos = bos
        self.eos = eos

    def create_tokenizations(self, data, elmo_file):
        tokenizer = Tokenizer()
        tokenized_premises, tokenized_premises_lengths = tokenizer.get_batched_elmo_tokenization(data["premises"])
        tokenized_hypotheses, tokenized_hypotheses_lengths = tokenizer.get_batched_elmo_tokenization(data["hypotheses"])

        tokenized_data = {
            "ids": data["ids"],
            "premises": tokenized_premises,
            "premises_lengths": tokenized_premises_lengths,
            "hypotheses": tokenized_hypotheses,
            "hypotheses_lengths": tokenized_hypotheses_lengths,
            "labels": data["labels"],
            "similarity": data["similarity"],
            'max_premise_length': max(tokenized_premises_lengths),
            'max_hypothesis_length': max(tokenized_hypotheses_lengths)
        }

        print("\t* Saving result...")
        with open(elmo_file, "wb") as pkl_file:
            pickle.dump(tokenized_data, pkl_file)

        return

    def read_data_sim(self, path):
        data = pd.read_csv(path, sep='\t', encoding="utf-8")
        data = shuffle(data)
        data.reset_index(drop=True, inplace=True)
        text_a = data['text_a']
        text_b = data['text_b']
        label = data['labels']
        sim = data['similarity']
        ids, premises, hypotheses, labels, similarity = [], [], [], [], []
        premises_length, hypotheses_length = [], []
        j = 0
        for i in range(len(text_a)):
            if type(text_a[i]) == float or type(text_b[i]) == float:
                continue

            lsent = text_a[i].lower().translate(str.maketrans('', '', string.punctuation)).split()
            premises.append(lsent)
            premises_length.append(len(lsent))

            rsent = text_b[i].lower().translate(str.maketrans('', '', string.punctuation)).split()
            hypotheses.append(rsent)
            hypotheses_length.append(len(rsent))
            #print(i)
            if 'sts' in path:
                labels.append(float(label[i]) / 5)
            else:
                labels.append(int(label[i]))
            similarity.append(sim[i])

            ids.append(j)
            j = j + 1

        #print(len(ids), len(premises))

        return {"ids": ids,
                "premises": premises,
                "hypotheses": hypotheses,
                "labels": labels,
                "similarity": similarity,
                "premises_lengths": premises_length,
                "hypotheses_lengths": hypotheses_length,
                "max_premise_length": max(premises_length),
                "max_hypothesis_length": max(hypotheses_length)
                }

    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        for sentence in data["premises"]:
            for word in sentence:
                words.append(word)
        for sentence in data["hypotheses"]:
            for word in sentence:
                words.append(word)


        counts = Counter(words)
        num_words = self.num_words

        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1
        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset


    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])


        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "labels": [],
                            "similarity": [],
                            "premises_lengths": [],
                            "hypotheses_lengths": [],
                            "max_premise_length": 0,
                            "max_hypothesis_length":  0
                            }


        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            transformed_data["labels"].append(label)
            transformed_data["ids"].append(data["ids"][i])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)
            transformed_data["premises_lengths"].append(len(indices))

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)
            transformed_data["hypotheses_lengths"].append(len(indices))

            similarity = data["similarity"][i]
            transformed_data["similarity"].append(similarity)

        transformed_data["max_premise_length"] = max(transformed_data["premises_lengths"])
        transformed_data["max_hypothesis_length"] = max(transformed_data["hypotheses_lengths"])

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))
        print(num_words)
        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            #print(i)
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 proportion=1,
                 isRandom=False,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        length = int(len(data["ids"])*proportion)
        print(length)


        if isRandom == True:
            print('get random data!!!')
            train = pd.DataFrame(columns=("premises", "hypotheses", "labels", "similarity"))
            train["premises"] = data['premises']
            train["hypotheses"] = data['hypotheses']
            train["labels"] = data['labels']
            train["similarity"] = data['similarity']
            train_shuffled = shuffle(train)
            train_shuffled.reset_index(drop=True, inplace=True)

            train = train[:length]
            data = {}
            data["premises"] = list(train["premises"])
            data["hypotheses"] = list(train["hypotheses"])
            data["labels"] = list(train["labels"])
            data["similarity"] = list(train["similarity"])

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long),#if dataset is "STS_B", the dtype is  torch.Float32
                     'similarity': torch.ones((self.num_sequences,self.max_premise_length,
                                               self.max_hypothesis_length),
                                              dtype=torch.float32) * padding_idx}

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(i)
            end = min(len(premise), self.max_premise_length)
            left = len(premise)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            right = len(hypothesis)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])


            similarity = data["similarity"][i]
            similarity = torch.tensor(np.array(eval(similarity)))


            self.data["similarity"][i, :left, :right] = similarity

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                'similarity': self.data["similarity"][index],
                "label": self.data["labels"][index]}


