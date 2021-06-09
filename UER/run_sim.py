"""
This script provides an exmaple to wrap UER-py for classification.
"""
import torch
import random
import argparse
import string
import time
import collections
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.utils import shuffle
from nltk.corpus import wordnet as wn
from uer_sim.utils.vocab import Vocab
from uer_sim.utils.constants import *
from uer_sim.utils.tokenizer import *
from uer_sim.layers.embeddings import *
from uer_sim.encoders.bert_encoder import *
from uer_sim.encoders.rnn_encoder import *
from uer_sim.encoders.birnn_encoder import *
from uer_sim.encoders.cnn_encoder import *
from uer_sim.encoders.attn_encoder import *
from uer_sim.encoders.gpt_encoder import *
from uer_sim.encoders.mixed_encoder import *
from uer_sim.utils.optimizers import *
from uer_sim.utils.config import load_hyperparam
from uer_sim.utils.seed import set_seed
from uer_sim.model_saver import save_model

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
        self.encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, sim, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg, sim)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = nn.MSELoss()(logits, soft_tgt)
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def count_labels_num(path):
    '''labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["labels"]])
            labels_set.add(label)'''
    data = pd.read_csv(path, sep='\t', encoding='utf-8')
    labels = list(data["labels"])
    labels_set = set(labels)
    print('label num: ', len(labels_set))
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.train_steps * args.warmup, t_total=args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, sim, soft_tgt=None):
    instances_num = src.size()[0]

    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        sim_batch = sim[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, sim_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, sim_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        sim_batch = sim[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, sim_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, sim_batch, None



def read_dataset(args, path, isshuffle=False):
    dataset, columns = [], {}
    #f = open(sim_path, 'r', encoding = 'utf-8')
    #similarity = f.readlines()
    data = pd.read_csv(path, sep='\t', encoding="utf-8", dtype=str)

    if isshuffle == True:
        print('shuffle train data !!!')
        data = data.sample(frac=1)
        data.reset_index(drop=True, inplace=True)
        text_a, text_b, label, similarity = data['text_a'], data['text_b'], data['labels'], data['similarity']
        proportion = args.proportion
        text_a, text_b, label, similarity = text_a.iloc[:int(len(text_a) * proportion)], text_b.iloc[:int(len(text_b) * proportion)], \
                                            label.iloc[:int(len(label) * proportion)], similarity.iloc[:int(len(similarity) * proportion)]
    else:
        text_a, text_b, label, similarity = data['text_a'], data['text_b'], data['labels'], data['similarity']


    print('data num: ', len(text_a))
    for i in range(len(text_a)):
        #print(i)
        sim = np.array(eval(similarity[i]))
        src_a = [args.vocab.get(t) for t in args.tokenizer.tokenize(text_a[i].lower().translate(str.maketrans('', '', string.punctuation)))]
        # print(args.tokenizer.convert_ids_to_tokens(src_a))
        src_a = [CLS_ID] + src_a + [SEP_ID]
        src_b = [args.vocab.get(t) for t in args.tokenizer.tokenize(text_b[i].lower().translate(str.maketrans('', '', string.punctuation)))]
        src_b = src_b + [SEP_ID]
        src = src_a + src_b
        seg = [1] * len(src_a) + [2] * len(src_b)
        tgt = int(label[i])
        sim_matrix = np.zeros((args.seq_length, args.seq_length))
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
            sim_matrix[:args.seq_length, :args.seq_length] = sim[:args.seq_length, :args.seq_length]
        else:
            #print(sim.shape)
            sim_matrix[:len(seg), :len(seg)] = sim
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)

        dataset.append((src, tgt, seg, sim_matrix))

    return dataset

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, sim_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    sim_batch = sim_batch.to(args.device)

    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, sim_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    sim = torch.FloatTensor([sample[3] for sample in dataset])

    batch_size = args.batch_size
    instances_num = src.size()[0]

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, sim_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg, sim)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        sim_batch = sim_batch.to(args.device)

        with torch.no_grad():
            loss, logits = args.model(src_batch, tgt_batch, seg_batch, sim_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / confusion[i, :].sum().item()
            r = confusion[i, i].item() / confusion[:, i].sum().item()
            f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default="./models/English_uncased_base_model.bin", type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str, default="./models/google_uncased_en_vocab.txt",
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--proportion", type=float, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")
    # Model options.
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", "cnn", "gatedcnn", "attn", "synt", "rcnn", "crnn", "gpt", "bilstm"],
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", choices=["O0", "O1", "O2", "O3"], default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=30,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=1000,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    #set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.model = model

    # Build tokenizer.
    args.tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Training phase.
    trainset = read_dataset(args, args.train_path, isshuffle=True)
    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path)

    #random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    similarity = torch.FloatTensor([example[3] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[4] for example in trainset])
    else:
        soft_tgt = None

    print(soft_tgt)
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    total_loss, result, best_result = 0., 0., 0.

    print("Start training.")
    patience_counter = 0
    patience = 5  # early stopping
    for epoch in range(1, args.epochs_num + 1):
        model.train()
        time_start = time.time()
        for i, (src_batch, tgt_batch, seg_batch, sim_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, similarity, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, sim_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Time: {:.4f}s, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                                 time.time() - time_start,
                                                                                                 total_loss / args.report_steps))
                total_loss = 0.


        print("Start evaluation on dev dataset.")
        result = evaluate(args, devset)
        if result <= best_result:
            patience_counter += 1
        else:
            best_result = result
            patience_counter = 0
            save_model(model, args.output_model_path)

        print("Start evaluation on test dataset.")
        evaluate(args, testset)
        print('--' * 30)

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            print("Final evaluation on the test dataset.")
            if args.test_path is not None:
                print("Test set evaluation.")
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(torch.load(args.output_model_path))
                else:
                    model.load_state_dict(torch.load(args.output_model_path))
                evaluate(args, testset, True)

            break
        if epoch == args.epochs_num:
            print("-> Early stopping: patience limit reached, stopping...")
            print("Final evaluation on the test dataset.")
            if args.test_path is not None:
                print("Test set evaluation.")
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(torch.load(args.output_model_path))
                else:
                    model.load_state_dict(torch.load(args.output_model_path))
                evaluate(args, testset, True)



if __name__ == "__main__":
    main()
