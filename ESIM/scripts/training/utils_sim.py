"""
Utility functions for training and validating models.
"""

import time
import torch
import gc
import numpy as np
import torch.nn as nn
import sys
sys.path.insert(0, "../../")
from tqdm import tqdm
from esim.utils import correct_predictions


def train(model,
          dataloader,
          embeddings,
          optimizer,
          criterion,
          batch_size,
          max_gradient_norm,
          testing=True):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    count = 0
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premises"]  # .to(device)
        premises_lengths = torch.LongTensor(batch["premises_lengths"]).to(device)
        hypotheses = batch["hypotheses"]  # .to(device)
        hypotheses_lengths = torch.LongTensor(batch["hypotheses_lengths"]).to(device)
        labels = torch.LongTensor(batch["labels"]).to(device)
        similarity = batch["similarity"].to(device)

        optimizer.zero_grad()
        batch_embeddings = embeddings.get_batch(count)

        logits, probs, _ = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths,
                              batch_embeddings,
                              similarity,
                              batch['max_premise_length'],
                              batch['max_hypothesis_length'])
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        count = count + batch_size

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_pred, out_class = correct_predictions(probs, labels)
        correct_preds += correct_pred
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

        if testing: break
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, embeddings, criterion, batch_size, testing):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            sim = []
            for line in batch["similarity"]:
                sim.append(eval(line))

            premises = batch["premises"]
            premises_lengths = torch.LongTensor(batch["premises_lengths"]).to(device)
            hypotheses = batch["hypotheses"]
            hypotheses_lengths = torch.LongTensor(batch["hypotheses_lengths"]).to(device)
            labels = torch.LongTensor(batch["labels"])
            similarity = sim
            batch_embeddings = embeddings.get_batch(count)

            logits, probs, _ = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths,
                                  batch_embeddings,
                                  similarity,
                                  batch['max_premise_length'],
                                  batch['max_hypothesis_length'])
            # return logits, probs
            # print(premises_lengths)
            # premises_mask, embedded_premises, encoded_premises, premises = model(
            #     premises, premises_lengths, hypotheses, hypotheses_lengths)
            # return premises_mask, embedded_premises, encoded_premises, premises
            loss = criterion(logits, labels)
            running_loss += loss.item()
            accuracy, out_class = correct_predictions(probs, labels)
            # running_accuracy += correct_predictions(probs, labels)
            running_accuracy += accuracy
            count = count + batch_size
            if testing: break

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader))
    return epoch_time, epoch_loss, epoch_accuracy

'''
def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.

    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.

    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    #tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(dataloader):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)
        similarity = batch["similarity"].to(device)

        optimizer.zero_grad()

        logits, probs = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths,
                              similarity)
        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probs, labels)


    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)
            similarity = batch["similarity"].to(device)

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths,
                                  similarity)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy
'''