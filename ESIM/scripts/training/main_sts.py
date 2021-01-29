"""
Train the ESIM model on the preprocessed SNLI dataset.
"""
# Aurelien Coet, 2018.

import os
import argparse
import pickle
import torch
import json
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
sys.path.insert(0, "../../")
from torch.utils.data import DataLoader
from esim.data import NLIDataset
from esim.utils import correct_predictions
from esim.model import ESIM
from scipy.stats import spearmanr, pearsonr

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
    preds = []
    golds = []
    #tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(dataloader):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(premises,
                      premises_lengths,
                      hypotheses,
                      hypotheses_lengths)

        loss = criterion(logits.squeeze(1), labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        preds.extend(logits.squeeze(1).data.cpu().numpy())
        golds.extend(labels.data.cpu().numpy())
        #correct_preds += correct_predictions(probs, labels)


    p = pearsonr(preds, golds)
    s = spearmanr(preds, golds)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = p[0]

    return epoch_time, epoch_loss, p[0], s[0]


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

    preds = []
    golds = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            logits = model(premises,
                          premises_lengths,
                          hypotheses,
                          hypotheses_lengths)

            loss = criterion(logits.squeeze(1), labels)
            running_loss += loss.item()
            preds.extend(logits.squeeze(1).data.cpu().numpy())
            golds.extend(labels.data.cpu().numpy())
    p = pearsonr(preds, golds)
    s = spearmanr(preds, golds)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = p[0]
    return epoch_time, epoch_loss, p[0], s[0]

def test(model, dataloader):
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0

    preds = []
    golds = []

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            logits = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths)

            preds.extend(logits.squeeze(1).data.cpu().numpy())
            golds.extend(labels.data.cpu().numpy())
            batch_time += time.time() - batch_start

    p = pearsonr(preds, golds)
    s = spearmanr(preds, golds)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start

    #print("Report precision, recall, and f1:" , p, r, f1)
    return batch_time, total_time, p[0], s[0]


def main(train_file,
         valid_file,
         test_file,
         embeddings_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=2,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None,
         proportion=1,
         output=None):
    """
    Train the ESIM model on the SNLI dataset.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the model.
        valid_file: A path to some preprocessed data that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    with open(train_file, "rb") as pkl:
        train_data = NLIDataset(pickle.load(pkl), proportion, isRandom=True)#training data will be shuffled first, then we will get random data of different proportion

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last = False)


    print("\t* Loading validation data...")
    with open(valid_file, "rb") as pkl:
        valid_data = NLIDataset(pickle.load(pkl))

    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last = False)

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data = NLIDataset(pickle.load(pkl))

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last = False)

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = ESIM(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device,
                 isSTS=True).to(device)

    # -------------------- Preparation for training  ------------------- #
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.

    _, valid_loss, p, s = validate(model,
                                             valid_loader,
                                             criterion)
    print("\t* Validation loss before training: {:.4f}, p correlation: {:.4f}, s correlation: {:.4f}"
          .format(valid_loss, (p*100), (s*100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training ESIM model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, p, s = train(model,
                                           train_loader,
                                           optimizer,
                                           criterion,
                                           epoch,
                                           max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, p correlation: {:.4f}, s correlation: {:.4f}"
              .format(epoch_time, epoch_loss, (p*100), (s*100)))


        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, val_p, s = validate(model,
                                              valid_loader,
                                              criterion)
        
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, p correlation: {:.4f}, s correlation: {:.4f}"
              .format(epoch_time, epoch_loss, (val_p*100), (s*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(val_p)


        print("* Testing for epoch {}:".format(epoch))
        batch_time, total_time, p, s = test(model, test_loader)

        print("-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, p correlation: {:.4f}, s correlation: {:.4f}".format(batch_time, total_time, (p*100), (s*100)))
        print(40 * "==")
        # Early stopping on validation accuracy.
        if epoch > 10:
            if val_p <= best_score:
                patience_counter += 1
            else:
                best_score = val_p
                patience_counter = 0
                # Save the best model. The optimizer is not saved to avoid having
                # a checkpoint file that is too heavy to be shared. To resume
                # training from the best model, use the 'esim_*.pth.tar'
                # checkpoints instead.
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "best_score": best_score,
                            "epochs_count": epochs_count,
                            "train_losses": train_losses,
                            "valid_losses": valid_losses},
                           os.path.join(target_dir, output + "_" + str(proportion) + "_best.pth.tar"))

            if patience_counter >= patience:
                print("-> Early stopping: patience limit reached, stopping...")
                checkpoint = torch.load(os.path.join(target_dir, output + "_" + str(proportion) + "_best.pth.tar"))
                # Retrieving model parameters from checkpoint.
                vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
                embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
                hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
                num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
                print("\t* Final test...")
                model = ESIM(vocab_size,
                             embedding_dim,
                             hidden_size,
                             num_classes=num_classes,
                             device=device,
                             isSTS=True).to(device)
                model.load_state_dict(checkpoint["model"])
                batch_time, total_time, p, s = test(model, test_loader)
                print("-> Final p correlation: {:.4f}, s correlation: {:.4f}".format((p*100), (s*100)))
                os.remove(os.path.join(target_dir, output + "_" + str(proportion) + "_best.pth.tar"))
                break


        if epoch == 30:
            checkpoint = torch.load(os.path.join(target_dir, output + "_" + str(proportion) + "_best.pth.tar"))
            # Retrieving model parameters from checkpoint.
            vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
            embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
            hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
            num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
            print("\t* Final test...")
            model = ESIM(vocab_size,
                         embedding_dim,
                         hidden_size,
                         num_classes=num_classes,
                         device=device,
                         isSTS=True).to(device)
            model.load_state_dict(checkpoint["model"])
            batch_time, total_time, p, s = test(model, test_loader)
            print("-> Final p correlation: {:.4f}, s correlation: {:.4f}".format((p * 100), (s * 100)))
            os.remove(os.path.join(target_dir, output + "_" + str(proportion) + "_best.pth.tar"))
            break




if __name__ == "__main__":
    default_config = "../../config/training/sts_training.json"

    parser = argparse.ArgumentParser(description="Train the ESIM model on PI")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    parser.add_argument("--proportion", required=True, type=float,
                        help="{Proportion of training data}")
    parser.add_argument("--output",
                        default=None, type=str, required=True,
                        help="where to Save model")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(os.path.normpath(os.path.join(script_dir, config["train_data"])),
         os.path.normpath(os.path.join(script_dir, config["valid_data"])),
         os.path.normpath(os.path.join(script_dir, config["test_data"])),
         os.path.normpath(os.path.join(script_dir, config["embeddings"])),
         os.path.normpath(os.path.join(script_dir, config["target_dir"])),
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         args.checkpoint,
         args.proportion,
         args.output)
