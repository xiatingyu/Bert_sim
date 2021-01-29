import time
import subprocess
import itertools
from multiprocessing import Process
import os
import threading
import json
import copy

def get_config_path_from_options(config, options):
    curr_config = copy.deepcopy(config)
    curr_config["hidden_size"] = options[0]
    curr_config["dropout"] = options[1]
    curr_config["multipassiterations"] = options[2]
    curr_config["lstm"] = options[3]
    curr_config["weightedattention"] = options[4]
    curr_config["lr"] = options[5]
    dir = config["target_dir"]
    curr_config["target_dir"] = curr_config["target_dir"] + \
                                    "multi" + str(options[2]) + \
                                    "hidden_size" + str(options[0]) + \
                                    "dropout" + str(options[1]) + \
                                    "lstm" + str(curr_config["lstm"]) + \
                                    "weightedAtt" + str(curr_config["weightedattention"]) + "/"
                                    # "lr" + str(config["lr"]) + \
                                    # "grad" + str(config["max_gradient_norm"]) + \
    print(curr_config)
    os.system("mkdir -p " + curr_config["target_dir"])
    filePath = curr_config["target_dir"] + "config.json"
    outFile = open(filePath, "w")
    outFile.write(json.dumps(curr_config))
    return curr_config["target_dir"]

def create_new_process(config_path, i):
    if i % 2 == 0:
        os.system("export CUDA_VISIBLE_DEVICES=0")
    else:
        os.system("export CUDA_VISIBLE_DEVICES=1")
    os.system("mkdir -p " + config["target_dir"])
    cmd = "CUDA_VISIBLE_DEVICES=" + str(i%2) + " python -u /home/soumyasharma/ESIM-github/ESIM/scripts/training/train_mednli.py --config " + config_path + "config.json | tee " + config_path + "console.output"
    print(cmd)
    child = subprocess.Popen(cmd, shell=True)
    return child

def wait_timeout(config, processes, task_list, seconds):
    curr = 0
    while curr != len(task_list):
        for idx, proc in enumerate(processes):
            result = proc.poll()
            if result is None:
                continue
            else:
                if curr < len(task_list):
                    child_process = create_new_process(get_config_path_from_options(config, task_list[curr]), idx)
                    processes[idx] = child_process
                    curr += 1
        time.sleep(seconds)
    for proc in processes:
        result = proc.poll()
        while result is None:
            time.sleep(seconds)
            result = proc.poll()

    return

if __name__ == "__main__":

    embedding_dim_options = [150,200]
    dropout_options = [0.5]
    multipassiterations_options = [1]
    lstm_options = [True]# , False]
    weightedattention_options = [False] #[True, False]
    lr_options = [0.001, 0.0001]

    s = [embedding_dim_options, dropout_options, multipassiterations_options, lstm_options, weightedattention_options, lr_options]
    options_list = list(itertools.product(*s))
    print("Hello!")
    config = {
        "train_data": "../../data/preprocessed/conceptnet+umls/train_data.pkl",
        "valid_data": "../../data/preprocessed/conceptnet+umls/dev_data.pkl",
        "test_data": "../../data/preprocessed/conceptnet+umls/test_data.pkl",
        "train_embeddings": "../../data/preprocessed/conceptnet+umls/train_elmo.pkl",
        "valid_embeddings": "../../data/preprocessed/conceptnet+umls/dev_elmo.pkl",
        "test_embeddings": "../../data/preprocessed/conceptnet+umls/test_elmo.pkl",
        "target_dir": "../../data/checkpoints/conceptnet+umls/distmult1multi1hidden_size500dropout0.5lr0.0001grad10lstmTweightedAttF/",
        "dataset": "elmo",
        "distmult": 1,
        "distmultPath": "../../data/dataset/conceptnet_glove_style_embedding.vec",
        "distmultEmbeddingDim": 300,

        "embedding_dim": 1024,
        "hidden_size": 150,
        "dropout": 0.5,
        "num_classes": 3,
        "epochs": 64,
        "batch_size": 32,
        "lr": 0.0001,
        "patience": 5,
        "max_gradient_norm": 10.0,
        "testing": False,
        "multipassiterations": 1,
        "lstm": True,
        "weightedattention": False,
        "sentiment": False
    }

    print("Hello!")

    embedding_dim_options_ = [150, 200]
    dropout_options_ = [0.3, 0.4, 0.5]
    multipassiterations_options_ = [1]
    lstm_options_ = [True]# , False]
    weightedattention_options_ = [False] #[True, False]
    lr_options_ = [0.001, 0.0001]

    s_ = [embedding_dim_options_, dropout_options_, multipassiterations_options_, lstm_options_, weightedattention_options_, lr_options_]
    options_list_ = list(itertools.product(*s_))

    processes = []
    no_processes = min(6, len(options_list))
    for idx in range(no_processes):
        processes.append(create_new_process(get_config_path_from_options(config, options_list[idx]), idx))

    wait_timeout(config, processes, options_list[no_processes:], 1)

