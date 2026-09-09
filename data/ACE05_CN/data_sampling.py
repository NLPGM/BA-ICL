
import argparse
import json
import os
import random

import numpy as np
import torch

import sys


def sampling(labels, train_set, num_samples=5):
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    sampled_train_set = []
    if num_samples == -1:
        return sampled_train_set
    count = np.zeros((len(labels),), dtype=np.int64)

    random.shuffle(train_set)

    for instance in train_set:
        print(instance)
        triggers=instance["triggers"]
        type_triggers=[trigger["event_type"] for trigger in triggers]

        indexes = np.where(count < num_samples)[0]
        label_indexed = [id2label[index] for index in indexes]

        if len(label_indexed) == 0:
            break
        if len(set(label_indexed).intersection(set(type_triggers))) == 0:
            continue
        sampled_train_set.append(instance)

        for type in type_triggers:
            count[label2id[type]] += 1

    return sampled_train_set


def get_labels(label_filepath):
    with open(label_filepath, "r", encoding='UTF-8') as f:  # 打开文件
        lines = f.readlines()
    labels = [item.replace('\n', '') for item in lines]
    return labels


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    args = parser.parse_args()

    set_seeds(seed=42)

    label_filepath = 'labels.txt'
    labels = get_labels(label_filepath)

    raw_train_data_filepath = 'train.json'

    with open(raw_train_data_filepath, encoding='utf-8') as f:
        train_set = json.loads(f.read())
    # print(train_set[0])

    for sample_num in [5,10,20,50]:
        sample_train_set = sampling(labels, train_set, num_samples=sample_num)


        sample_data_file = os.path.join("sampled_"+str(sample_num) + '_shot_train.json')
        with open(sample_data_file, encoding='utf-8', mode='w') as f:
            json.dump(sample_train_set, f, indent=2)