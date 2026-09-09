import json
import os
import random

from torch.utils.data import TensorDataset, RandomSampler
from transformers import BertModel, BertTokenizer

from config.configuration import data_config
from utils.dataset import DataLoader
from utils.instance import Instance
from utils.vocab import Vocab, MultiVocab, BERTVocab
import torch
import collections
import pickle
import numpy as np



def read_insts(file_reader):
    new_inst = {'tokens': [], 'ner_tags': []}
    for line in file_reader:
        try:
            tokens = line.strip().split()

            if line.strip() == '' or len(tokens) < 2:
                if len(new_inst['tokens']) > 0:
                    yield Instance(**new_inst)
                new_inst = {'tokens': [], 'ner_tags': []}
            else:
                new_inst['tokens'].append(tokens[0])
                new_inst['ner_tags'].append(tokens[-1])
        except Exception as e:
            print('exception occur: ', e)

    if len(new_inst['tokens']) > 0:
        yield Instance(**new_inst)

def read_json_insts(file_reader):
    examples=json.loads(file_reader.read())
    for example in examples:
        tokens=example["sentence"]
        triggers=example["triggers"]
        new_inst = {'tokens': tokens, 'ner_tags': []}
        ner_tags=["O"]*len(tokens)
        for trigger in triggers:
            start=trigger["start"]
            end=trigger["end"]
            event_type=trigger["event_type"]

            ner_tags[start]='B-'+event_type
            for idx in range(start+1,end):
                ner_tags[idx] = 'I-' + event_type
        new_inst["ner_tags"]=ner_tags
        # print(new_inst)
        yield Instance(**new_inst)


def load_data(path):
    assert os.path.exists(path)
    dataset = []
    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_insts(fr):
            if len(inst.tokens) < 512:
                dataset.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset

def load_json_data(path):
    assert os.path.exists(path)
    dataset = []
    too_long = 0
    with open(path, 'r', encoding='utf-8') as fr:
        for inst in read_json_insts(fr):
            if len(inst.tokens) < 512:
                dataset.append(inst)
            else:
                too_long += 1
    print(f'{too_long} sentences exceeds 512 tokens')
    return dataset


def create_vocab(data_sets, bert_model_path=None, embed_file=None):
    bert_vocab = BERTVocab(bert_model_path)
    bert_vocab.build_vocab()

    ner_tag_vocab = Vocab(unk=None, bos=None, eos=None)
    for inst in data_sets:
        ner_tag_vocab.add(inst.ner_tags)

    # if embed_file is not None:
    #     embed_count = char_vocab.load_embeddings(embed_file)
    #     print("%d word pre-trained embeddings loaded..." % embed_count)

    return MultiVocab(dict(
        ner=ner_tag_vocab,
        bert=bert_vocab
    ))


def batch_variable(batch_data, mVocab):
    batch_size = len(batch_data)
    max_seq_len = 1 + max(len(inst.tokens) for inst in batch_data)

    bert_vocab = mVocab['bert']
    ner_tag_vocab = mVocab['ner']
    ner_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    chars = []
    for i, inst in enumerate(batch_data):
        seq_len = len(inst.tokens) + 1
        chars.append(inst.tokens)
        mask[i, :seq_len].fill_(1)
        ner_ids[i, :seq_len] = torch.tensor([ner_tag_vocab.inst2idx(nt) for nt in ['O'] + inst.ner_tags])

    bert_inps = bert_vocab.batch_bertwd2id(chars)
    return Batch(bert_inp=bert_inps,
                 ner_ids=ner_ids,
                 mask=mask)


class Batch:
    def __init__(self, **args):
        for prop, v in args.items():
            setattr(self, prop, v)

    def to_device(self, device):
        for prop, val in self.__dict__.items():
            if torch.is_tensor(val):
                setattr(self, prop, val.to(device))
            elif isinstance(val, collections.abc.Sequence) or isinstance(val, collections.abc.Iterable):
                val_ = [v.to(device) if torch.is_tensor(v) else v for v in val]
                setattr(self, prop, val_)
        return self


def save_to(path, obj):
    if os.path.exists(path):
        return None
    with open(path, 'wb') as fw:
        pickle.dump(obj, fw)
    print('Obj saved!')


def load_from(pkl_file):
    with open(pkl_file, 'rb') as fr:
        obj = pickle.load(fr)
    return obj


def _is_chinese(a_chr):
    return u'\u4e00' <= a_chr <= u'\u9fff'


def extract_entity_span_label_BIO(tags):
    """
    :param labels_id: [B-PER,I-PER,0]
    :return: [{"start":0,"end":1,"label":PER}]
    """

    spans_label = []
    entity_start = None
    entity_label = None
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            # 开始新的实体
            if entity_start is not None:
                # 上一个实体还未结束，先将其添加到列表中
                entity_end = i - 1
                spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
            entity_start = i
            entity_label = tag[2:]
        elif tag.startswith('I-'):
            # 实体内部
            if entity_start is None:
                # 非法的标签序列，直接跳过。指的是O, I-ORG这种
                continue
            if entity_label != tag[2:]:
                # 非法的标签序列，将前面已有的作为一个预测。指的是B-LOC, I-ORG这种
                entity_end = i - 1  # 最后一个实体的i - 1
                spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
                entity_start = None
        else:
            # 标签为O，表示实体结束
            if entity_start is not None:
                entity_end = i - 1  # 最后一个实体的i - 1
                spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
                entity_start = None
                entity_label = None
    if entity_start is not None:
        # 最后一个实体还未结束，将其添加到列表中
        entity_end = len(tags) - 1
        spans_label.append({"start": entity_start, "end": entity_end, "label": entity_label})
    return spans_label


def extract_entity_span_label_IO(labels_id):
    """
    :param labels_id: [2,2,0]
    :return: [{"start":0,"end":1,"label":2}]
    """
    # Note here that it is important to handle both the common case of 4 4 4 0 3
    # and the case of 4 4 4 3, which is a different entity class but adjacent to each other
    span_label_golds = []
    span = {}
    last = 0

    for i in range(len(labels_id)):
        if labels_id[i] != last and last == 0:
            span["start"] = i
            last = labels_id[i]
        elif labels_id[i] != last and last > 0:
            span["end"] = i - 1
            span["label"] = labels_id[i - 1]
            span_label_golds.append(span)
            span = {}
            if labels_id[i] == 0:
                last = 0
            else:
                span["start"] = i
                last = labels_id[i]
    if labels_id[-1] > 0:  # To handle examples with entities at the end
        span["end"] = len(labels_id) - 1
        span["label"] = labels_id[-1]
        span_label_golds.append(span)

    return span_label_golds


