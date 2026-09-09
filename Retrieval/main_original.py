import json
import os
import random

import numpy as np
import torch

from config.configuration import args_config, data_config
from logger.logger import logger
from train import Trainer
from utils.dataset import DataLoader
from utils.datautil import load_data, load_json_data


def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def write_CONLL_formatted(example_tokens, example_tags, filepath):
    with open(filepath, 'a') as f:
        for i in range(len(example_tokens)):
            f.write(example_tokens[i] + ' ' + example_tags[i] + '\n')
            if i == len(example_tokens) - 1:
                f.write('\n')
    pass


if __name__ == '__main__':

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    data_path = data_config('config/data_path.json')
    ########################################################

    for path in ['./output', './predict_results']:
        if not os.path.exists(path):
            os.makedirs(path)

    ########################################################
    few_shot_filepath = f'{args.sample_data_dir}/{args.dataset}/sampled_{str(args.k_shot)}_shot_train.json'

    few_shot_train_set = load_json_data(few_shot_filepath)

    # val_set = load_json_data(data_path[args.dataset]['dev'])
    iid_test_set = load_json_data(data_path[args.dataset]['iid_test'])
    iid_test_loader = DataLoader(iid_test_set, batch_size=args.test_batch_size)

    ckpt_dir = f'./checkpoint/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    args.ckpt_dir=ckpt_dir

    args.model_chkp = os.path.join(args.ckpt_dir,
                                   f'bert_ed_retriever.ckpt')
    args.vocab_chkp = os.path.join(args.ckpt_dir,
                                   f'vocab.pkl')

    aug_res_iid = {'p': [], 'r': [], 'f': []}

    # seeds = [0, 42, 77, 88, 99]
    seeds = [44]

    for seed in seeds:
        set_seeds(seed)
        ori_trainer = Trainer(args, data_path, train_set=few_shot_train_set, val_set=[], test_set=iid_test_set)

        run_aug_prf = ori_trainer.run(do_validate=False)

        # run的结果不能直接作为最终结果，要把重新加载参数的作为最终结果
        ori_trainer.restore_states(args.model_chkp)

        # IID
        ori_trainer.predict_results_file = os.path.join(args.predict_results_dir,
                                                        f'IID_ORIGINAL_{str(args.k_shot)}_{args.dataset}_predict_results.txt')

        aug_prf_iid = ori_trainer.evaluate(iid_test_loader)
        aug_res_iid['p'].append(aug_prf_iid['p'])
        aug_res_iid['r'].append(aug_prf_iid['r'])
        aug_res_iid['f'].append(aug_prf_iid['f'])

    aug_res_iid_average = {'p': sum(aug_res_iid['p']) / len(seeds),
                           'r': sum(aug_res_iid['r']) / len(seeds),
                           'f': sum(aug_res_iid['f']) / len(seeds)}

    with open('./results.txt', 'w') as f:
        f.write('---------------------------------------------' + '\n')
        f.write('Sample num: ' + str(args.k_shot) + '\n')

        f.write('ORIGINAL Result  -IID: ' + str(aug_res_iid) + '\n')

        f.write('ORIGINAL Result (Average) -IID: ' + str(aug_res_iid_average) + '\n')
        f.write('---------------------------------------------' + '\n')
