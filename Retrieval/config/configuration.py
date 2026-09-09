import os
import json
import argparse


def data_config(data_path):
    assert os.path.exists(data_path)
    with open(data_path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    print(opts)
    return opts


def args_config():
    parse = argparse.ArgumentParser('Parameter Configuration')
    parse.add_argument('--cuda', type=int, default=0, help='cuda device, default cpu')
    parse.add_argument('--seed', type=int, default=0, help='cuda device, default cpu')

    parse.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate of training')
    # parse.add_argument('-bt1', '--beta1', type=float, default=0.9, help='beta1 of Adam optimizer 0.9')
    # parse.add_argument('-bt2', '--beta2', type=float, default=0.99, help='beta2 of Adam optimizer 0.999')
    parse.add_argument('-eps', '--eps', type=float, default=1e-8, help='eps of Adam optimizer 1e-8')
    parse.add_argument('-warmup', '--warmup_step', type=int, default=10000, help='warm up steps for optimizer')
    parse.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for Adam optimizer')
    parse.add_argument('--scheduler', choices=['cosine', 'inv_sqrt', 'exponent', 'linear', 'step', 'const'],
                       default='linear', help='the type of lr scheduler')
    parse.add_argument('--grad_clip', type=float, default=5., help='the max norm of gradient clip')
    parse.add_argument('--bert_grad_clip', type=float, default=1., help='the max norm of gradient clip')
    parse.add_argument('--patient', type=int, default=5, help='patient number in early stopping')

    parse.add_argument('--batch_size', type=int, default=16, help='batch size of source inputs')
    parse.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parse.add_argument('--epoch', type=int, default=20, help='number of training')
    parse.add_argument('--update_step', type=int, default=1, help='gradient accumulation and update per x steps')

    parse.add_argument("--bert_lr", type=float, default=2e-5, help='bert learning rate')
    parse.add_argument("--bert_layer", type=int, default=8, help='the number of last bert layers')
    parse.add_argument('--bert_embed_dim', type=int, default=768, help='feature size of bert inputs')
    parse.add_argument('--hidden_size', type=int, default=400, help='feature size of hidden layer')
    parse.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')

    parse.add_argument('--dataset', type=str, default='ACE05', help='dataset')

    parse.add_argument('--model_chkp', type=str, default='model.pkl', help='model saving path')

    parse.add_argument('--vocab_chkp', type=str, default='vocab.pkl', help='vocab saving path')

    parse.add_argument('--k_shot', type=int, default=5, help='k_shot')

    parse.add_argument("--do_entity_semi_generate", action="store_true",
                       help="Whether to do_gpt3_context_semi_generate.")
    parse.add_argument("--do_context_semi_generate", action="store_true",
                       help="Whether to do_gpt3_context_semi_generate.")
    parse.add_argument("--do_gpt3_NeuroCF_generate", action="store_true",
                       help="Whether to do_gpt3_NeuroCF_generate.")
    parse.add_argument("--do_gpt3_LooseSame_generate", action="store_true",
                       help="Whether to do_gpt3_LooseSame_generate.")
    parse.add_argument("--do_gpt3_ReplaceCasualTerm_generate", action="store_true",
                       help="Whether to do_gpt3_ReplaceCasualTerm_generate.")

    parse.add_argument("--do_gpt3_FineGrainedCF_generate", action="store_true",
                       help="Whether to do_gpt3_FineGrainedCF_generate.")

    parse.add_argument("--do_ood1_test", action="store_true",
                       help="Whether to do_ood1_test.")

    parse.add_argument("--do_ood2_test", action="store_true",
                       help="Whether to do_ood2_test.")

    parse.add_argument("--do_validate", action="store_true",
                       help="Whether to do_validate.")

    parse.add_argument("--output_dir", default="./output/", type=str,
                       help="The output_dir.", )

    parse.add_argument("--sample_data_dir", default="../data", type=str,
                       help="The aug_data_dir.", )

    # parse.add_argument("--sample_id", type=int, default=0,help="sample sample_id")

    parse.add_argument("--aug_data_dir", default="./AugData", type=str,
                       help="The aug_data_dir.", )
    parse.add_argument("--raw_data_dir", default="./RawData", type=str,
                       help="The raw data dir. Before the data augmentation.", )

    parse.add_argument("--max_seq_length", default=128, type=int,
                       help="The maximum total input sequence length after tokenization. "
                            "Sequences longer than this will be truncated, "
                            "sequences shorter will be padded.", )

    parse.add_argument("--predict_results_dir", default="./predict_results/", type=str,
                       help="The output_dir.", )

    args = parse.parse_args()

    print(vars(args))

    return args
