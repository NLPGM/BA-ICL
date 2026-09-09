import argparse
import copy
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from prompt_tools import parse_outputs_stage1, \
    construct_original_scm_prompt, \
    parse_reasoning_explanations_from_text, get_formed_key_phase, split_triggers_rationals
from utils import prepare_label_set, prepare_llm, templated_prompts, log_result, summarize_token_usage

import sys
import os

# sys.path.append(os.path.abspath(os.path.join(__file__, "../", "..")))

from Retrieval.model import BertEmbedding, EDContrastiveRetriever
from Retrieval.utils.ed_utils import GetDataEnumerater

import torch

#
def format_pair_intervention_data(paired_intervention_data):
    formatted_train_samples = []
    cat_labels = []

    processed_golden_sub_rationals = []

    for paired_intervention_instance in paired_intervention_data:
        # sentence_str = paired_intervention_instance["sentence_str"]
        # golden_triggers = paired_intervention_instance["triggers"]

        golden_event_type = paired_intervention_instance["golden_event_type"]
        golden_sub_rational = paired_intervention_instance["golden_sub_rational"]

        pred_event_type = paired_intervention_instance["pred_event_type"]
        pred_sub_rational = paired_intervention_instance["pred_sub_rational"]

        cat_label = f'{pred_event_type}#{golden_event_type}'
        cat_labels.append(cat_label)
        formatted_sample = {
            "cat_label": cat_label,
            "rational": pred_sub_rational,
        }
        formatted_train_samples.append(formatted_sample)
        if golden_event_type != 'Other':  # 对于 无事件类别的不记录golden
            if golden_sub_rational not in processed_golden_sub_rationals:  # 一个 子 golden_sub_rational 只处理一次

                cat_label = f'{golden_event_type}#{golden_event_type}'  # 实际上是两个一样的relation type拼在了一起
                cat_labels.append(cat_label)
                formatted_sample = {
                    "cat_label": cat_label,
                    "rational": golden_sub_rational,
                }
                formatted_train_samples.append(formatted_sample)

            processed_golden_sub_rationals.append(golden_sub_rational)

    cat_labels = list(set(cat_labels))
    return formatted_train_samples, cat_labels


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_stage1_retriever(args):
    ckpt_dir = f'Retrieval/checkpoint/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'
    checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, 'bert_ed_retriever.ckpt'), map_location=args.device,weights_only=False)

    trained_ED_BertEmbedding_model = BertEmbedding(model_path='Retrieval/bert-base-uncased').to(args.device)

    trained_ED_BertEmbedding_model.load_state_dict(checkpoint_bert_model['bert_model_state'])
    emb_retriever_stage1 = trained_ED_BertEmbedding_model

    if args.retrieval_stage1_mode == 'task_specific_retrieval':
        ckpt_dir = f'Retrieval/checkpoint/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'
        checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, 'bert_ed_retriever.ckpt'), map_location=args.device,weights_only=False)

        trained_ED_BertEmbedding_model = BertEmbedding(model_path='Retrieval/bert-base-uncased').to(args.device)

        trained_ED_BertEmbedding_model.load_state_dict(checkpoint_bert_model['bert_model_state'])
        emb_retriever_stage1 = trained_ED_BertEmbedding_model

    elif args.retrieval_stage1_mode == 'simcse_retrieval':
        # 此时不需要加载微调后的checkpoint，直接加载simcse即可
        ed_bert_model = BertEmbedding(model_path=args.simcse_path).to(args.device)

        emb_retriever_stage1 = ed_bert_model

    return emb_retriever_stage1


def prepare_prompts_for_stage1(args, test_examples, labeled_samples_with_x, emb_retriever_stage1):
    if 'retrieval' in args.retrieval_stage1_mode:  # 有两种可能：simcse_retrieval or task_specific_retrieval
        # 换一种写法，把预测样本的向量存起来

        dataloader_test_examples = GetDataEnumerater(args=args,
                                                     samples=test_examples,
                                                     batch_size=32,
                                                     )
        dataloader_demo_examples = GetDataEnumerater(args=args,
                                                     samples=labeled_samples_with_x,
                                                     batch_size=32,
                                                     )
        logits_test_examples = None
        for out_batch_idx, batch in tqdm(enumerate(dataloader_test_examples), desc=f'Running retrieval',
                                         total=len(dataloader_test_examples)):
            batch_samples_emb = emb_retriever_stage1.get_emb(
                input_ids=batch[0].to(args.device),
                special_mask=batch[1].to(args.device),
                token_type_ids=batch[2].to(args.device),
                attention_mask=batch[3].to(args.device),
            )
            batch_logits_test_examples = None

            for inner_batch_idx, batch in enumerate(dataloader_demo_examples):
                batch_demos_emb = emb_retriever_stage1.get_emb(
                    input_ids=batch[0].to(args.device),
                    special_mask=batch[1].to(args.device),
                    token_type_ids=batch[2].to(args.device),
                    attention_mask=batch[3].to(args.device),
                )

                batch_logits = torch.matmul(batch_demos_emb, batch_samples_emb.T).detach().cpu()
                # batch_demos_emb = batch_demos_emb.detach().cpu()

                if inner_batch_idx == 0:
                    batch_logits_test_examples = batch_logits
                else:
                    batch_logits_test_examples = torch.cat((batch_logits_test_examples, batch_logits), dim=0)
                # print(batch_logits_test_examples.size())
            if out_batch_idx == 0:
                logits_test_examples = batch_logits_test_examples
            else:
                logits_test_examples = torch.cat((logits_test_examples, batch_logits_test_examples), dim=1)

        logits_test_examples = logits_test_examples.T

        # logits_test_examples 是 (示例数 * 测试样本数).T
        print(logits_test_examples.size())
        _rank_values, ranks_test_examples = logits_test_examples.topk(args.demo_num, dim=1)
        print(ranks_test_examples.size())
        ranks_test_examples = ranks_test_examples.numpy()

    elif args.retrieval_stage1_mode == 'random':
        pass

    original_scm_prompts = []
    all_icl_ins_stage1_retrieved = []
    for test_idx, test_example in enumerate(test_examples):
        if 'retrieval' in args.retrieval_stage1_mode:
            rank_this_test = ranks_test_examples[test_idx]
            selected_demos = [labeled_samples_with_x[demo_idx] for demo_idx in rank_this_test]
        elif args.retrieval_stage1_mode == 'random':
            selected_demos = random.sample(labeled_samples_with_x, k=args.demo_num)

        original_scm_prompt = construct_original_scm_prompt(args=args, sample=test_example, icl_samples=selected_demos)

        original_scm_prompts.append(original_scm_prompt)
        all_icl_ins_stage1_retrieved.append(selected_demos)
    return original_scm_prompts, all_icl_ins_stage1_retrieved


def load_stage2_retriever(args):
    ckpt_dir = f'Retrieval/checkpoint-{args.llm_type}/{args.dataset}-{str(args.seed)}-{str(args.k_shot)}-shot'

    checkpoint_bert_model = torch.load(os.path.join(ckpt_dir, f'{args.retrieval_stage2_mode}.ckpt'),
                                       map_location=args.device,weights_only=False)

    if args.retrieval_stage2_mode == 'ed_contrastive_retriever':
        ED_contrastive_retriever = EDContrastiveRetriever(args=args,
                                                          PLM=args.retriever_plm,
                                                          PLM_hidden_size=768).to(args.device)
        ED_contrastive_retriever.load_state_dict(checkpoint_bert_model['model_state_dict'])
        biasx_emb_retriever = ED_contrastive_retriever

    return biasx_emb_retriever


def check_rational_after_feedback(args, rational_after_feedback_samples, paired_intervention_data,
                                  emb_retriever_stage2):
    # 准备有偏无偏簇的原型表示；
    # 在 paired_intervention_data 中，所有的 golden_sub_rational 都属于无偏簇，归结到“unbiased_{golden_event_type}”原型簇中；
    # 存在两类 instance；如果 pred_event_type==golden_event_type，则 pred_sub_rational 属于无偏簇，归结到“unbiased_{golden_event_type}” 簇中；
    #                  否则，pred_sub_rational 属于有偏簇，归结到“biased_{pred_event_type}” 簇中；

    unbiased_clusters = {}
    biased_clusters = {}

    for paired_instance_idx, paired_intervention_instance in enumerate(paired_intervention_data):
        golden_event_type = paired_intervention_instance["golden_event_type"]
        golden_sub_rational = paired_intervention_instance["golden_sub_rational"]

        pred_event_type = paired_intervention_instance["pred_event_type"]
        pred_sub_rational = paired_intervention_instance["pred_sub_rational"]

        # 在 paired_intervention_data 中，所有的 golden_sub_rational 都属于无偏簇，归结到“unbiased_{golden_event_type}”原型簇中；
        if golden_event_type not in unbiased_clusters.keys():
            unbiased_clusters[golden_event_type] = [{"paired_instance_idx": paired_instance_idx,
                                                     "rational": golden_sub_rational}]
        else:
            unbiased_clusters[golden_event_type].append({"paired_instance_idx": paired_instance_idx,
                                                         "rational": golden_sub_rational})

        # 存在两类 instance；如果 pred_event_type==golden_event_type，则 pred_sub_rational 属于无偏簇，归结到“unbiased_{golden_event_type}” 簇中；
        if pred_event_type == golden_event_type:
            unbiased_clusters[golden_event_type].append({"paired_instance_idx": paired_instance_idx,
                                                         "rational": pred_sub_rational})
        else:
            #  否则，pred_sub_rational 属于有偏簇，归结到“biased_{pred_event_type}” 簇中；
            if pred_event_type not in biased_clusters.keys():
                biased_clusters[pred_event_type] = [{"paired_instance_idx": paired_instance_idx,
                                                     "rational": pred_sub_rational}]
            else:
                biased_clusters[pred_event_type].append({"paired_instance_idx": paired_instance_idx,
                                                         "rational": pred_sub_rational})

    def get_rational_instances_emb(args, emb_retriever_stage2, instances):
        dataloader_instances = GetDataEnumerater(args=args,
                                                 samples=instances,
                                                 batch_size=16
                                                 )
        emb_instances = []
        for batch_idx, batch in enumerate(dataloader_instances):
            batch_mean_emb = emb_retriever_stage2.get_emb(
                input_ids=batch[0].to(args.device),
                special_mask=batch[1].to(args.device),
                token_type_ids=batch[2].to(args.device),
                attention_mask=batch[3].to(args.device),
            ).detach()
            if batch_idx == 0:
                emb_instances = batch_mean_emb
            else:
                emb_instances = torch.cat((emb_instances, batch_mean_emb), dim=0)
        return emb_instances

    collected_rational_after_feedback_samples = {}  # 收集具有同个预测标签且待纠偏的样本，集中键值为第一次预测的 test_biased_label

    # status_rational_after_feedback_samples 记录每一个测试样本 应不应该被 纠偏；
    # 内部为 False，表示没有预测结果；
    # 内部为[False, list[] ]表示有预测结果，且有需要纠偏的部分; list[] 为下一步所建议的 pair intervention示例索引列表
    status_rational_after_feedback_samples = [False] * len(rational_after_feedback_samples)
    for rational_after_feedback_sample_idx, rational_after_feedback_sample in (
            enumerate(rational_after_feedback_samples)):
        sub_trigger_rational_dict_after_feedback = rational_after_feedback_sample[
            "sub_trigger_rational_dict_after_feedback"]
        length_pred_triggers = len(sub_trigger_rational_dict_after_feedback.keys())
        status_rational_after_feedback_samples[rational_after_feedback_sample_idx] = [False] * length_pred_triggers

    for rational_after_feedback_sample_idx, rational_after_feedback_sample in (
            enumerate(rational_after_feedback_samples)):
        sub_trigger_rational_dict_after_feedback = rational_after_feedback_sample[
            "sub_trigger_rational_dict_after_feedback"]
        # print(sub_trigger_rational_dict_after_feedback)

        for inner_idx, item in enumerate(sub_trigger_rational_dict_after_feedback.values()):
            # pred_trigger_text = item['trigger_text']
            pred_trigger_event_type = item['trigger_event_type']
            # pred_trigger_sub_rational = item['sub_rational']

            if pred_trigger_event_type not in collected_rational_after_feedback_samples.keys():
                collected_rational_after_feedback_samples[pred_trigger_event_type] = [{
                    "out_index": rational_after_feedback_sample_idx,
                    "inner_idx": inner_idx
                }]
            else:
                collected_rational_after_feedback_samples[pred_trigger_event_type].append({
                    "out_index": rational_after_feedback_sample_idx,
                    "inner_idx": inner_idx,  # inner_idx 用来记录具体对应的sub rational
                })
    ################################################################3

    for test_biased_label in tqdm(collected_rational_after_feedback_samples.keys(),
                                  total=len(collected_rational_after_feedback_samples.keys()),
                                  desc='Processing in collects.'):
        if test_biased_label == args.event_type_set[0]:
            # TODO: 对于预测出来的负样本，不做处理；原因是其来源可能过于多样？
            continue

        if test_biased_label not in biased_clusters.keys():
            # 预测的标签不在有偏簇中出现过，直接不考虑
            continue

        # TODO: 暂时这么做，不过后面需要改pair intervention的形成过程
        if test_biased_label not in unbiased_clusters.keys():
            # 预测的标签不在无偏簇中出现过，直接不考虑
            continue

        collected_idxes = collected_rational_after_feedback_samples[test_biased_label]

        collected_pred_rationals = []
        for out_and_in_index in collected_idxes:
            out_index = out_and_in_index["out_index"]
            inner_idx = out_and_in_index["inner_idx"]

            # print(rational_after_feedback_samples[out_index]["sub_trigger_rational_dict_after_feedback"].values())
            # print(rational_after_feedback_samples[out_index]["sub_trigger_rational_dict_after_feedback"].values()[inner_idx])
            sub_rational = \
            list(rational_after_feedback_samples[out_index]["sub_trigger_rational_dict_after_feedback"].values()
                 )[inner_idx]['sub_rational']

            collected_pred_rationals.append(
                {
                    "rational": sub_rational
                }
            )

            # rational_after_feedback_samples[out_index]["sub_trigger_rational_dict_after_feedback"].values()[inner_idx]
            # ['sub_rational']

        # biased_demos_cluster = biased_clusters[test_biased_label]

        biased_rationals = biased_clusters[test_biased_label]
        unbiased_rationals = unbiased_clusters[test_biased_label]

        # 这里依然先保留对于具体的样本的计算来看有无偏差，后面再考虑删去这一步看有没有副作用
        # 接着按照计算结果 把 status_test_examples 中的部分 True 改为 False （即如果与无偏的更接近，则选择不纠偏）
        # 接着，把 status_test_examples 中的其余 True 改为 具体的 pair_ins 列表（filtered_biased_pair_instances中的）

        emb_unbiased_rationals = get_rational_instances_emb(args=args,
                                                            emb_retriever_stage2=emb_retriever_stage2,
                                                            instances=unbiased_rationals)

        emb_biased_rationals = get_rational_instances_emb(args=args,
                                                          emb_retriever_stage2=emb_retriever_stage2,
                                                          instances=biased_rationals)

        emb_collected_pred_rationals = get_rational_instances_emb(args=args,
                                                                  emb_retriever_stage2=emb_retriever_stage2,
                                                                  instances=collected_pred_rationals)

        logits_unbiased_rationals = torch.matmul(emb_collected_pred_rationals, emb_unbiased_rationals.T)
        logits_biased_rationals = torch.matmul(emb_collected_pred_rationals, emb_biased_rationals.T)

        max_logit_unbiased_rationals = torch.max(logits_unbiased_rationals, dim=-1).values.tolist()
        max_logit_biased_rationals = torch.max(logits_biased_rationals, dim=-1).values.tolist()

        # print(max_logit_unbiased_rationals)
        # print(max_logit_biased_rationals)

        # 该 local_idx 和 collected_pred_rationals 一一对应；
        for local_idx, out_and_in_index in enumerate(collected_idxes):
            out_index = out_and_in_index["out_index"]
            inner_idx = out_and_in_index["inner_idx"]

            if max_logit_unbiased_rationals[local_idx] > max_logit_biased_rationals[local_idx]:
                status_rational_after_feedback_samples[out_index][inner_idx] = False  # 即如果与无偏的更接近，则选择不纠偏
            else:
                # 否则，则将与 max_logit_biased_rationals 最接近的 biasx_retriever_topk 个对应的 paired_instance_idx
                # 存成列表 放在 status_test_examples[test_idx]
                # assert status_test_examples[test_idx] == True

                rank_indices = logits_biased_rationals[local_idx].topk(
                    min(args.biasx_retriever_topk, len(biased_rationals)),
                    dim=0).indices.cpu().numpy().tolist()
                # rank_indices.reverse()

                # 该 rank_indices 指的是在 logits_biased_rationals[local_idx] 中的索引，并非 paired instance 的索引，所以要转化一下
                paired_ins_indices = [biased_rationals[rank_indice]["paired_instance_idx"]
                                      for rank_indice in rank_indices]

                status_rational_after_feedback_samples[out_index][inner_idx] = paired_ins_indices

    for out_index, status in enumerate(status_rational_after_feedback_samples):
        if status != False:  # 当flag为一个列表时，要考虑 要不要把这个列表转为 False

            flag = True
            for s in status:
                if s != False:
                    flag = False
            if flag:  # 如果status中所有元素均为 False，则status置为 False
                status_rational_after_feedback_samples[out_index] = False

    return status_rational_after_feedback_samples


def prepare_prompts_for_stage2(args, test_examples, paired_intervention_data, emb_retriever_stage2):
    # print(status_test_examples[:100])
    # 这里这么命名是为了后续循环方便

    rational_after_feedback_samples = []
    for test_idx, test_example in enumerate(test_examples):
        rational_after_feedback_samples.append(
            {
                "test_idx": test_idx,
                "sub_trigger_rational_dict_after_feedback": test_example["biased_sub_trigger_rational_dict"],
            }
        )
    status_rational_after_feedback_samples = check_rational_after_feedback(args=args,
                                                                           rational_after_feedback_samples=rational_after_feedback_samples,
                                                                           paired_intervention_data=paired_intervention_data,
                                                                           emb_retriever_stage2=emb_retriever_stage2)

    # status_rational_after_feedback_samples 记录每一个测试样本 应不应该被 纠偏；
    # 内部为 False，表示没有预测结果；
    # 内部为[False, list[] ]表示有预测结果，且有需要纠偏的部分; list[] 为下一步所建议的 pair intervention示例索引列表

    # 因为这里 len(status_rational_after_feedback_samples)==len(rational_after_feedback_samples)
    # 所以直接进行转换
    status_test_examples = status_rational_after_feedback_samples

    # paired_intervention_data  status_test_examples  test_examples
    # 这三者可以共同，构成下一轮次的 prompt

    feedback_reconstructed_prompt_dicts = []

    for test_idx, (status, test_example) in (
            enumerate(zip(status_test_examples, test_examples))):

        if status == False:
            temp_dict = {
                "prompt": '[No Feedback Here]',
                "biased_triggers": [],
            }
        else:
            # 首先要将 每个 trigger 对应的逐一收集，当达到最大值 args.biasx_retriever_topk 时停止收集
            selected_paired_ins_idx = []
            flag = False
            for topk in range(args.biasx_retriever_topk):
                for sub_status in status:
                    if sub_status != False:  # 表示 sub_status 为列表
                        if topk >= len(sub_status):  # 当前topk大于该sub 提供的选择
                            continue
                        paired_ins_idx = sub_status[topk]
                        if paired_ins_idx not in selected_paired_ins_idx:
                            selected_paired_ins_idx.append(paired_ins_idx)
                            if len(selected_paired_ins_idx) == args.biasx_retriever_topk:
                                flag = True
                                break
                if flag == True:
                    break

            feedback_icl_paired_instances = [paired_intervention_data[paired_ins_idx] for paired_ins_idx in
                                             selected_paired_ins_idx]
            selected_feedback_icl = []
            for paired_instance in feedback_icl_paired_instances:
                selected_feedback_icl.append(
                    {
                        "sentence": paired_instance["sentence_str"].split(' '),
                        "triggers": paired_instance["triggers"],
                        "explanations": paired_instance["explanations"]
                    }
                )

            feedback_reconstructed_prompt = construct_original_scm_prompt(args=args,
                                                                          sample=test_example,
                                                                          icl_samples=selected_feedback_icl)

            biased_sub_trigger_rational_dict = test_example['biased_sub_trigger_rational_dict']
            biased_triggers = []
            for idx, trigger_span in enumerate(biased_sub_trigger_rational_dict.keys()):
                if status[idx] != False:
                    biased_triggers.append(biased_sub_trigger_rational_dict[trigger_span])

            temp_dict = {
                "prompt": feedback_reconstructed_prompt,
                "biased_triggers": biased_triggers,
            }

        feedback_reconstructed_prompt_dicts.append(temp_dict)

    return feedback_reconstructed_prompt_dicts


def parse_outputs_stage2(args, outputs, test_examples, all_biased_triggers_stage2, response_record_filepath):
    for idx, (output, test_example, biased_triggers) in enumerate(
            zip(outputs, test_examples, all_biased_triggers_stage2)):
        pred_triggers_stage1 = test_example['biased_pred_triggers']
        reasoning_explanations_stage1 = test_example['biased_explanations']
        sub_trigger_rational_dict_stage1 = split_triggers_rationals(triggers=pred_triggers_stage1,
                                                                    rationals_str=reasoning_explanations_stage1)

        pred_triggers_stage1_dict = {}
        for trigger in pred_triggers_stage1:
            pred_triggers_stage1_dict[trigger['text']] = trigger['event_type']

        biased_triggers_text = [item["trigger_text"] for item in biased_triggers]

        rational_after_feedback = ''

        prompt = output.prompt  # 获取原始的输入提示
        generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本

        if prompt == '[No Feedback Here]':
            rational_after_feedback = '[No Feedback Here]'

            # biased_label即为第一次预测的结果，如果无偏，不需要纠正
            pred_triggers = pred_triggers_stage1
        else:

            new_pred_triggers, rational_after_feedback = parse_reasoning_explanations_from_text(args=args,
                                                                                                generated_text=generated_text)
            new_sub_trigger_rational_dict = split_triggers_rationals(triggers=new_pred_triggers,
                                                                     rationals_str=rational_after_feedback)

            # 我们并不是要保留所有的 此时的 pred_triggers
            # 只是对需要纠偏的trigger，更新原来的预测结果
            new_pred_triggers_dict = {}
            for trigger_text in new_sub_trigger_rational_dict.keys():
                new_pred_triggers_dict[trigger_text] = new_sub_trigger_rational_dict[trigger_text]["trigger_event_type"]

            for biased_trigger_text in biased_triggers_text:
                if biased_trigger_text in new_pred_triggers_dict.keys():
                    pred_triggers_stage1_dict[biased_trigger_text] = new_pred_triggers_dict[biased_trigger_text]
                else:
                    pred_triggers_stage1_dict[biased_trigger_text] = args.event_type_set[0]  # 不存在置为空



            # 从 pred_triggers_stage1_dict 还原为 pred_triggers
            pred_triggers = []
            reasoning_explanations = []  # 这个也需要重组
            for trigger_text in pred_triggers_stage1_dict.keys():
                current_event_type = pred_triggers_stage1_dict[trigger_text]
                if current_event_type != args.event_type_set[0]:
                    pred_triggers.append({
                        "text": trigger_text,
                        "event_type": current_event_type,
                    })
            #         if trigger_text in biased_triggers_text and trigger_text in new_sub_trigger_rational_dict.keys():
            #             reasoning_explanations.append(new_sub_trigger_rational_dict[trigger_text]["sub_rational"])
            #         else:
            #             # 对于本来就没计划纠偏的，直接从 sub_trigger_rational_dict_stage1 取出
            #             reasoning_explanations.append(sub_trigger_rational_dict_stage1[trigger_text]["sub_rational"])
            # rational_after_feedback = ' '.join(reasoning_explanations)

            # if 'Prediction:' in rational_after_feedback:
            #     rational_after_feedback = rational_after_feedback.split('Prediction:')[0]

            if rational_after_feedback == '':
                rational_after_feedback = '[No Feedback Here]'

        if rational_after_feedback != '[No Feedback Here]':
            # if 1:

            print(f'--------样本序号：{idx}-------------', file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(prompt, file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(generated_text, file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'第一阶段预测的结果: {pred_triggers_stage1}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'True label: {test_example["triggers"]}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'初步icl更新后的预测 label: {pred_triggers}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print(f'Rational after Feedback: {rational_after_feedback}', file=response_record_filepath)
            print('########################', file=response_record_filepath)
            print('--------------------------------------', file=response_record_filepath)
            print('\n', file=response_record_filepath)

        test_example['rational_after_feedback'] = rational_after_feedback
        test_example['pred_triggers'] = pred_triggers
        # if rational_after_feedback != '[No Feedback Here]':
        #     test_example['biased_sub_trigger_rational_dict'] = split_triggers_rationals(
        #         triggers=pred_triggers,
        #         rationals_str=rational_after_feedback)
        # 这两个都没动
        test_example['biased_explanations'] = test_example['biased_explanations']
        test_example['biased_pred_triggers'] = test_example['biased_pred_triggers']

    return test_examples


def cal_metric(test_examples_after_stage2):
    num_pred = 0
    num_golden = 0
    num_true_micro_f1 = 0

    for idx, test_example in enumerate(test_examples_after_stage2):
        pred_triggers = test_example["pred_triggers"]
        golden_triggers = test_example["triggers"]

        num_pred += len(pred_triggers)
        num_golden += len(golden_triggers)

        pred_triggers_dict = {}
        for trigger in pred_triggers:
            pred_triggers_dict[trigger["text"]] = trigger["event_type"]
        golden_triggers_dict = {}
        for trigger in golden_triggers:
            golden_triggers_dict[trigger["text"]] = trigger["event_type"]
        golden_triggers_keys = golden_triggers_dict.keys()

        for text_key, event_value in zip(pred_triggers_dict.keys(), pred_triggers_dict.values()):
            if text_key in golden_triggers_keys:
                if golden_triggers_dict[text_key] == pred_triggers_dict[text_key]:
                    num_true_micro_f1 += 1

    precision = num_true_micro_f1 / num_pred
    recall = num_true_micro_f1 / num_golden
    micro_f1 = 2 * (precision * recall) / (precision + recall)

    # print(f'micro_f1  ↑: {micro_f1}', file=response_record_filepath)

    metric = {'micro_f1': micro_f1,
              'precision': precision,
              'recall': recall,
              'num_test_examples': len(test_examples),
              }
    # print(f'metric  ↑: {metric}', file=response_record_filepath)

    return metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', type=int, default=42, help='Set a random seed')
    parser.add_argument('--dataset', type=str, default='SemEval', help='Choose one dataset for experiments.')
    parser.add_argument('--num_test_examples', type=int, default=1000, help='Set the number of test examples')
    parser.add_argument('--k_shot', type=int, default=10, help='number of demo examples per class')
    parser.add_argument('--llm_type', type=str, default='Qwen3-4B-Instruct-2507-FP8', help='llm type ')

    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='max seq length, used for retriever truncation.')

    parser.add_argument('--demo_num', type=int, default=6, help='number of demonstration examples')

    parser.add_argument('--biasx_retriever_topk', type=int, default=6, help='number of demonstration examples')

    parser.add_argument('--retrieval_stage1_mode', type=str, default='retrieval', help='in [random, retrieval]')
    parser.add_argument('--retriever_plm', type=str, default='Retrieval/bert-base-uncased',
                        help='plm used for retrievers')

    parser.add_argument('--retrieval_stage2_mode', type=str, default='ed_contrastive_retriever',
                        help='in [ed_contrastive_retriever]')

    parser.add_argument('--do_stage1', action='store_true', help='内存限制，stage1和2分开做')
    parser.add_argument('--do_stage2', action='store_true', help='内存限制，stage1和2分开做')

    parser.add_argument('--do_local', action='store_true', help='网络限制，本地运行')


    parser.add_argument('--simcse_path', type=str, default='princeton-nlp/sup-simcse-bert-base-uncased',
                        help='the path for simcse pretrained LM')
    parser.add_argument('--run_id', type=int, default=0, help='第几次重复运行')


    # parser.add_argument('--retrieval_stage2_mode', type=str, default='random', help='retrieval_stage2_mode')

    args = parser.parse_args()

    set_seeds(args.seed)

    # 自动检测是否支持CUDA，并据此选择使用GPU还是CPU
    # args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    # args.gpu_id = 0  # 这里设置为0是因为在命令行中设置了可见gpu
    # print('***************** working on gpu id: ', args.gpu_id, ' *****************')
    # args.device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    # 默认测试文件为数据集对应的测试集，同时取某一个特定数字方便测试
    test_filepath = f'./data/{args.dataset}/test.json'
    test_examples = json.load(open(test_filepath, 'r', encoding='utf-8'))

    # test_examples = test_examples[: 10]
    test_examples = test_examples[: args.num_test_examples]
    # test_examples = random.sample(test_examples,k=args.num_test_examples)

    ######## 准备类别标签集合 #########################################
    label_set, label_dict = prepare_label_set(dataset=args.dataset)
    if args.dataset in ["SemEval", "TACRED"]:
        setattr(args, 'relation_type_set', label_set)
        setattr(args, 'relation_set_dict', label_dict)
        args.label_key = 'relation_type'
    elif args.dataset in ["ACE05", "ACE05_CN", "MAVEN","DuEE","WikiEvents"]:
        setattr(args, 'event_type_set', label_set)
        setattr(args, 'event_set_dict', label_dict)
        args.label_key = 'triggers'

    ################################################################

    ######## 准备prompt记录文件 #########################################
    response_record_dir = 'response_record_test_examples'
    if not os.path.exists(response_record_dir):
        os.mkdir(response_record_dir)

    results_record_path = f'{response_record_dir}/results_{str(args.k_shot)}_shot_{args.dataset}_{args.llm_type}.txt'
    results_record_filepath = open(results_record_path, encoding='utf-8', mode='a')

    ################################################################

    # 设置本实验中所用retriever所用到的tokenizer
    setattr(args, 'tokenizer', BertTokenizer.from_pretrained(args.retriever_plm))

    # analysis_function_plt(args=args)

    # assert 1==0

    if args.do_stage1:
        response_record_filepath = f'{response_record_dir}/stage1_{str(args.k_shot)}_shot_{args.dataset}_{args.llm_type}.txt'
        response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

        # 带有初始自我解释的示例文件路径，在此之前需要进行预先准备自我解释
        # 准备带解释的标注样本集
        labeled_samples_with_x_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_x.json'
        with open(labeled_samples_with_x_filepath, mode='r', encoding='utf-8') as f:
            labeled_samples_with_x = json.loads(f.read())

        prompts_stage1_temp_file = f'temp_files/run{args.run_id}_prompts_stage1_temp_file_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}.json'

        if (not args.do_local) or args.llm_type != 'gpt-3.5-turbo':
            ######## 训练好的检索器1 加载 #########################################
            emb_retriever_stage1 = load_stage1_retriever(args=args)
            ################################################################
            # 这里对每一个demo做prompt测试，以检测bias
            prompts_stage1, all_icl_ins_stage1_retrieved = prepare_prompts_for_stage1(args=args,
                                                                                      test_examples=test_examples,
                                                                                      labeled_samples_with_x=labeled_samples_with_x,
                                                                                      emb_retriever_stage1=emb_retriever_stage1)

            with open(prompts_stage1_temp_file, encoding='utf-8', mode='w') as f:
                json.dump(prompts_stage1, f, indent=2)

            all_icl_ins_stage1_retrieved_file = f'temp_files/run{args.run_id}_all_icl_ins_stage1_retrieved_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}.json'
            with open(all_icl_ins_stage1_retrieved_file, encoding='utf-8', mode='w') as f:
                json.dump(all_icl_ins_stage1_retrieved, f, indent=2)

            if args.llm_type == 'gpt-3.5-turbo':
                assert 1 == 0

        # do_local时，读取上面的获得的临时prompt
        with open(prompts_stage1_temp_file, encoding='utf-8', mode='r') as f:
            prompts_stage1 = json.loads(f.read())

        all_icl_ins_stage1_retrieved_file = f'temp_files/run{args.run_id}_all_icl_ins_stage1_retrieved_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}.json'
        with open(all_icl_ins_stage1_retrieved_file, encoding='utf-8', mode='r') as f:
            all_icl_ins_stage1_retrieved = json.loads(f.read())

        ################################################################
        # TODO: 还是要放在这里，不然检索占用显存，在50-shot用不了
        llm, sampling_params = prepare_llm(args=args)

        #############################################################################################
        #############################################################################################
        prompts_stage1 = templated_prompts(args=args,prompts=prompts_stage1)

        outputs_stage1 = llm.generate(prompts_stage1, sampling_params)

        test_examples_after_stage1, metric_stage1 = parse_outputs_stage1(args=args,
                                                                         outputs=outputs_stage1,
                                                                         test_examples=test_examples,
                                                                         response_record_filepath=response_record_filepath,
                                                                         all_icl_ins_stage1_retrieved=all_icl_ins_stage1_retrieved,
                                                                         )

        test_stage1_pred_filepath = f'temp_files/run{args.run_id}_test_stage1_temp_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}_{args.llm_type}.json'
        with open(test_stage1_pred_filepath, encoding='utf-8', mode='w') as f:
            json.dump(test_examples_after_stage1, f, indent=2)

        log_result(
            results_record_path,
            method=f"RawStage1-{args.retrieval_stage1_mode}",
            dataset=args.dataset,
            k_shot=args.k_shot,
            llm_type=args.llm_type,
            retrieval_stage1_mode=args.retrieval_stage1_mode,
            run_id=args.run_id,
            metric=metric_stage1,
            token_usage=summarize_token_usage(outputs=outputs_stage1),
        )

        # # print(f'Report at stage1. Metric: {metric_stage1}')
        # print(f'\033[34mReport at stage1. Metric: {metric_stage1}\033[0m')
        # print(f'Report at stage1. Metric: {metric_stage1}', file=results_record_filepath)

    elif args.do_stage2:
        response_record_filepath = f'{response_record_dir}/stage2_{str(args.k_shot)}_shot_{args.dataset}_{args.llm_type}.txt'
        response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

        # if 1:
        paired_intervention_data_filepath = \
            f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_paired_intervention_data.json'
        with open(paired_intervention_data_filepath, encoding='utf-8', mode='r') as f:
            paired_intervention_data = json.loads(f.read())
        _, cat_labels = format_pair_intervention_data(paired_intervention_data)
        setattr(args, 'cat_labels_num', len(cat_labels))

        # 基于阶段1的输出结果进行实验
        test_stage1_pred_filepath = f'temp_files/run{args.run_id}_test_stage1_temp_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}_{args.llm_type}.json'

        with open(test_stage1_pred_filepath, encoding='utf-8', mode='r') as f:
            test_examples_after_stage1 = json.loads(f.read())

        print(f'Report at stage2. Num of test examples after stage1: {len(test_examples_after_stage1)}')

        prompts_stage2_temp_file = f'temp_files/run{args.run_id}_prompts_stage2_temp_file_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}.json'

        if (not args.do_local) or args.llm_type != 'gpt-3.5-turbo':
            ######## 训练好的检索器2 加载 #########################################
            emb_retriever_stage2 = load_stage2_retriever(args=args)
            ################################################################
            # 注意，这里的test_examples中间是含有  biased_explanations
            prompts_stage2 = prepare_prompts_for_stage2(args=args,
                                                        test_examples=test_examples_after_stage1,
                                                        paired_intervention_data=paired_intervention_data,
                                                        emb_retriever_stage2=emb_retriever_stage2)
            # print(prompts_stage2[0])

            with open(prompts_stage2_temp_file, encoding='utf-8', mode='w') as f:
                json.dump(prompts_stage2, f, indent=2)

        # do_local时，读取上面的获得的临时prompt
        with open(prompts_stage2_temp_file, encoding='utf-8', mode='r') as f:
            prompts_with_biased_triggers_stage2 = json.loads(f.read())[:]

        prompts_stage2 = [item["prompt"] for item in prompts_with_biased_triggers_stage2]
        all_biased_triggers_stage2 = [item["biased_triggers"] for item in prompts_with_biased_triggers_stage2]

        # selected_samples_stage2 = [item["selected_samples_stage2"] for item in prompts_with_demos_stage2]

        # print(prompts_stage2)

        ################################################################
        # TODO: 还是要放在这里，不然检索占用显存，在50-shot用不了

        llm, sampling_params = prepare_llm(args=args)
        # emb_retriever_stage2 = load_stage2_retriever(args=args)

        #############################################################################################
        #############################################################################################
        test_stage2_pred_filepath = f'temp_files/run{args.run_id}_test_stage2_temp_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}.json'

        if (args.do_local and args.llm_type == 'gpt-3.5-turbo') or args.llm_type != 'gpt-3.5-turbo':
            prompts_stage2 = templated_prompts(args=args,prompts=prompts_stage2)
            outputs_stage2 = llm.generate(prompts_stage2, sampling_params)

            test_examples_after_stage2 = parse_outputs_stage2(args=args,
                                                              outputs=outputs_stage2,
                                                              test_examples=test_examples_after_stage1,
                                                              all_biased_triggers_stage2=all_biased_triggers_stage2,
                                                              response_record_filepath=response_record_filepath,
                                                              )

            with open(test_stage2_pred_filepath, encoding='utf-8', mode='w') as f:
                json.dump(test_examples_after_stage2, f, indent=2)

            metric = cal_metric(test_examples_after_stage2=test_examples_after_stage2)
            print(f'\033[34mReport at stage2 after weak supervision. Metric: {metric}\033[0m')
            # print(f'\033[34mReport at stage2 after weak supervision. Metric: {metric}\033[0m',file=results_record_filepath)
            # gpt时，本地跑到这里，然后把 test_stage2_pred_filepath 传到服务器
            if args.llm_type == 'gpt-3.5-turbo':
                assert 1 == 0

        elif ((not args.do_local) and args.llm_type == 'gpt-3.5-turbo') or args.llm_type != 'gpt-3.5-turbo':
            with open(test_stage2_pred_filepath, encoding='utf-8', mode='r') as f:
                test_examples_after_stage2 = json.loads(f.read())

        metric = cal_metric(test_examples_after_stage2=test_examples_after_stage2)
        final_test_stage2_pred_filepath = f'temp_files/run{args.run_id}_final_test_stage2_temp_{args.dataset}_{args.retrieval_stage1_mode}_{str(args.k_shot)}.json'
        with open(final_test_stage2_pred_filepath, encoding='utf-8', mode='w') as f:
            json.dump(test_examples_after_stage2, f, indent=2)



        log_result(
            results_record_path,
            method=f"OursStage2-{args.retrieval_stage1_mode}",
            dataset=args.dataset,
            k_shot=args.k_shot,
            llm_type=args.llm_type,
            retrieval_stage1_mode=args.retrieval_stage1_mode,
            biasx_retriever_topk=args.biasx_retriever_topk,
            run_id=args.run_id,
            metric=metric,
            token_usage=summarize_token_usage(outputs=outputs_stage2),
        )

        # print(f'Report at stage2 after weak supervision. Metric: {metric}')
        # print(f'Report at stage2 after weak supervision. Metric: {metric}', file=results_record_filepath)
        # print(f'dataset: {args.dataset}-k_shot: {args.k_shot}-retrieval_stage1_mode: '
        #       f'{args.retrieval_stage1_mode}-biasx_retriever_topk: {args.biasx_retriever_topk}',
        #     file=results_record_filepath)
        # print(f'-------------------------------------------------------', file=results_record_filepath)

