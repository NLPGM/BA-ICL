import argparse
import copy
import json
import os
import random
import re
import time

from tqdm import tqdm

import sys
import os

from utils import prepare_label_set, prepare_llm, templated_prompts
from prompt_tools import ChatGPT_Class, construct_original_scm_prompt, \
    construct_hard_intervention_prompt, construct_check_intervention_prompt, parse_original_scm_output, \
    parse_hard_intervention_output, parse_soft_intervention_output, get_formed_key_phase, split_triggers_rationals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--k_shot', type=int, default=10, help='number of examples per class')
    parser.add_argument('--dataset', type=str, default='SemEval', help='name of the dataset')
    parser.add_argument('--llm_type', type=str, default='llama2_7b_chat', help='llm type ')
    parser.add_argument('--supervisor_ablation_mode', type=str, default='None', help='None; do_ablationXXX')

    args = parser.parse_args()

    # 准备带解释的标注样本集
    labeled_samples_with_x_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_x.json'
    with open(labeled_samples_with_x_filepath, mode='r', encoding='utf-8') as f:
        labeled_samples_with_x = json.loads(f.read())[:]

    # if len(labeled_samples_with_x)>50000:
    #     labeled_samples_with_x = labeled_samples_with_x[:int(0.1*len(labeled_samples_with_x))]

    ################################################################
    ######## 准备类别标签集合 #########################################
    label_set, label_dict = prepare_label_set(dataset=args.dataset)
    if args.dataset in ["SemEval", "TACRED"]:
        setattr(args, 'relation_type_set', label_set)
        setattr(args, 'relation_set_dict', label_dict)
        args.label_key = 'relation_type'
    elif args.dataset in ["ACE05", "ACE05_CN","MAVEN","DuEE","WikiEvents"]:
        setattr(args, 'event_type_set', label_set)
        setattr(args, 'event_set_dict', label_dict)
        args.label_key = 'triggers'

    ################################################################

    # 准备记录文件
    response_record_dir = 'llm_generated_do_intervention'
    if not os.path.exists(response_record_dir):
        os.mkdir(response_record_dir)
    response_record_filepath = \
        f'{response_record_dir}/{args.dataset}_{str(args.k_shot)}_shot_record_{args.llm_type}_do_intervention.txt'
    response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

    # 准备llm
    llm, sampling_params = prepare_llm(args=args)

    # # 统计标注样本的标签分布
    # label_samples_dict = {}
    # # 给标注样本附上独特的id；如：“l-1”表示“标注样本1”；
    # for item_idx, item in tqdm(enumerate(labeled_samples_with_x),total=len(labeled_samples_with_x)):
    #     item_sample_id = f'l-{item_idx}'
    #     item['sample_id'] = item_sample_id
    #     item_label = item['relation_type']
    #     if item_label not in label_samples_dict:
    #         label_samples_dict[item_label] = [item_sample_id]
    #     else:
    #         label_samples_dict[item_label].append(item_sample_id)

    # 做一个反向的sample_id到sample的映射表
    labeled_samples_with_x_dict = {}
    # 给每个标注样本新增一个属性，id_list_with_same_label；表示具有同标签，但是不是同一个样本的列表。
    for item_idx, item in tqdm(enumerate(labeled_samples_with_x), total=len(labeled_samples_with_x)):
        item_sample_id = f'l-{item_idx}'

        # item_label = item['relation_type']
        # id_list_with_same_label = label_samples_dict[item_label]
        # tmp_id_list_with_same_label = copy.deepcopy(id_list_with_same_label)
        # 去掉自己的id
        # tmp_id_list_with_same_label.remove(item_sample_id)
        # item['id_list_with_same_label'] = tmp_id_list_with_same_label

        labeled_samples_with_x_dict[item_sample_id] = item

    # 接下来，对每个样本进行original_scm的生成、do_hard_intervention、do_soft_intervention（及检查）；
    # 并把最终的paired explanations保存至特定文件；
    # 注意pair-explanation的关键信息保留，头尾实体、关系类型、原句子都要做保留
    # 过程中保留生成的过程；

    selected_icl_list = []
    original_scm_prompt_list = []
    hard_intervention_prompt_list = []

    num_select_other_id = 10
    num_select_id_list_with_same_label = 0

    for item_idx, labeled_sample in tqdm(enumerate(labeled_samples_with_x),
                                         desc='Prepare prompts for original_scm and hard_intervention',
                                         total=len(labeled_samples_with_x)):
        item_id = f'l-{item_idx}'

        other_id_list = [f'l-{i}' for i in range(len(labeled_samples_with_x))]
        other_id_list.remove(item_id)
        selected_id_list = random.sample(other_id_list, k=num_select_other_id)
        # selected_id_list = selected_id_list + random.sample(id_list_with_same_label, k=num_select_id_list_with_same_label)

        for selected_id in selected_id_list:
            icl_sample = labeled_samples_with_x_dict[selected_id]
            selected_icl_list.append(icl_sample)

            original_scm_prompt = construct_original_scm_prompt(args=args,
                                                                sample=labeled_sample,
                                                                icl_samples=[icl_sample])

            hard_intervention_prompt = construct_hard_intervention_prompt(args=args,
                                                                          sample=labeled_sample,
                                                                          icl_samples=[icl_sample])

            original_scm_prompt_list.append(original_scm_prompt)
            hard_intervention_prompt_list.append(hard_intervention_prompt)

    all_prompts = original_scm_prompt_list + hard_intervention_prompt_list
    all_prompts=templated_prompts(args=args,prompts=all_prompts)

    all_outputs = llm.generate(all_prompts, sampling_params)
    # 计算每个部分的长度
    part_length = len(original_scm_prompt_list)
    # 分割列表到三个部分
    original_scm_outputs = all_outputs[:part_length]
    hard_intervention_outputs = all_outputs[part_length:2 * part_length]

    original_scm_list = []
    hard_intervention_x_list = []

    check_original_scm_prompt_list = []
    check_hard_intervention_prompt_list = []

    repeated_labeled_samples_with_x = []
    for item in labeled_samples_with_x:
        repeated_labeled_samples_with_x.extend([item] * (num_select_other_id + num_select_id_list_with_same_label))

    for item_idx, (labeled_sample,
                   selected_icl,
                   original_scm_output,
                   hard_intervention_output) in enumerate(zip(repeated_labeled_samples_with_x,
                                                              selected_icl_list,
                                                              original_scm_outputs,
                                                              hard_intervention_outputs)):
        # sample_label = labeled_sample['relation_type']

        original_scm_prompt = original_scm_output.prompt  # 获取原始的输入提示
        original_scm_generated_text = original_scm_output.outputs[0].text  # 从输出对象中获取生成的文本
        hard_intervention_prompt = hard_intervention_output.prompt  # 获取原始的输入提示
        hard_intervention_generated_text = hard_intervention_output.outputs[0].text  # 从输出对象中获取生成的文本

        original_scm = parse_original_scm_output(args=args,
                                                 output=original_scm_generated_text)
        original_scm_list.append(original_scm)

        # 对于original_scm也要进行检查，以免出现应该是无偏的，被标记为有偏的
        soft_intervention_prompt = construct_check_intervention_prompt(args=args,
                                                                       sample=labeled_sample,
                                                                       intervention_x=original_scm["x"],
                                                                       icl_samples=[selected_icl], )
        check_original_scm_prompt_list.append(soft_intervention_prompt)

        hard_intervention_x = parse_hard_intervention_output(args=args,
                                                             output=hard_intervention_generated_text)
        hard_intervention_x_list.append(hard_intervention_x)

        soft_intervention_prompt = construct_check_intervention_prompt(args=args,
                                                                       sample=labeled_sample,
                                                                       intervention_x=hard_intervention_x,
                                                                       icl_samples=[selected_icl], )
        check_hard_intervention_prompt_list.append(soft_intervention_prompt)

        print('%%%%%%%%%%%%%%%%%%%%%% original_scm prompt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(original_scm_prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% original_scm generated_text %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(original_scm_generated_text, file=response_record_filepath)
        print(original_scm["pred_label"], file=response_record_filepath)

        print('\n', file=response_record_filepath)

        print('%%%%%%%%%%%%%%%%%%%%%% hard_intervention prompt %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(hard_intervention_prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% hard_intervention generated_text %%%%%%%%%%%%%%%%%%%%%%',
              file=response_record_filepath)
        print(hard_intervention_generated_text, file=response_record_filepath)
        print('\n', file=response_record_filepath)
        print('================================================================================\n'
              '================================================================================',
              file=response_record_filepath)

    check_prompt_list = check_original_scm_prompt_list + check_hard_intervention_prompt_list
    check_prompt_list=templated_prompts(args=args,prompts=check_prompt_list)

    check_outputs = llm.generate(check_prompt_list, sampling_params)
    check_original_scm_outputs = check_outputs[:len(check_original_scm_prompt_list)]
    check_hard_intervention_outputs = check_outputs[len(check_original_scm_prompt_list):]


    for MODE in ["None","do_ablationNegPairs", "do_ablationAllPairs"]:
        args.supervisor_ablation_mode = MODE


        paired_intervention_data = []
        for item_idx, (labeled_sample,
                       original_scm,
                       hard_intervention_x,
                       check_original_scm_output,
                       check_hard_intervention_output) in enumerate(zip(repeated_labeled_samples_with_x,
                                                                        original_scm_list,
                                                                        hard_intervention_x_list,
                                                                        check_original_scm_outputs,
                                                                        check_hard_intervention_outputs)):

            check_original_scm_prompt = check_original_scm_output.prompt  # 获取原始的输入提示
            check_original_scm_generated_text = check_original_scm_output.outputs[0].text  # 从输出对象中获取生成的文本
            print('%%%%%%%%%%%%%%%%%%%%%% check_original_scm prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
            print(check_original_scm_prompt, file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% check_original_scm generated_text %%%%%%%%%%%%%%%%%%%%%%',
                  file=response_record_filepath)
            print(check_original_scm_generated_text, file=response_record_filepath)
            print('\n', file=response_record_filepath)
            print('================================================================================\n'
                  '================================================================================',
                  file=response_record_filepath)

            check_hard_intervention_prompt = check_hard_intervention_output.prompt  # 获取原始的输入提示
            check_hard_intervention_generated_text = check_hard_intervention_output.outputs[0].text  # 从输出对象中获取生成的文本
            print('%%%%%%%%%%%%%%%%%%%%%% check_hard_intervention prompt %%%%%%%%%%%%%%%%%%%%%%',
                  file=response_record_filepath)
            print(check_hard_intervention_prompt, file=response_record_filepath)
            print('%%%%%%%%%%%%%%%%%%%%%% check_hard_intervention generated_text %%%%%%%%%%%%%%%%%%%%%%',
                  file=response_record_filepath)
            print(check_hard_intervention_generated_text, file=response_record_filepath)
            print('\n', file=response_record_filepath)
            print('================================================================================\n'
                  '================================================================================',
                  file=response_record_filepath)

            check_original_scm_pred_triggers = parse_soft_intervention_output(args=args,
                                                                              output=check_original_scm_generated_text)

            original_scm["pred_label"] = check_original_scm_pred_triggers  # 注意这里是为了保证 解释和预测标签 的一致性

            check_hard_intervention_pred_triggers = parse_soft_intervention_output(args=args,
                                                                                   output=check_hard_intervention_generated_text)

            golden_triggers = labeled_sample["triggers"]

            golden_triggers_dict = {}
            for trigger in golden_triggers:
                golden_triggers_dict[trigger["text"]] = trigger["event_type"]
            golden_triggers_keys = golden_triggers_dict.keys()

            pred_triggers_dict = {}
            for trigger in check_hard_intervention_pred_triggers:
                pred_triggers_dict[trigger["text"]] = trigger["event_type"]

            # flag 为 True表示所有的 pred_triggers 都是对的（不保证召回率为1，但保证精确率为1）
            flag = True
            for text_key, event_value in zip(pred_triggers_dict.keys(), pred_triggers_dict.values()):
                if text_key not in golden_triggers_keys:
                    flag = False
                else:
                    if golden_triggers_dict[text_key] != pred_triggers_dict[text_key]:
                        flag = False


            sentence_str = ' '.join(labeled_sample["sentence"])
            # 这里还是保持严格的一致，才能保证 sub hard_intervention_x 的有效性
            if flag:
                # 此时表示软干预成功->硬干预获得的结果是无偏的

                # 在形成pair数据时，要对 sub rational单独处理
                original_scm_rationals = original_scm["x"]
                original_scm_triggers = original_scm["pred_label"]
                original_scm_sub_trigger_rational_dict = split_triggers_rationals(
                    triggers=original_scm_triggers,
                    rationals_str=original_scm_rationals)

                hard_intervention_rationals = hard_intervention_x
                hard_intervention_pred_triggers = check_hard_intervention_pred_triggers
                hard_intervention_sub_trigger_rational_dict = split_triggers_rationals(
                    triggers=hard_intervention_pred_triggers,
                    rationals_str=hard_intervention_rationals)

                pred_triggers_text = original_scm_sub_trigger_rational_dict.keys()
                golden_triggers_text = hard_intervention_sub_trigger_rational_dict.keys()

                for trigger_text, trigger_info in zip(
                        original_scm_sub_trigger_rational_dict.keys(),
                        original_scm_sub_trigger_rational_dict.values()):
                    pred_event_type=trigger_info["trigger_event_type"]
                    pred_sub_rational=trigger_info["sub_rational"]

                    if trigger_text in golden_triggers_text:
                        # 说明span对了，但是具体的内容预测错了
                        # 找到对应的真实值
                        golden_event_type = hard_intervention_sub_trigger_rational_dict[trigger_text]["trigger_event_type"]
                        golden_sub_rational = hard_intervention_sub_trigger_rational_dict[trigger_text]["sub_rational"]

                        instance={
                            "sentence_str": sentence_str,
                            "triggers": golden_triggers,
                            "explanations": hard_intervention_rationals,

                            "golden_event_type": golden_event_type,
                            "golden_sub_rational": golden_sub_rational,
                            "pred_event_type": pred_event_type,
                            "pred_sub_rational": pred_sub_rational,
                        }

                        if args.supervisor_ablation_mode in ["do_ablationAllPairs"]:
                            continue
                        elif args.supervisor_ablation_mode in ["do_ablationNegPairs"]:
                            if golden_event_type != pred_event_type:
                                continue
                            else:
                                paired_intervention_data.append(instance)
                        else:
                            paired_intervention_data.append(instance)

                    else:
                        golden_event_type = args.event_type_set[0]  # 表示为Other
                        golden_sub_rational = "None"
                        # 说明是过度抽取的，本身应该是无事件类型的
                        instance = {
                                "sentence_str": sentence_str,
                                "triggers": golden_triggers,
                                "explanations": hard_intervention_rationals,

                                "golden_event_type": golden_event_type,
                                "golden_sub_rational": golden_sub_rational,
                                "pred_event_type": pred_event_type,
                                "pred_sub_rational": pred_sub_rational,
                        }
                        if args.supervisor_ablation_mode in ["do_ablationAllPairs"]:
                            continue
                        elif args.supervisor_ablation_mode in ["do_ablationNegPairs"]:
                            if golden_event_type!=pred_event_type:
                                continue
                            else:
                                paired_intervention_data.append(instance)
                        else:
                            paired_intervention_data.append(instance)

                for trigger_text, trigger_info in zip(
                        hard_intervention_sub_trigger_rational_dict.keys(),
                        hard_intervention_sub_trigger_rational_dict.values()):
                    if trigger_text not in pred_triggers_text:
                        # 对于没有召回的这部分 golden sub triggers 也要保留为pair数据
                        golden_event_type=trigger_info["trigger_event_type"]
                        golden_sub_rational=trigger_info["sub_rational"]

                        instance={
                            "sentence_str": sentence_str,
                            "triggers": golden_triggers,
                            "explanations": hard_intervention_rationals,

                            "golden_event_type": golden_event_type,
                            "golden_sub_rational": golden_sub_rational,
                            "pred_event_type": golden_event_type,  # 这里是故意的
                            "pred_sub_rational": golden_sub_rational,
                        }

                        if args.supervisor_ablation_mode in ["do_ablationAllPairs"]:
                            continue
                        else:
                            paired_intervention_data.append(instance)




        if args.supervisor_ablation_mode=="None":
            paired_intervention_data_filepath = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_paired_intervention_data.json'
        else:
            paired_intervention_data_filepath = f'./data/{args.dataset}/{args.supervisor_ablation_mode}_sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_paired_intervention_data.json'

        with open(paired_intervention_data_filepath, encoding='utf-8', mode='w') as f:
            json.dump(paired_intervention_data, f, indent=2, ensure_ascii=False)
