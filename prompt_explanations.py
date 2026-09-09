import argparse
import json
import os
import random
import re
import time

from tqdm import tqdm

import sys
import os

from utils import prepare_llm, prepare_label_set, templated_prompts
from prompt_tools import ChatGPT_Class, convert_triggers_to_formatted_str


def get_formed_key_phase(head_entity, tail_entity, given_sentence):
    real_start = min(head_entity["start_idx"], tail_entity["start_idx"])
    real_end = max(head_entity["end_idx"], tail_entity["end_idx"])
    key_phase = ' '.join(given_sentence[real_start:real_end + 1])
    formed_key_phase = f'In the given sentence, the key phrase "{key_phase}" implies that'
    return formed_key_phase


def get_prompt_for_RE_explanations(args, labeled_sample):
    if args.dataset in ['TACRED']:
        basic_RE_X_prompt = '''Instruction: Given a sentence, explain why there is certain relation between the head and tail entities in the sentence.
        Demonstration:
        <Start of Instance>
        Given Sentence: "Space X was founded by Musk."
        Head Entity: "Space X"
        Tail Entity: "Musk"
        Prior Knowledge: The relation type between "Space X" and "Musk" is "org:founded_by"
        Reasoning Explanations: In the given sentence, the key phrase "was founded by" implies that the company "Space X" was created by the person "Musk". Therefore, the head entity "Space X" serves as the "org" while the tail entity "Musk" servers as the "founded_by" person. 
        Prediction: Given the sentence, the relation between the head entity "Space X" and the tail entity "Musk" is "org:founded_by"
        <End of Instance>
        <Hint> 
        Please learn the demonstration and follow the instruction, output the explanations part of the new given instance. 
        Please end with <End of Instance> when complete the instance.
        <Hint>
        <Start of Instance>
        Given Sentence: "{given_sentence}"
        Head Entity: "{head_entity}"
        Tail Entity: "{tail_entity}"
        Prior Knowledge: The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
        Reasoning Explanations: {formed_key_phase}'''
    elif args.dataset in ['SemEval']:
        basic_RE_X_prompt = '''Instruction: Given a sentence, explain why there is certain relation between the head and tail entities in the sentence.
        Demonstration:
        <Start of Instance>
        Given Sentence: "The therapist treats the patient with a certain kind of manual therapy."
        Head Entity: "therapist"
        Tail Entity: "therapy"
        Prior Knowledge: The relation type between "therapist" and "therapy" is "Instrument-Agency"
        Reasoning Explanations: In the given sentence, the key phrase "treats the patient with" implies that the therapy is the tool employed by the therapist to treat the patient. Therefore, the head entity "therapy" serves as the "Instrument" while the tail entity "therapist" servers as the "Agency". 
        Prediction: Given the sentence, the relation between the head entity "therapy" and the tail entity "therapist" is "Instrument-Agency"
        <End of Instance>
        <Hint> 
        Please learn the demonstration and follow the instruction, output the explanations part of the new given instance. 
        Please end with <End of Instance> when complete the instance.
        <Hint>
        <Start of Instance>
        Given Sentence: "{given_sentence}"
        Head Entity: "{head_entity}"
        Tail Entity: "{tail_entity}"
        Prior Knowledge: The relation type between "{head_entity}" and "{tail_entity}" is "{relation_type}"
        Reasoning Explanations: {formed_key_phase}'''

    basic_RE_X_prompt = basic_RE_X_prompt.replace('\n        ', '\n')

    test_given_sentence = labeled_sample["sentence"]
    test_head_entity = labeled_sample["head_entity"]
    test_tail_entity = labeled_sample["tail_entity"]
    relation_type = labeled_sample["relation_type"]

    formed_key_phase = get_formed_key_phase(head_entity=test_head_entity, tail_entity=test_tail_entity,
                                            given_sentence=test_given_sentence)

    re_x_prompt = basic_RE_X_prompt.format(
        given_sentence=' '.join(test_given_sentence),
        head_entity=test_head_entity["span"],
        tail_entity=test_tail_entity["span"],
        relation_type=relation_type,
        formed_key_phase=formed_key_phase,
    )
    return re_x_prompt


def get_prompt_for_ED_explanations(args, labeled_sample):
    if args.dataset in ['ACE05', "ACE05_CN"]:
        basic_ED_X_prompt = '''Instruction: Given a sentence, explain why some certain words in the sentence are triggers of certain events.
        Demonstration:
        <Start of Instance>
        Given Sentence: "Davies is leaving to become chairman of the London School of Economics, one of the best-known parts of the University of London ."
        Prior Knowledge: The word "leaving" triggers a "Personnel:End-Position" event; The word "become" triggers a "Personnel:Start-Position" event; 
        Reasoning Explanations: The trigger word "leaving" implies that the person Davies has quited or leaved the current position, which implies a "Personnel:End-Position" event happened; The trigger word "become" implies that Davies is going to have a new position (chairman), which implies a "Personnel:Start-Position" event happened.
        Prediction: The word "leaving" is the trigger of a "Personnel:End-Position" event; The word "become" is the trigger of a "Personnel:Start-Position" event;
        <End of Instance>
        <Hint> 
        Please learn the demonstration and follow the instruction, output the Reasoning Explanations and Prediction parts of the new given instance. 
        Format requirement: 
            In Reasoning Explanations, write exactly one sentence per trigger word listed in Prior Knowledge, separated by a semicolon ";". 
            Each sentence must strictly follow the pattern: 
                'The trigger word "[trigger]" implies that [explanation], which implies a "[EventType]" event happened'
            where [trigger] is the exact trigger word verbatim, [explanation] describes the semantic evidence in the sentence, and [EventType] matches exactly the event type listed in Prior Knowledge.
            A trigger word must appear quoted only in its own sentence and must not appear quoted in any other sentence.
            The number of sentences in Reasoning Explanations must equal the number of trigger words listed in Prior Knowledge.
            Each sentence must end with a semicolon ";".
        <Hint>
        <Start of Instance>
        Given Sentence: "{given_sentence}"
        Prior Knowledge: {prior_knowledge}
        '''
    elif args.dataset in ['WikiEvents']:
        basic_ED_X_prompt = '''Instruction: Given a sentence, explain why some certain words in the sentence are triggers of certain events.
            Demonstration:
            <Start of Instance>
            Given Sentence: "According to Spiegel , Omar D . was detained in 2008 at the Cologne / Bonn airport after boarding a plane to Amsterdam , along with another Somali native , both suspected of planning to join the armed jihad . Both were released shortly after their arrest at that time ."
            Prior Knowledge: The word "detained" triggers a "Justice.ArrestJailDetain.Unspecified" event; The word "released" triggers a "Justice.ReleaseParole.Unspecified" event; The word "arrest" triggers a "Justice.ArrestJailDetain.Unspecified" event; 
            Reasoning Explanations: The trigger word "detained" implies that Omar D. was taken into custody by authorities, which implies a "Justice.ArrestJailDetain.Unspecified" event happened; The trigger word "released" implies that the detained individuals were set free from custody, which implies a "Justice.ReleaseParole.Unspecified" event happened; The trigger word "arrest" refers to the act of being taken into legal custody mentioned earlier, which implies a "Justice.ArrestJailDetain.Unspecified" event happened.
            Prediction: The word "detained" is the trigger of a "Justice.ArrestJailDetain.Unspecified" event; The word "released" is the trigger of a "Justice.ReleaseParole.Unspecified" event; The word "arrest" is the trigger of a "Justice.ArrestJailDetain.Unspecified" event;
            <End of Instance>
            <Hint> 
            Please learn the demonstration and follow the instruction, output the Reasoning Explanations and Prediction parts of the new given instance. 
            Format requirement: 
                In Reasoning Explanations, write exactly one sentence per trigger word listed in Prior Knowledge, separated by a semicolon ";". 
                Each sentence must strictly follow the pattern: 
                    'The trigger word "[trigger]" implies that [explanation], which implies a "[EventType]" event happened'
                where [trigger] is the exact trigger word verbatim, [explanation] describes the semantic evidence in the sentence, and [EventType] matches exactly the event type listed in Prior Knowledge.
                A trigger word must appear quoted only in its own sentence and must not appear quoted in any other sentence.
                The number of sentences in Reasoning Explanations must equal the number of trigger words listed in Prior Knowledge.
                Each sentence must end with a semicolon ";".
            <Hint>
            <Start of Instance>
            Given Sentence: "{given_sentence}"
            Prior Knowledge: {prior_knowledge}
            '''
    elif args.dataset in ['MAVEN']:
        basic_ED_X_prompt = '''Instruction: Given a sentence, explain why some certain words in the sentence are triggers of certain events.
            Demonstration:
            <Start of Instance>
            Given Sentence: "The preachers of the League sanctioned regicide , to avenge the murder of Guise ."
            Prior Knowledge: The word "murder" triggers a "Killing" event; The word "avenge" triggers a "Revenge" event; The word "sanctioned" triggers a "Revenge" event; 
            Reasoning Explanations: The trigger word "murder" implies that Guise was killed, which implies a "Killing" event happened; The trigger word "avenge" implies that an act is being carried out to retaliate for the murder of Guise, which implies a "Revenge" event happened; The trigger word "sanctioned" implies that the League's preachers officially approved retaliatory action (regicide) in response to the murder, which implies a "Revenge" event happened.
            Prediction: The word "murder" is the trigger of a "Killing" event; The word "avenge" is the trigger of a "Revenge" event; The word "sanctioned" is the trigger of a "Revenge" event;
            <End of Instance>
            <Hint> 
            Please learn the demonstration and follow the instruction, output the Reasoning Explanations and Prediction parts of the new given instance. 
            Format requirement: 
                In Reasoning Explanations, write exactly one sentence per trigger word listed in Prior Knowledge, separated by a semicolon ";". 
                Each sentence must strictly follow the pattern: 
                    'The trigger word "[trigger]" implies that [explanation], which implies a "[EventType]" event happened'
                where [trigger] is the exact trigger word verbatim, [explanation] describes the semantic evidence in the sentence, and [EventType] matches exactly the event type listed in Prior Knowledge.
                A trigger word must appear quoted only in its own sentence and must not appear quoted in any other sentence.
                The number of sentences in Reasoning Explanations must equal the number of trigger words listed in Prior Knowledge.
                Each sentence must end with a semicolon ";".
            <Hint>
            <Start of Instance>
            Given Sentence: "{given_sentence}"
            Prior Knowledge: {prior_knowledge}
            '''

    elif args.dataset in ["DuEE"]:
        raise "You need to add a template here!!!"


    basic_ED_X_prompt = basic_ED_X_prompt.replace('\n        ', '\n')

    test_given_sentence = labeled_sample["sentence"]
    labeled_triggers = labeled_sample["triggers"]

    prior_knowledge, _ = convert_triggers_to_formatted_str(triggers=labeled_triggers)

    ed_x_prompt = basic_ED_X_prompt.format(
        given_sentence=' '.join(test_given_sentence),
        prior_knowledge=prior_knowledge
    )
    return ed_x_prompt


def prepare_prompts_for_explanations(args, labeled_samples):
    if args.dataset in ["SemEval", "TACRED"]:
        re_x_prompts = []
        for idx, labeled_sample in enumerate(tqdm(labeled_samples)):
            re_x_prompt = get_prompt_for_RE_explanations(
                args=args,
                labeled_sample=labeled_sample,
            )
            re_x_prompts.append(re_x_prompt)
        return re_x_prompts
    elif args.dataset in ["ACE05", "ACE05_CN", "MAVEN","DuEE","WikiEvents"]:
        ed_x_prompts = []
        for idx, labeled_sample in enumerate(tqdm(labeled_samples)):
            ed_x_prompt = get_prompt_for_ED_explanations(
                args=args,
                labeled_sample=labeled_sample,
            )
            ed_x_prompts.append(ed_x_prompt)
        return ed_x_prompts

def parse_explanation_from_text(generated_text):
    # Step 1: 截断到<End of Instance>之前
    text = generated_text.split('<End of Instance>')[0].strip()

    # Step 2: 提取Explanations之后的部分，兼容"Reasoning Explanations:"和"Explanations:"
    for marker in ['Reasoning Explanations:', 'Explanations:']:
        if marker in text:
            text = text.split(marker)[1].strip()
            break

    # Step 3: 截断到Prediction之前
    if 'Prediction' in text:
        text = text.split('Prediction')[0].strip()
    else:
        print(f"[Warning] 'Prediction' not found in generated text: {generated_text[:100]}")

    parsed_explanations = text.replace('\n', ' ').strip()
    return parsed_explanations

# def parse_explanation_from_text(generated_text):
#     parsed_explanations = generated_text
#
#     # 第一步，找到<End of Instance>并只保留这之前的部分
#     generated_text = generated_text.split('<End of Instance>')[0]
#
#     if 'Explanations:' in generated_text:
#         generated_text = generated_text.split('Explanations:')[1]
#
#     if 'Prediction' in generated_text:
#         parsed_explanations = generated_text.split('Prediction')[0]
#
#     parsed_explanations = parsed_explanations.replace('\n', '')
#     return parsed_explanations


def parse_key_phases_from_explanation(explanations):
    key_phases = None

    if 'phrase' not in explanations:
        pass
    else:
        phrase_sentence = explanations.split('phrase')[1]
        # print(phrase_sentence)

        matches = re.findall(r'\"(.*?)\"', phrase_sentence)

        if len(matches) > 0:
            key_phases = matches[0]
        else:
            key_phases = None

    return key_phases


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--k_shot', type=int, default=10, help='number of examples per class')
    parser.add_argument('--dataset', type=str, default='SemEval', help='name of the dataset')
    parser.add_argument('--llm_type', type=str, default='llama2_7b_chat', help='llm type ')

    args = parser.parse_args()

    # llm_type = args.llm_type
    # k_shot = args.k_shot
    # dataset = args.dataset

    with open(f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train.json') as f:
        labeled_samples = json.loads(f.read())

    ################################################################
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
    response_record_dir = 'llm_generated_re_x'
    if not os.path.exists(response_record_dir):
        os.mkdir(response_record_dir)
    response_record_filepath = \
        f'{response_record_dir}/{args.dataset}_{str(args.k_shot)}_shot_response_record_{args.llm_type}_x.txt'
    response_record_filepath = open(response_record_filepath, encoding='utf-8', mode='w')

    prompts = prepare_prompts_for_explanations(args=args, labeled_samples=labeled_samples)

    prompts=templated_prompts(args=args,prompts=prompts)

    llm, sampling_params = prepare_llm(args=args)


    outputs = llm.generate(prompts, sampling_params)

    demonstrations_with_x = []

    for output, labeled_sample in zip(outputs, labeled_samples):
        prompt = output.prompt  # 获取原始的输入提示
        generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        explanations = parse_explanation_from_text(generated_text=generated_text)
        if args.dataset in ["SemEval", "TACRED"]:
            formed_key_phase = get_formed_key_phase(head_entity=labeled_sample["head_entity"],
                                                    tail_entity=labeled_sample["tail_entity"],
                                                    given_sentence=labeled_sample["sentence"])
            explanations = formed_key_phase + explanations
        elif args.dataset in ["ACE05", "ACE05_CN", "MAVEN", "DuEE","WikiEvents"]:
            explanations = explanations

        print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(generated_text, file=response_record_filepath)

        print('########################', file=response_record_filepath)
        print(labeled_sample[f"{args.label_key}"], file=response_record_filepath)
        print('########################', file=response_record_filepath)

        print('########################', file=response_record_filepath)
        print(explanations, file=response_record_filepath)
        print('########################', file=response_record_filepath)

        print('\n', file=response_record_filepath)

        labeled_sample['explanations'] = explanations

        labeled_sample['sentence_str'] = ' '.join(labeled_sample['sentence'])

        demonstrations_with_x.append(labeled_sample)

    demonstrations_with_x_file = f'./data/{args.dataset}/sampled_{str(args.k_shot)}_shot_train_with_{args.llm_type}_x.json'
    with open(demonstrations_with_x_file, encoding='utf-8', mode='w') as f:
        json.dump(demonstrations_with_x, f, indent=2, ensure_ascii=False)
