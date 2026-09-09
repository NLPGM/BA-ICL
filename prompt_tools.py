import re

from openai import OpenAI
from tqdm import tqdm


def parse_reasoning_explanations_from_text(args, generated_text):
    if '<End of Instance>' in generated_text:
        generated_text = generated_text.split('<End of Instance>')[0]

    if 'Reasoning Explanations:' in generated_text:
        reasoning_explanations = generated_text.split('Reasoning Explanations:')[1]
    else:
        if 'Reasoning Explanation:' in generated_text:
            reasoning_explanations = generated_text.split('Reasoning Explanation:')[1]

        else:
            reasoning_explanations = ''

    reasoning_explanations = reasoning_explanations.split('Prediction')[0].split('Correct')[0]
    reasoning_explanations = reasoning_explanations.replace('\n', '')

    if 'Prediction' in generated_text:
        text_for_match = generated_text.split('Prediction')[1]
    else:
        text_for_match = generated_text

    pattern = r"\"(?P<span_text>[^']*)\" is the trigger of a \"(?P<event_type>[^']*)\""

    # 找到所有匹配

    pred_triggers = []

    splits_text_for_match = text_for_match.split(';')
    for split_text_for_match in splits_text_for_match:
        # 找到所有匹配
        matches = re.findall(pattern, split_text_for_match)
        if len(matches) > 0:
            span_text, event_type = matches[0]
            if event_type not in args.event_type_set:
                # 预测成不在候选列表的不予保留
                continue

            if event_type == args.event_type_set[0]:
                # 预测成其他的也不予保留
                continue
            # print(span_text, event_type)
            pred_triggers.append(
                {
                    "text": span_text,
                    "event_type": event_type,
                }
            )

    return pred_triggers, reasoning_explanations


def get_formed_key_phase(head_entity, tail_entity, given_sentence):
    real_start = min(head_entity["start_idx"], tail_entity["start_idx"])
    real_end = max(head_entity["end_idx"], tail_entity["end_idx"])
    key_phase = ' '.join(given_sentence[real_start:real_end + 1])
    formed_key_phase = f'In the given sentence, the key phrase "{key_phase}" implies that'
    return formed_key_phase


def parse_outputs_stage1(args, outputs, test_examples, response_record_filepath, all_icl_ins_stage1_retrieved):

    num_true_micro_f1 = 0
    num_pred = 0
    num_golden = 0

    for idx, (output, test_example, icl_ins_stage1_retrieved) in enumerate(
            zip(outputs, test_examples, all_icl_ins_stage1_retrieved)):
        golden_triggers = test_example["triggers"]

        prompt = output.prompt  # 获取原始的输入提示
        generated_text = output.outputs[0].text  # 从输出对象中获取生成的文本

        pred_triggers, reasoning_explanations = parse_reasoning_explanations_from_text(args=args, generated_text=generated_text)

        num_pred+=len(pred_triggers)
        num_golden+=len(golden_triggers)


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
                    num_true_micro_f1+=1

        print(f'--------样本序号：{idx}-------------', file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(prompt, file=response_record_filepath)
        print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%', file=response_record_filepath)
        print(generated_text, file=response_record_filepath)
        print('########################', file=response_record_filepath)
        print(f'Golden triggers: {golden_triggers}', file=response_record_filepath)
        print('########################', file=response_record_filepath)

        print('########################', file=response_record_filepath)
        print(f'Bias pred triggers: {pred_triggers}', file=response_record_filepath)
        print(f'Bias explanations: {reasoning_explanations}', file=response_record_filepath)
        print('########################', file=response_record_filepath)
        print('--------------------------------------', file=response_record_filepath)
        print('\n', file=response_record_filepath)


        # print(f'--------样本序号：{idx}-------------')
        # print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%')
        # print(prompt)
        # print('%%%%%%%%%%%%%%%%%%%%%% prompt %%%%%%%%%%%%%%%%%%%%%%')
        # print(generated_text)
        # print('########################')
        # print(f'Golden triggers: {golden_triggers}')
        # print('########################')
        #
        # print('########################')
        # print(f'Bias pred triggers: {pred_triggers}')
        # print(f'Bias explanations: {reasoning_explanations}')
        # print('########################')
        # print('--------------------------------------')
        # print('\n')

        biased_sub_trigger_rational_dict=split_triggers_rationals(triggers=pred_triggers, rationals_str=reasoning_explanations)
        test_example['biased_explanations'] = reasoning_explanations
        test_example['biased_pred_triggers'] = pred_triggers
        test_example['biased_sub_trigger_rational_dict'] = biased_sub_trigger_rational_dict


        test_example['icl_ins_stage1_retrieved'] = icl_ins_stage1_retrieved


    precision = num_true_micro_f1 / num_pred
    recall = num_true_micro_f1 / num_golden
    micro_f1 = 2 * (precision * recall) / (precision + recall)

    metric = {'micro_f1': micro_f1,
              'precision': precision,
              'recall': recall,
              'num_test_examples': len(test_examples)}
    print(f'metric of stage1: {metric}', file=response_record_filepath)

    return test_examples, metric


# def split_triggers_rationals(triggers, rationals_str):
#     sub_trigger_rational_dict = {}
#
#     # 用 findall 直接提取每个以 trigger word 开头的完整句子
#     # 匹配从句子开头到下一个同类句子开头之前的所有内容
#     pattern = r'(?:The |the )?trigger word\s+"[^"]+".+?(?=(?:The |the )?trigger word\s+"[^"]+"|$)'
#     segments = [s.strip() for s in re.findall(pattern, rationals_str, re.DOTALL) if s.strip()]
#
#     for trigger in triggers:
#         trigger_text = trigger["text"]
#         expected_re = re.compile(
#             r'^(?:The |the )?trigger word\s+"' + re.escape(trigger_text) + r'"'
#         )
#         matched = [seg for seg in segments if expected_re.match(seg)]
#
#         if len(matched) == 0:
#             print(f'[warn] no rational matched for trigger "{trigger_text}"')
#             continue
#         if len(matched) > 1:
#             print(f'[warn] multiple rationals matched for trigger "{trigger_text}", using the first one')
#
#         sub_trigger_rational_dict[trigger_text] = {
#             "trigger_text": trigger_text,
#             "trigger_event_type": trigger["event_type"],
#             "sub_rational": matched[0],
#         }
#
#     return sub_trigger_rational_dict

def split_triggers_rationals(triggers, rationals_str):
    # rationals="""The trigger word "hit" implies that an attack or violent action was taken against a specific location, which implies a "Conflict:Attack" event happened; The trigger word "dead" implies that at least four people were killed in the attack, which implies a "Conflict:Demonstrate" event happened; The trigger word "soldiers" implies that the attack was carried out by military personnel, which implies a "Conflict:Attack" event happened; The trigger word "journalists" implies that at least two journalists were killed in the attack, which implies a "Conflict:Demonstrate" event happened."""
    # triggers="[{'text': 'hit', 'event_type': 'Conflict:Attack'}, {'text': 'dead', 'event_type': 'Conflict:Demonstrate'}]"
    sub_trigger_rational_dict={}

    # 希望找到预测triggers中每个trigger对应的rationals中的子序列；注意对rationals切分时可以按照";"字符，在匹配到对应的trigger是要注意'text'字段的value在对应地子序列中""以出现
    split_rationals = rationals_str.split(";")

    for trigger in triggers:
        for rational in split_rationals:
            if f'"{trigger["text"]}"'in rational:
                sub_trigger_rational_dict[trigger['text']] = {
                    "trigger_text": trigger['text'],
                    "trigger_event_type": trigger['event_type'],
                    "sub_rational": rational,
                }

    return sub_trigger_rational_dict

class ChatGPT_Class():
    def __init__(self, llm_type):
        super(ChatGPT_Class, self).__init__()
        self.key = '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'
        self.client = OpenAI(api_key=self.key)
        self.llm_type = llm_type

    def generate(self, prompts, sampling_params):
        outputs = []
        for prompt in tqdm(prompts[:], desc='Generating using GPT-3.5'):
            if prompt in ["[No Feedback Here]", ]:
                output_dict = {
                    "prompt": prompt,
                    "outputs": [
                        {
                            "text": 'None'
                        }
                    ]
                }
                # 现在我们在创建 Output 对象时传递 outputs 列表
                output_object = Output(prompt=output_dict["prompt"], outputs=output_dict["outputs"])
                outputs.append(output_object)
                continue

            completion = self.client.chat.completions.create(
                model=self.llm_type,
                messages=[
                    {"role": "system", "content": "You are a text completion assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=sampling_params['temperature'],
                n=1,
            )
            generated_text = completion.choices[0].message.content

            print(prompt)
            print(generated_text)

            output_dict = {
                "prompt": prompt,
                "outputs": [
                    {
                        "text": generated_text
                    }
                ]
            }

            # 现在我们在创建 Output 对象时传递 outputs 列表
            output_object = Output(prompt=output_dict["prompt"], outputs=output_dict["outputs"])

            outputs.append(output_object)
        return outputs


class OutputText:
    def __init__(self, text):
        self.text = text


class Output:
    def __init__(self, prompt, outputs):
        self.prompt = prompt
        self.outputs = [OutputText(text=output["text"]) for output in outputs]


# 之前的实验中发现，在LLM_TYPES=("Meta-Llama-3.1-8B-Instruct-GPTQ-INT4")时，需要加上下面这句话。但是现在也不考虑这个模型了
# You don't need to repeat the Given Sentence and Event Type Set again, just response with the "Reasoning Explanations" and "Prediction" parts.

def construct_original_scm_prompt(args, sample, icl_samples):
    basic_instruction_prompt = '''Instruction: Please find event trigger words in the sentence that trigger certain types of events. The event type is from the event type set.
    Demonstrations:
    {demonstrations}
    <Hint> 
    Please learn the demonstration and follow the instruction, complete the "Reasoning Explanations" and "Prediction" parts of the new given instance. 
    You only need to solve the only instance given. 
    Format requirement:
        Reasoning Explanations:
            Write exactly one sentence per trigger word found, separated by a semicolon ";", including after the last sentence.
            Each sentence must strictly follow the pattern:
                'The trigger word "[trigger]" implies that [explanation], which implies a "[EventType]" event happened'
            where [trigger] is the exact trigger word verbatim from the sentence, [explanation] describes the semantic evidence, and [EventType] must come from the Event Type Set.
            A trigger word must appear quoted only in its own sentence and must not appear quoted in any other sentence.
        Prediction:
            Write exactly one clause per trigger word, separated by a semicolon ";", including after the last clause.
            Each clause must strictly follow the pattern:
                'The word "[trigger]" is the trigger of a "[EventType]" event'
    Please end with <End of Instance> when complete the instance.
    <Hint>
    Inference:
    <Start of Instance>
    Given Sentence: "{given_sentence}"
    Event Type Set: {event_type_set}
    '''
    basic_instruction_prompt = basic_instruction_prompt.replace('\n    ', '\n')

    basic_icl_prompt = '''<Start of Instance>
    Given Sentence: "{given_sentence}"
    Event Type Set: {event_type_set}
    Reasoning Explanations: {explanations}
    Prediction: {prediction}
    <End of Instance>
    '''

    basic_icl_prompt = basic_icl_prompt.replace('\n    ', '\n')

    event_type_set_str = '{' + ', '.join(args.event_type_set) + '}'

    demonstrations_formatted = []
    for demo_idx, demo in enumerate(icl_samples):
        demonstrations_formatted.append(f'Demo Index: {str(demo_idx)}\n')

        labeled_triggers = demo['triggers']
        labeled_triggers_start_idx = []
        formatted_predictions_list = []
        # 先按照每个trigger的开始index进行排序
        for labeled_trigger in labeled_triggers:
            span_text = labeled_trigger["text"]
            event_type = labeled_trigger["event_type"]
            start_idx = labeled_trigger["start"]
            labeled_triggers_start_idx.append(start_idx)

            formatted_prediction = f'The word "{span_text}" is the trigger of a "{event_type}" event;'
            formatted_predictions_list.append(formatted_prediction)

        # 按照labeled_triggers_start_idx的顺序对formatted_triggers_list重新排序
        formatted_predictions_list = sorted(formatted_predictions_list,
                                            key=lambda x: labeled_triggers_start_idx[
                                                formatted_predictions_list.index(x)])
        prediction_str = ' '.join(formatted_predictions_list)

        demonstrations_formatted.append(
            basic_icl_prompt.format(
                given_sentence=' '.join(demo["sentence"]),
                event_type_set=event_type_set_str,
                explanations=demo["explanations"],
                prediction=prediction_str,
            )
        )
    demonstrations_str = ''.join(demonstrations_formatted)
    demonstrations_str = demonstrations_str.replace('\n    ', '\n')

    original_scm_prompt = basic_instruction_prompt.format(
        demonstrations=demonstrations_str,
        given_sentence=' '.join(sample["sentence"]),
        event_type_set=event_type_set_str,
    )

    return original_scm_prompt


def convert_triggers_to_formatted_str(triggers):
    labeled_triggers_start_idx = []
    formatted_triggers_list = []
    formatted_predictions_list = []

    # 先按照每个trigger的开始index进行排序
    for labeled_trigger in triggers:
        span_text = labeled_trigger["text"]
        event_type = labeled_trigger["event_type"]
        start_idx = labeled_trigger["start"]
        labeled_triggers_start_idx.append(start_idx)

        formatted_trigger = f'The word "{span_text}" triggers a "{event_type}" event;'
        formatted_triggers_list.append(formatted_trigger)

        formatted_prediction = f'The word "{span_text}" is the trigger of a "{event_type}" event;'
        formatted_predictions_list.append(formatted_prediction)

    # 按照labeled_triggers_start_idx的顺序对formatted_triggers_list重新排序
    formatted_triggers_list = sorted(formatted_triggers_list,
                                     key=lambda x: labeled_triggers_start_idx[formatted_triggers_list.index(x)])
    prior_knowledge_str = ' '.join(formatted_triggers_list)

    # 按照labeled_triggers_start_idx的顺序对formatted_triggers_list重新排序
    formatted_predictions_list = sorted(formatted_predictions_list,
                                        key=lambda x: labeled_triggers_start_idx[formatted_predictions_list.index(x)])
    prediction_str = ' '.join(formatted_predictions_list)
    return prior_knowledge_str, prediction_str


def construct_hard_intervention_prompt(args, sample, icl_samples, specified_bias_hard_label=None):
    basic_instruction_prompt = '''Instruction: Given a sentence, explain why some certain words in the sentence are triggers of certain events.
    Demonstration:
    {demonstrations}
    <Hint> 
    Please learn the demonstration and follow the instruction, output the Reasoning Explanations and Prediction parts of the new given instance. 
    Format requirement: 
        In Reasoning Explanations, write exactly one sentence per trigger word listed in Prior Knowledge, separated by a semicolon ";". 
        Each sentence must strictly follow the pattern: 
            'The trigger word "[trigger]" implies that [explanation], which implies a "[EventType]" event happened'
        where [trigger] is the exact trigger word verbatim, [explanation] describes the semantic evidence in the sentence, and [EventType] matches exactly the event type listed in Prior Knowledge.
        The number of sentences in Reasoning Explanations must equal the number of trigger words listed in Prior Knowledge.
    <Hint>
    <Start of Instance>
    Given Sentence: "{given_sentence}"
    Prior Knowledge: {prior_knowledge}
    '''
    basic_instruction_prompt = basic_instruction_prompt.replace('\n    ', '\n')

    basic_icl_prompt = '''<Start of Instance>
    Given Sentence: "{given_sentence}"
    Prior Knowledge: {prior_knowledge}
    Reasoning Explanations: {explanations}
    Prediction: {prediction}
    <End of Instance>
    '''
    basic_icl_prompt = basic_icl_prompt.replace('\n    ', '\n')

    demonstrations_formatted = []
    for demo_idx, demo in enumerate(icl_samples):
        demonstrations_formatted.append(f'Demo Index: {str(demo_idx)}\n')

        demo_prior_knowledge_str, demo_prediction_str = convert_triggers_to_formatted_str(triggers=demo['triggers'])

        demonstrations_formatted.append(
            basic_icl_prompt.format(
                given_sentence=' '.join(demo["sentence"]),
                prior_knowledge=demo_prior_knowledge_str,
                explanations=demo["explanations"],
                prediction=demo_prediction_str,
            )
        )
    demonstrations_str = ''.join(demonstrations_formatted)
    demonstrations_str = demonstrations_str.replace('\n    ', '\n')

    prior_knowledge_str, _ = convert_triggers_to_formatted_str(triggers=sample["triggers"])

    hard_intervention_prompt = basic_instruction_prompt.format(
        demonstrations=demonstrations_str,
        given_sentence=' '.join(sample["sentence"]),
        prior_knowledge=prior_knowledge_str,
    )
    return hard_intervention_prompt


def construct_check_intervention_prompt(args, sample, intervention_x, icl_samples):
    basic_instruction_prompt = '''Instruction: Given a sentence and corresponding reasoning explanations, try to derive the prediction of event triggers in the sentence.
    Demonstration:
    {demonstrations}
    <Hint>Please learn the demonstration and follow the instruction, output the Prediction part of the new given instance. <Hint>
    <Start of Instance>
    Given Sentence: "{given_sentence}"
    Event Type Set: {event_type_set}
    Reasoning Explanations: {explanations}
    '''
    basic_instruction_prompt = basic_instruction_prompt.replace('\n    ', '\n')

    basic_icl_prompt = '''<Start of Instance>
    Given Sentence: "{given_sentence}"
    Event Type Set: {event_type_set}
    Reasoning Explanations: {explanations}
    Prediction: Based on the above reasoning explanations: {prediction}
    <End of Instance>
    '''

    basic_icl_prompt = basic_icl_prompt.replace('\n    ', '\n')

    event_type_set_str = '{' + ', '.join(args.event_type_set) + '}'

    demonstrations_formatted = []
    for demo_idx, demo in enumerate(icl_samples):
        demonstrations_formatted.append(f'Demo Index: {str(demo_idx)}\n')

        demo_explanations = demo["explanations"]
        _, demo_prediction_str = convert_triggers_to_formatted_str(triggers=demo['triggers'])

        demonstrations_formatted.append(
            basic_icl_prompt.format(
                given_sentence=' '.join(demo["sentence"]),
                event_type_set=event_type_set_str,
                explanations=demo_explanations,
                prediction=demo_prediction_str,
            )
        )
    demonstrations_str = ''.join(demonstrations_formatted)
    demonstrations_str = demonstrations_str.replace('\n    ', '\n')

    soft_intervention_prompt = basic_instruction_prompt.format(
        demonstrations=demonstrations_str,
        given_sentence=' '.join(sample["sentence"]),
        event_type_set=event_type_set_str,
        explanations=intervention_x,
    )

    return soft_intervention_prompt


def parse_original_scm_output(args, output):
    pred_triggers, explanations = parse_reasoning_explanations_from_text(args=args, generated_text=output)
    original_scm = {
        "x": explanations,
        "pred_label": pred_triggers,
    }
    return original_scm


def parse_hard_intervention_output(args, output):
    _, hard_intervention_x = parse_reasoning_explanations_from_text(args=args, generated_text=output)
    return hard_intervention_x


def parse_soft_intervention_output(args, output):
    pred_triggers, _ = parse_reasoning_explanations_from_text(args=args, generated_text=output)

    return pred_triggers
