import json
import os

from prompt_tools import ChatGPT_Class


def prepare_label_set(dataset):
    # 注意，这里的label列表第一个（索引为0）设置为Other或者无关系类型，因为在解析不出来时会默认为此Other/no_relation
    # if dataset in ['TACRED', 'TACRED_ZeroShot']:
    #     label_set = ['no_relation', 'org:founded_by', 'per:employee_of',
    #                  'org:alternate_names', 'per:cities_of_residence',
    #                  'per:children', 'per:title', 'per:siblings',
    #                  'per:religion', 'per:age', 'org:website',
    #                  'per:stateorprovinces_of_residence',
    #                  'org:member_of', 'org:top_members/employees',
    #                  'per:countries_of_residence', 'org:city_of_headquarters',
    #                  'org:members', 'org:country_of_headquarters',
    #                  'per:spouse', 'org:stateorprovince_of_headquarters',
    #                  'org:number_of_employees/members', 'org:parents',
    #                  'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation',
    #                  'per:other_family', 'per:stateorprovince_of_birth',
    #                  'org:dissolved', 'per:date_of_death', 'org:shareholders',
    #                  'per:alternate_names', 'per:parents', 'per:schools_attended',
    #                  'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death',
    #                  'org:founded', 'per:country_of_birth', 'per:date_of_birth',
    #                  'per:city_of_birth', 'per:charges', 'per:country_of_death']

    if dataset in ["ACE05","ACE05_CN",]:
        label_set = [
            'Justice:Acquit',
            'Justice:Release-Parole',
            'Justice:Pardon',
            'Justice:Convict',
            'Conflict:Demonstrate',
            'Conflict:Attack',
            'Personnel:Start-Position',
            'Justice:Appeal',
            'Contact:Phone-Write',
            'Life:Marry',
            'Transaction:Transfer-Money',
            'Life:Be-Born',
            'Justice:Sue',
            'Business:Start-Org',
            'Personnel:Elect',
            'Justice:Fine',
            'Justice:Extradite',
            'Justice:Execute',
            'Transaction:Transfer-Ownership',
            'Movement:Transport',
            'Justice:Arrest-Jail',
            'Life:Die',
            'Personnel:Nominate',
            'Business:Declare-Bankruptcy',
            'Personnel:End-Position',
            'Life:Divorce',
            'Life:Injure',
            'Justice:Trial-Hearing',
            'Justice:Charge-Indict',
            'Business:End-Org',
            'Business:Merge-Org',
            'Justice:Sentence',
            'Contact:Meet']
    elif dataset in ["DuEE"]:
        label_set = [
            "组织关系-辞/离职", "交往-点赞", "财经/交易-上市", "组织行为-罢工",
    "财经/交易-降息", "财经/交易-出售/收购", "人生-求婚", "财经/交易-涨停",
    "人生-庆生", "财经/交易-涨价", "司法行为-举报", "竞赛行为-退役",
    "组织关系-解雇", "人生-分手", "产品行为-下架", "人生-婚礼",
    "交往-探班", "产品行为-召回", "竞赛行为-晋级", "灾害/意外-洪灾",
    "财经/交易-跌停", "人生-订婚", "灾害/意外-地震", "人生-出轨",
    "人生-结婚", "人生-死亡", "产品行为-获奖", "竞赛行为-夺冠",
    "组织关系-解散", "人生-离婚", "灾害/意外-坠机", "灾害/意外-坍/垮塌",
    "竞赛行为-退赛", "竞赛行为-胜负", "财经/交易-融资", "灾害/意外-车祸",
    "产品行为-发布", "灾害/意外-袭击", "司法行为-开庭", "组织关系-停职",
    "财经/交易-降价", "灾害/意外-起火", "组织关系-加盟", "司法行为-约谈",
    "人生-产子/女", "人生-怀孕", "司法行为-入狱", "财经/交易-加息",
    "组织行为-游行", "竞赛行为-禁赛", "产品行为-上映", "人生-失联",
    "司法行为-立案", "交往-会见", "司法行为-拘捕", "组织行为-开幕",
    "组织行为-闭幕", "交往-感谢", "司法行为-起诉", "组织关系-裁员",
    "组织关系-退出", "交往-道歉", "司法行为-罚款", "灾害/意外-爆炸",
    "组织关系-解约"
        ]

    elif dataset in ["WikiEvents"]:
        label_set = [
            "Personnel.EndPosition.Unspecified",
            "Life.Die.Unspecified",
            "Justice.Convict.Unspecified",
            "Transaction.ExchangeBuySell.Unspecified",
            "Contact.ThreatenCoerce.Correspondence",
            "Contact.Contact.Unspecified",
            "Transaction.Donation.Unspecified",
            "ArtifactExistence.DamageDestroyDisableDismantle.Dismantle",
            "GenericCrime.GenericCrime.GenericCrime",
            "Justice.ArrestJailDetain.Unspecified",
            "Contact.ThreatenCoerce.Unspecified",
            "Life.Infect.Unspecified",
            "Justice.ReleaseParole.Unspecified",
            "Conflict.Demonstrate.DemonstrateWithViolence",
            "Movement.Transportation.Unspecified",
            "ArtifactExistence.DamageDestroyDisableDismantle.Unspecified",
            "Movement.Transportation.IllegalTransportation",
            "Cognitive.IdentifyCategorize.Unspecified",
            "Medical.Intervention.Unspecified",
            "Cognitive.Research.Unspecified",
            "Disaster.DiseaseOutbreak.Unspecified",
            "Contact.Contact.Broadcast",
            "Contact.RequestCommand.Broadcast",
            "Justice.Acquit.Unspecified",
            "Contact.Contact.Meet",
            "Personnel.StartPosition.Unspecified",
            "Movement.Transportation.PreventPassage",
            "Justice.ChargeIndict.Unspecified",
            "ArtifactExistence.ManufactureAssemble.Unspecified",
            "Contact.RequestCommand.Unspecified",
            "Conflict.Demonstrate.Unspecified",
            "ArtifactExistence.DamageDestroyDisableDismantle.Destroy",
            "Disaster.Crash.Unspecified",
            "Cognitive.Inspection.SensoryObserve",
            "Conflict.Attack.Unspecified",
            "Movement.Transportation.Evacuation",
            "Contact.Contact.Correspondence",
            "Contact.RequestCommand.Meet",
            "Justice.TrialHearing.Unspecified",
            "Control.ImpedeInterfereWith.Unspecified",
            "Justice.InvestigateCrime.Unspecified",
            "Contact.ThreatenCoerce.Broadcast",
            "Cognitive.TeachingTrainingLearning.Unspecified",
            "Conflict.Defeat.Unspecified",
            "ArtifactExistence.DamageDestroyDisableDismantle.Damage",
            "ArtifactExistence.DamageDestroyDisableDismantle.DisableDefuse",
            "Justice.Sentence.Unspecified",
            "Life.Injure.Unspecified",
            "Conflict.Attack.DetonateExplode",
        ]

    elif dataset in ["MAVEN"]:
        label_set = [
            'Rewards_and_punishments',
            'Rite',
            'Emptying',
            'Exchange',
            'Escaping',
            'Besieging',
            'Preserving',
            'Employment',
            'Testing',
            'Cause_to_amalgamate',
            'Change_event_time',
            'Resolve_problem',
            'Hostile_encounter',
            'Warning',
            'Quarreling',
            'Conquering',
            'Cause_change_of_strength',
            'Attack',
            'Cause_change_of_position_on_a_scale',
            'Imposing_obligation',
            'Check',
            'Change_tool',
            'Aiming',
            'Control',
            'Renting',
            'Commerce_buy',
            'Containing',
            'Know',
            'Lighting',
            'Institutionalization',
            'Limiting',
            'Body_movement',
            'GiveUp',
            'Patrolling',
            'Theft',
            'Self_motion',
            'Confronting_problem',
            'Submitting_documents',
            'Destroying',
            'Justifying',
            'Death',
            'Committing_crime',
            'Research',
            'Reveal_secret',
            'Hiding_objects',
            'Request',
            'Extradition',
            'Change_of_leadership',
            'Cause_to_make_progress',
            'Cure',
            'Deciding',
            'Statement',
            'Using',
            'Risk',
            'Legality',
            'Presence',
            'Bodily_harm',
            'Sending',
            'Forming_relationships',
            'Becoming_a_member',
            'Bearing_arms',
            'Use_firearm',
            'Becoming',
            'Expansion',
            'Revenge',
            'Process_start',
            'Perception_active',
            'Giving',
            'Create_artwork',
            'Commerce_sell',
            'Reforming_a_system',
            'Change',
            'Coming_to_believe',
            'Process_end',
            'Being_in_operation',
            'Damaging',
            'Motion_directional',
            'Motion',
            'Commerce_pay',
            'Terrorism',
            'Choosing',
            'Recovering',
            'Writing',
            'Protest',
            'Receiving',
            'Cause_to_be_included',
            'Vocalizations',
            'Agree_or_refuse_to_act',
            'Dispersal',
            'Labeling',
            'Military_operation',
            'Supporting',
            'Manufacturing',
            'Building',
            'Hindering',
            'Commitment',
            'Response',
            'Creating',
            'Competition',
            'Prison',
            'GetReady',
            'Expend_resource',
            'Award',
            'Earnings_and_losses',
            'Surrendering',
            'Criminal_investigation',
            'Preventing_or_letting',
            'Expressing_publicly',
            'Connect',
            'Arriving',
            'Causation',
            'Traveling',
            'Name_conferral',
            'Reporting',
            'Arranging',
            'Removing',
            'Come_together',
            'Ingestion',
            'Suspicion',
            'Defending',
            'Hold',
            'Incident',
            'Education_teaching',
            'Departing',
            'Rescuing',
            'Filling',
            'Collaboration',
            'Supply',
            'Adducing',
            'Violence',
            'Convincing',
            'Scrutiny',
            'Temporary_stay',
            'Having_or_lacking_access',
            'Achieve',
            'Practice',
            'Participation',
            'Sign_agreement',
            'Action',
            'Breathing',
            'Recording',
            'Judgment_communication',
            'Killing',
            'Telling',
            'Legal_rulings',
            'Releasing',
            'Arrest',
            'Wearing',
            'Ratification',
            'Carry_goods',
            'Catastrophe',
            'Getting',
            'Robbery',
            'Kidnapping',
            'Social_event',
            'Placing',
            'Emergency',
            'Influence',
            'Bringing',
            'Assistance',
            'Scouring',
            'Publishing',
            'Coming_to_be',
            'Communication',
            'Cost',
            'Surrounding',
            'Openness',
            'Change_sentiment'
        ]

    label_set=['Other'] + label_set
    print("===================== label set in this running ======================",label_set)

    label_dict = {}
    for idx, item in enumerate(label_set):
        label_dict[item] = idx

    return label_set, label_dict


def prepare_llm(args):
    temperature = float(os.environ.get('LLM_TEMPERATURE', 0.0))  # 默认0.0
    print(f"======We set LLM_TEMPERATURE as {temperature}")

    if args.llm_type == 'gpt-3.5-turbo':
        sampling_params = {
            "temperature": temperature,
        }
        llm = ChatGPT_Class(llm_type=args.llm_type)

    else:
        from vllm import LLM, SamplingParams

        try:
            args.run_id = args.run_id
        except:
            args.run_id = 0

        vllm_seed=args.seed + args.run_id * 1000
        print(f"======We set vllm_seed as {vllm_seed}")

        # 定义采样参数，temperature 控制生成文本的多样性，top_p 控制核心采样的概率
        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=512)

        # root_llm_path="../../../LLMs" # on 4090
        root_llm_path="/mnt/data/LLMS" # on L20

        if "Qwen" in args.llm_type:
            model_path=f"{root_llm_path}/Qwen/{args.llm_type}"
        elif "gemma-3-4b-it-GPTQ-4b-128g" in args.llm_type:
            model_path=f"{root_llm_path}/ISTA-DASLab/{args.llm_type}"
        elif "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" in args.llm_type:
            if args.dataset in ["ACE05", "ACE05_CN"]:
                model_path=f"{root_llm_path}/hugging-quants/{args.llm_type}"
            else:
                model_path=f"{root_llm_path}/shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ"

        gpu_memory_utilization=0.8


        if args.llm_type in ["Qwen3-4B-Instruct-2507-FP8", "gemma-3-4b-it-GPTQ-4b-128g"]:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=102400,
                seed=vllm_seed,

            )
        elif args.llm_type in ["Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=90000,
                seed=vllm_seed,

            )
        else:
            llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                seed=vllm_seed,
            )
    return llm, sampling_params


from jinja2 import Template

# 定义 chat template（注意：需符合 Jinja2 语法）
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] != 'assistant' %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% if message['content'] is not none %}"
    "{{ message['content'] + '<|im_end|>' }}"
    "{% endif %}"
    "{{ '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)
CHAT_TEMPLATE_LLAMA = (
    "{{ bos_token }}"
    "{% set system_message = '' %}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% set messages = messages[1:] %}"
    "{% endif %}"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n"
    "{{ system_message }}<|eot_id|>"
    "{% for message in messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'].strip() + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)

def templated_prompts(args,
                      prompts,
                      system_prompt="You are a helpful assistant. Please strictly follow the format given in the demonstration part in the prompt."
                      ):
    if args.llm_type in ["Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]:
        template = Template(CHAT_TEMPLATE_LLAMA)
        results = []

        for user_prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            rendered = template.render(
                messages=messages,
                add_generation_prompt=True,
                bos_token="<|begin_of_text|>"
            )
            results.append(rendered)
    else:
        template = Template(CHAT_TEMPLATE)
        results = []

        for user_prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            # 生成 prompt（不包含 generation 占位符，因为只是用于输入）
            rendered = template.render(
                messages=messages,
                add_generation_prompt=True  # 添加 assistant 开头，让模型知道要开始生成
            )
            results.append(rendered)

    return results


def log_result(filepath, **kwargs):
    """每次实验结束后写一行 JSON"""
    record = kwargs
    line = json.dumps(record, ensure_ascii=False)
    print(line)
    if filepath:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

def summarize_token_usage(outputs):
    """
    统计 vLLM 离线推理 outputs 的 token 使用情况

    Args:
        outputs: llm.generate() 返回的 List[RequestOutput]

    Returns:
        dict: 包含总数、样本数及平均值的统计信息
    """
    prompt_tokens = 0
    completion_tokens = 0
    num_prompts = len(outputs)
    num_completions = 0  # 生成的候选总数（n>1 时会大于 num_prompts）

    for output in outputs:
        prompt_tokens += len(output.prompt_token_ids)
        completion_tokens += sum(len(comp.token_ids) for comp in output.outputs)
        num_completions += len(output.outputs)

    return {
        "num_prompts": num_prompts,
        "num_completions": num_completions,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "avg_prompt_tokens": prompt_tokens / num_prompts if num_prompts else 0,
        "avg_completion_tokens": completion_tokens / num_completions if num_completions else 0,
        "avg_total_tokens_per_prompt": (prompt_tokens + completion_tokens) / num_prompts if num_prompts else 0,
    }