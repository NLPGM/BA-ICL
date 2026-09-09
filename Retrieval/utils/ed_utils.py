import random

import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader, TensorDataset


def GetDataLoader(args, samples, batch_size):
    features = []
    for sample in samples:
        if 'rational' in sample.keys():
            cat_label = sample["cat_label"]
            cat_label_id = args.cat_labels_dict[cat_label]
            rational = sample["rational"]

            features.append(convert_to_feature_rational(args, cat_label_id,rational))
        else:
            feature=convert_to_feature(args, sample)
            if feature is not None:
                features.append(feature)
    dataset = convert_features_to_dataset(features)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=batch_size)
    return train_dataloader

def GetDataEnumerater(args, samples, batch_size):
    features = []
    for sample in samples:
        if 'rational' in sample.keys():
            cat_label_id = 0 # GetDataEnumerater只用于测试阶段，所以这里只需要随意设一个id
            rational = sample["rational"]
            features.append(convert_to_feature_rational(args, cat_label_id,rational))
        else:
            feature=convert_to_feature(args, sample)
            if feature is not None:
                features.append(feature)

    dataset = convert_features_to_dataset(features)
    train_dataloader = DataLoader(dataset,
                                  batch_size=batch_size)
    return train_dataloader

def convert_to_feature_rational(args, cat_label_id,rational):
    max_seq_length = args.max_seq_length

    sample_tokens=args.tokenizer.tokenize(rational)
    sample_tokens = ["[CLS]"] + sample_tokens + ["[SEP]"]

    input_ids = args.tokenizer.convert_tokens_to_ids(sample_tokens)
    padding_length = max_seq_length - len(input_ids)

    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length

    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length]
        input_ids_padded = input_ids[:max_seq_length]

    special_mask_padded = [0] * max_seq_length
    # 将cls位置置为1，以便后续获取该句子的向量
    special_mask_padded[0]=1

    token_type_ids = [0] * max_seq_length

    assert len(input_ids_padded) == max_seq_length
    assert len(special_mask_padded) == max_seq_length

    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length


    return InputFeature(input_ids_padded, special_mask_padded, token_type_ids, attention_mask,
                        label_id=cat_label_id)

def convert_to_feature(args, sample):
    max_seq_length = args.max_seq_length


    words = sample["sentence"]

    words_after_special_mask = ["[CLS]"] + words + ["[SEP]"]
    words_special_mask = [1] + len(words)*[0] + [0]

    # assert 1==0

    sample_tokens = []
    special_mask = []

    for word, special_mask_id in zip(words_after_special_mask, words_special_mask):
        word_tokens = args.tokenizer.tokenize(word)
        if len(word_tokens) == 0:  # Meet special space character
            word_tokens = args.tokenizer.tokenize('[UNK]')
            word_special_mask_tokens = [0]
        else:
            word_special_mask_tokens = [special_mask_id]*len(word_tokens)

        sample_tokens.extend(word_tokens)
        special_mask.extend(word_special_mask_tokens)


    input_ids = args.tokenizer.convert_tokens_to_ids(sample_tokens)
    padding_length = max_seq_length - len(input_ids)

    if padding_length >= 0:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids_padded = input_ids + [0] * padding_length
        special_mask_padded = special_mask + [0] * padding_length

    else:
        attention_mask = ([1] * len(input_ids))[:max_seq_length]
        input_ids_padded = input_ids[:max_seq_length]
        special_mask_padded = special_mask[:max_seq_length]

    token_type_ids = [0] * max_seq_length

    assert len(input_ids_padded) == max_seq_length
    assert len(special_mask_padded) == max_seq_length

    assert len(token_type_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length


    return InputFeature(input_ids_padded, special_mask_padded, token_type_ids, attention_mask,
                        label_id=0)


class InputFeature(object):
    def __init__(self, input_ids, special_mask, token_type_ids, attention_mask, label_id):
        self.input_ids = input_ids
        self.special_mask = special_mask
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


def convert_features_to_dataset(features):
    # convert to Tensors
    all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
    all_special_mask = torch.tensor([feature.special_mask for feature in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([feature.token_type_ids for feature in features], dtype=torch.long)
    all_attention_mask = torch.tensor([feature.attention_mask for feature in features], dtype=torch.long)
    all_labels_id = torch.tensor([feature.label_id for feature in features])
    dataset = TensorDataset(all_input_ids, all_special_mask, all_token_type_ids, all_attention_mask, all_labels_id)
    return dataset




def prepare_label_set(dataset):
    # 注意，这里的label列表第一个（索引为0）设置为Other或者无关系类型，因为在解析不出来时会默认为此Other/no_relation
    if dataset in ['TACRED', 'TACRED_ZeroShot']:
        label_set = ['no_relation', 'org:founded_by', 'per:employee_of',
                     'org:alternate_names', 'per:cities_of_residence',
                     'per:children', 'per:title', 'per:siblings',
                     'per:religion', 'per:age', 'org:website',
                     'per:stateorprovinces_of_residence',
                     'org:member_of', 'org:top_members/employees',
                     'per:countries_of_residence', 'org:city_of_headquarters',
                     'org:members', 'org:country_of_headquarters',
                     'per:spouse', 'org:stateorprovince_of_headquarters',
                     'org:number_of_employees/members', 'org:parents',
                     'org:subsidiaries', 'per:origin', 'org:political/religious_affiliation',
                     'per:other_family', 'per:stateorprovince_of_birth',
                     'org:dissolved', 'per:date_of_death', 'org:shareholders',
                     'per:alternate_names', 'per:parents', 'per:schools_attended',
                     'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death',
                     'org:founded', 'per:country_of_birth', 'per:date_of_birth',
                     'per:city_of_birth', 'per:charges', 'per:country_of_death']
    elif dataset in ['SemEval', 'SemEval_ZeroShot']:
        label_set = ['Other',
                     'Component-Whole',
                     'Instrument-Agency',
                     'Member-Collection',
                     'Cause-Effect',
                     'Entity-Destination',
                     'Content-Container',
                     'Message-Topic',
                     'Product-Producer',
                     'Entity-Origin']
    elif dataset in ["ACE05"]:
        label_set = [
            'Other',
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
    elif dataset in ["MAVEN"]:
        label_set = [
            'Other',
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

    label_dict = {}
    for idx, item in enumerate(label_set):
        label_dict[item] = idx

    return label_set, label_dict

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)