from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import BertModel
import math
import os

from typing import List, Optional

import torch
import torch.nn as nn

class BertSeqTagger(nn.Module):
    def __init__(self, bert_embed_dim, hidden_size,
                 num_tag, num_bert_layer=8,
                 dropout=0.5, bert_model_path=None):
        super(BertSeqTagger, self).__init__()
        self.encoder = None
        self.bert_embed_dim = bert_embed_dim
        self.num_tag = num_tag
        self.dropout = dropout
        self.bert = BertEmbedding(bert_model_path,
                                  num_bert_layer,
                                  proj_dim=self.bert_embed_dim,
                                  use_proj=False)

        hidden_size = self.bert_embed_dim // 2

        self.hidden2tag = nn.Linear(2*hidden_size, num_tag)
        self.tag_crf = CRF(num_tags=num_tag, batch_first=True)

    def bert_named_params(self):
        return self.bert.bert.named_parameters()
        # return self.bert.named_parameters()

    def base_named_params(self):
        bert_params = list(map(id, self.bert.bert.parameters()))
        other_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and id(param) not in bert_params:
                other_params.append((name, param))
        return other_params




    def forward(self, bert_inp, mask=None):
        '''
        :param bert_inp: bert_ids, segments, bert_masks, bert_lens
        :param mask: (bs, seq_len)  0 for padding
        :return:
        '''
        bert_repr = self.bert(*bert_inp)
        if self.training:
            # bert_repr = timestep_dropout(bert_repr, p=self.dropout)
            bert_repr = F.dropout(bert_repr, p=self.dropout, training=self.training)

        enc_out = bert_repr

        if self.training:
            # enc_out = timestep_dropout(enc_out, p=self.dropout)
            enc_out = F.dropout(enc_out, p=self.dropout, training=self.training)

        tag_score = self.hidden2tag(enc_out)
        return tag_score

    def tag_loss(self, tag_score, gold_tags, mask=None, reduction='mean', alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)
        :param gold_tags: (b, t)
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg: 'greedy' and 'crf'
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            lld = self.tag_crf(tag_score, tags=gold_tags, mask=mask, reduction=reduction)
            return lld.neg()
        else:
            sum_loss = F.cross_entropy(tag_score.transpose(1, 2), gold_tags, ignore_index=0, reduction='sum')
            return sum_loss / mask.sum()

    def tag_decode(self, tag_score, mask=None, alg='crf'):
        '''
        :param tag_score: (b, t, nb_cls)  emission probs
        :param mask: (b, t)  1对应有效部分，0对应pad
        :param alg:
        :return:
        '''
        assert alg in ['greedy', 'crf']
        if alg == 'crf':
            best_tag_seq = self.tag_crf.decode(tag_score, mask=mask)
            # return best segment tags
            return pad_sequence(best_tag_seq, batch_first=True, padding_value=0)
        else:
            return tag_score.data.argmax(dim=-1) * mask.long()


class EDContrastiveRetriever(nn.Module):
    def __init__(self, args, PLM, PLM_hidden_size):
        super(EDContrastiveRetriever, self).__init__()
        self.args = args
        self.encoder = BertModel.from_pretrained(PLM)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(PLM_hidden_size, PLM_hidden_size),
        )

    def forward(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None, labels_id=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        # print(bert_output_raw.size())
        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]

        cls_mlp_output = self.mlp(cls_bert_output)
        # print(logits.size())

        loss = self.calculate_contrastive_loss(features=cls_mlp_output,
                                               label_ids=labels_id,
                                               temperature=0.2)  # 在llama-7b设置0.1可以获得基本效果，gpt-3.5可能不太行

        return loss

    def calculate_contrastive_loss(self, features, label_ids, temperature):
        cat_labels = []
        for label_id in label_ids:
            cat_labels.append(self.args.cat_labels[label_id])
        # print(cat_labels)

        """
        calculate traditional supervised contrastive loss for comparison
        Reference: https://github.com/HobbitLong/SupContrast
        """
        diagonal = torch.eye(label_ids.shape[0], dtype=torch.bool).float().to(self.args.device)

        # 使用eq函数进行比较，得到一个二维的布尔tensor
        mask_label_equal = label_ids.unsqueeze(1).eq(label_ids.unsqueeze(0))
        # 将布尔tensor转为整型int类型的tensor
        mask_label_equal = mask_label_equal.int()

        positive_mask = mask_label_equal - diagonal  # 1 only when label is same(not include itself)

        negative_mask = 1. - mask_label_equal
        ################################################################################
        # 负样本对，并不是除了标签一样的都可以视为负样本；
        # not_positive_or_negative = []
        # for i_cat_label in cat_labels:
        #     temp = []
        #     i_pred_label = i_cat_label.split('#')[0]
        #     i_true_label = i_cat_label.split('#')[0]
        #     for j_cat_label in cat_labels:
        #         j_pred_label = j_cat_label.split('#')[0]
        #         j_true_label = j_cat_label.split('#')[0]
        #         if i_pred_label != j_pred_label and i_true_label != j_true_label:
        #             temp.append(1)
        #         else:
        #             temp.append(0)
        #     temp = torch.tensor(temp)
        #     not_positive_or_negative.append(temp)
        # not_positive_or_negative = torch.stack(not_positive_or_negative).to(self.args.device)
        # print(not_positive_or_negative)
        # negative_mask = negative_mask - not_positive_or_negative
        ################################################################################
        # print(negative_mask)

        # features = torch.nn.functional.normalize(features, dim=-1, p=2)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)  # 计算两两样本间点乘相似度
        # for numerical stability,减去最大正样本对的值是为了防止模型以为本行已经训练好了；
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)

        denominator = (torch.sum(exp_logits * negative_mask, dim=1, keepdim=True)
                       + torch.sum(exp_logits * positive_mask, dim=1, keepdim=True))
        log_probs = logits - torch.log(denominator)   #(log(exp(logits)/denominator))
        # print(logits)
        # print(denominator)


        # 每一行的正样本对数
        num_positives_per_row = torch.sum(positive_mask, dim=1)
        # print(num_positives_per_row)
        # print(torch.sum(negative_mask, dim=1))

        # 除以每一行的正样本对数，对于没有正样本的忽略
        log_probs = (torch.sum(log_probs * positive_mask, dim=1)[num_positives_per_row > 0]
                     / num_positives_per_row[num_positives_per_row > 0])
        loss = -log_probs
        loss = loss.mean()

        # if torch.isnan(loss).any():
        #     print('这里发现nan')
        #     print(logits)
        #     # print(denominator)
        #     print(positive_mask)
        #     print(num_positives_per_row)
        #     print(log_probs)


        return loss

    def get_emb(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None):
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]

        # 取出special_mask为1的位置
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]

        return cls_bert_output







class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class BertEmbedding(nn.Module):
    def __init__(self, model_path, nb_layers=1, merge='none', fine_tune=True, use_proj=False, proj_dim=256):
        super(BertEmbedding, self).__init__()
        assert merge in ['none']
        self.merge = merge
        self.use_proj = use_proj
        self.proj_dim = proj_dim
        self.fine_tune = fine_tune
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)

        self.bert_layers = self.bert.config.num_hidden_layers + 1  # including embedding layer
        self.nb_layers = nb_layers if nb_layers < self.bert_layers else self.bert_layers
        self.hidden_size = self.bert.config.hidden_size


        if not self.fine_tune:
            for p in self.bert.parameters():
                p.requires_grad = False

        if self.use_proj:
            self.proj = nn.Linear(self.hidden_size, self.proj_dim, bias=False)
            self.hidden_size = self.proj_dim
        else:
            self.proj = None

    def save_bert(self, save_dir):
        # saved into config file and model
        assert os.path.isdir(save_dir)
        self.bert.save_pretrained(save_dir)
        print('BERT Saved !!!')

    def get_emb(self, input_ids=None, special_mask=None, token_type_ids=None, attention_mask=None):
        # 取出特定于任务的 emb，size为  batch_size * emb_size
        self.encoder=self.bert
        bert_outputs_raw = self.encoder(input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask
                                        )
        bert_output_raw = bert_outputs_raw.last_hidden_state

        # print(bert_output_raw.size())  # torch.Size([128, 128, 768])
        # print(attention_mask.size())   # torch.Size([128, 128])

        # 第一步，将 bert_output_raw 中 对应于attention_mask[i,j]=0的置为全零，形成 masked_bert_output_raw
        # 第二步，masked_bert_output_raw 按照行求和，形成 sum_masked_bert_output_raw torch.Size([128, 768])
        # 第三步，求出 attention_mask 每行值为1的数目；形成 torch.Size([128, 1])
        # 第四步，sum_masked_bert_output_raw 每行除以 attention_mask每行的值，形成 mean_masked_bert_output_raw torch.Size([128, 768])

        # # Step 1: Set elements in bert_output_raw to 0 where attention_mask is 0
        # attention_mask_extended = attention_mask.unsqueeze(-1).expand_as(bert_output_raw)
        # masked_bert_output_raw = bert_output_raw * attention_mask_extended.float()
        #
        # # Step 2: Calculate the sum across the sequence length dimension
        # sum_masked_bert_output_raw = masked_bert_output_raw.sum(dim=1)  # torch.Size([128, 768])
        #
        # # Step 3: Calculate the number of 1's in each row of attention_mask
        # num_ones = attention_mask.sum(dim=1, keepdim=True)  # torch.Size([128, 1])
        #
        # # Step 4: Calculate the average by dividing sum_masked_bert_output_raw by num_ones
        # mean_masked_bert_output_raw = sum_masked_bert_output_raw / num_ones  # torch.Size([128, 768])
        # batch_mean_emb = mean_masked_bert_output_raw

        # print(mean_masked_bert_output_raw.size())  # The expected size is [batch_size, hidden_size], e.g., torch.Size([128, 768]).

        # 下面是只取 cls位的embedding作为检索向量
        bert_output_raw_flatten = torch.flatten(bert_output_raw, start_dim=0, end_dim=1)[:]
        special_mask_flatten = torch.flatten(special_mask, start_dim=0, end_dim=1)[:]
        mask1 = special_mask_flatten == 1
        cls_bert_output = bert_output_raw_flatten[mask1]
        batch_mean_emb = cls_bert_output


        return batch_mean_emb

    def forward(self, bert_ids, segments, bert_mask, bert_lens):
        '''
        :param bert_ids: (bz, bpe_seq_len) subword indexs
        :param segments: (bz, bpe_seq_len)  只有一个句子，全0
        :param bert_mask: (bz, bep_seq_len)  经过bpe切词
        :param bert_lens: (bz, seq_len)  每个token经过bpe切词后的长度
        :return:
        '''
        bz, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        bert_mask = bert_mask.type_as(mask)

        if self.fine_tune:
            last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)
        else:
            with torch.no_grad():
                last_enc_out, _, all_enc_outs = self.bert(bert_ids, token_type_ids=segments, attention_mask=bert_mask, return_dict=False)

        if self.merge == 'linear':
            enc_out = self.scale(all_enc_outs[-self.nb_layers:])  # (bz, seq_len, 768)

            # encoded_repr = 0
            # soft_weight = F.softmax(self.weighing_params, dim=0)
            # for i in range(self.nb_layers):
            #     encoded_repr += soft_weight[i] * all_enc_outs[i]
            # enc_out = encoded_repr
        elif self.merge == 'mean':
            top_enc_outs = all_enc_outs[-self.nb_layers:]
            enc_out = sum(top_enc_outs) / len(top_enc_outs)
            # enc_out = torch.stack(tuple(top_enc_outs), dim=0).mean(0)
        else:
            enc_out = last_enc_out

        # 根据bert piece长度切分
        bert_chunks = enc_out[bert_mask].split(bert_lens[mask].tolist())
        bert_out = torch.stack(tuple([bc.mean(0) for bc in bert_chunks]))
        bert_embed = bert_out.new_zeros(bz, seq_len, self.bert.config.hidden_size)
        # 将bert_embed中mask对应1的位置替换成bert_out，0的位置不变
        output = bert_embed.masked_scatter_(mask.unsqueeze(dim=-1), bert_out)

        if self.proj:
            return self.proj(output)
        else:
            return output


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            penalty_ws = None,   # penalty for instance
            mixup_ws = None,   # penalty for instance
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        '''
        if penalty_ws is not None:
            numerator = torch.mul(numerator, penalty_ws.squeeze(1))  # 点乘
        if mixup_ws is not None:
            numerator = torch.mul(numerator, mixup_ws.squeeze(1))  # 点乘
        '''

        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        '''
        if penalty_ws is not None:
            denominator = torch.mul(denominator, penalty_ws.squeeze(1))  # 点乘
        if mixup_ws is not None:
            denominator = torch.mul(denominator, mixup_ws.squeeze(1))  # 点乘
        '''

        # shape: (batch_size,)
        llh = numerator - denominator

        if penalty_ws is not None:
            llh = torch.mul(penalty_ws.squeeze(1), llh)  # 点乘
        if mixup_ws is not None:
            llh = torch.mul(mixup_ws.squeeze(1), llh)  # 点乘

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()


        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            # best_tags_list.append(best_tags)
            best_tags_list.append(torch.tensor(best_tags, device=emissions.device))

        return best_tags_list
