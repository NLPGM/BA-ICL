import os
import time
import torch
import random
import numpy as np
from model import BertSeqTagger
from config.configuration import args_config, data_config
from utils.dataset import DataLoader
from utils.datautil import load_data, create_vocab, batch_variable, save_to, extract_entity_span_label_BIO
import torch.nn.utils as nn_utils
from logger.logger import logger


class Trainer(object):
    def __init__(self, args, data_config ,train_set,val_set,test_set):
        self.args = args
        self.data_config = data_config
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.do_validate=True

        self.predict_results_file = None

        print('train data size:', len(self.train_set))
        print('validate data size:', len(self.val_set))
        print('test data size:', len(self.test_set))

        self.dev_loader = DataLoader(self.val_set, batch_size=self.args.test_batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.test_batch_size)
        self.vocabs = create_vocab(self.train_set, data_config['pretrained']['bert_model'], embed_file=None)
        save_to(args.vocab_chkp, self.vocabs)


        self.model = BertSeqTagger(
            bert_embed_dim=args.bert_embed_dim,
            hidden_size=args.hidden_size,
            num_tag=len(self.vocabs['ner']),
            num_bert_layer=args.bert_layer,
            dropout=args.dropout,
            bert_model_path=data_config['pretrained']['bert_model']
        ).to(args.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Training %dM trainable parameters..." % (total_params / 1e6))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_bert_parameters = [
            {'params': [p for n, p in self.model.bert_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.bert_lr},
            {'params': [p for n, p in self.model.bert_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.bert_lr},

            {'params': [p for n, p in self.model.base_named_params() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate},
            {'params': [p for n, p in self.model.base_named_params() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.learning_rate}
            # {'params': [p for n, p in self.model.base_named_params()],
            #  'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate}
        ]

        sgd_parameters = [
            {'params': [p for n, p in self.model.bert_named_params()],
             'weight_decay': self.args.weight_decay, 'lr': self.args.bert_lr},

            {'params': [p for n, p in self.model.base_named_params()],
             'weight_decay': self.args.weight_decay, 'lr': self.args.learning_rate}
        ]

        self.optimizer = torch.optim.AdamW(optimizer_bert_parameters, lr=self.args.bert_lr, eps=self.args.eps)
        self.meta_opt = torch.optim.SGD(sgd_parameters, lr=self.args.learning_rate)
        # self.meta_opt = torch.optim.SGD(sgd_parameters, lr=self.args.bert_lr, momentum=0.9)
        # self.meta_opt = torch.optim.SGD(optimizer_bert_parameters, lr=self.args.bert_lr, momentum=0.9)

    def train_epoch(self, ep=0):
        # print('vanilla training ...')
        self.model.train()
        t1 = time.time()
        train_loss = 0.

        train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=True)

        self.model.zero_grad()
        for i, batch_train_data in enumerate(train_loader):
            batch = batch_variable(batch_train_data, self.vocabs)
            batch.to_device(self.args.device)
            tag_score = self.model(batch.bert_inp, batch.mask)
            loss = self.model.tag_loss(tag_score, batch.ner_ids, mask=batch.mask)
            loss.backward()
            loss_val = loss.data.item()
            train_loss += loss_val
            # nn_utils.clip_grad_norm_(self.model.base_params(), max_norm=self.args.grad_clip)
            # nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.bert_params()), max_norm=self.args.bert_grad_clip)
            nn_utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()),
                                     max_norm=self.args.grad_clip)
            self.optimizer.step()
            self.model.zero_grad()
            # logger.info('[Epoch %d] Iter%d time cost: %.2fs, train loss: %.3f' % (ep, i, (time.time() - t1), loss_val))
        return train_loss

    def save_states(self, save_path, best_dev_metric=None):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        # random generator state (Byte Tensor)
        rand_states = [random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state() if torch.cuda.is_available() else None]
        check_point = {'best_prf': best_dev_metric,
                       'rand_states': rand_states,
                       'model_state': self.model.state_dict(),
                       'bert_model_state': self.model.bert.state_dict(),
                       'optimizer_state': self.optimizer.state_dict(),
                       'args_settings': self.args}

        torch.save(check_point, save_path)


        # bert_ed_retriever_chkp = os.path.join(self.args.ckpt_dir,
        #                                f'bert_ed_retriever.ckpt')
        # bert_ed_retriever_check_point={
        #     'model_state_dict':self.model.bert.state_dict() # 这里打包存储的是 BertEmbedding类
        # }
        # torch.save(bert_ed_retriever_check_point, bert_ed_retriever_chkp)
        # logger.info(f'Saved the current model states to {save_path} ...')
        #


    def restore_states(self, load_path):
        ckpt = torch.load(load_path,weights_only=False)
        random.setstate(ckpt['rand_states'][0])
        np.random.set_state(ckpt['rand_states'][1])
        torch.set_rng_state(ckpt['rand_states'][2])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt['rand_states'][3])

        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.args = ckpt['args_settings']
        logger.info('Loading the previous model states ...')


    def run(self,do_validate=True):
        patient = 0
        best_dev_metric = dict()
        min_train_loss=100000
        for ep in range(self.args.epoch):
            train_loss = self.train_epoch(ep)

            # dev_metric = self.evaluate(self.dev_loader,dev=True)
            if do_validate:
                dev_metric = self.evaluate(self.dev_loader, dev=True)
                print(dev_metric)
                if dev_metric['f'] >= best_dev_metric.get('f', 0):
                    best_dev_metric = dev_metric
                    self.save_states(self.args.model_chkp, best_dev_metric)
                    logger.info('Find a BEST! GO ON!\n')
                    patient = 0
                else:
                    patient +=1
                if patient>5:
                    logger.info('No patient! BREAK!\n')
                    break


                logger.info('[Epoch %d] train loss: %.4f,  dev_metric: %s \n' %(ep, train_loss, best_dev_metric))
            # else:
            #     if train_loss<min_train_loss:
            #         min_train_loss=train_loss
            #         # logger.info('Find a min train loss!\n')
            #         self.save_states(self.args.model_chkp, best_dev_metric=None)
            #         patient = 0
            #     else:
            #         patient +=1
            #     if patient>4:
            #         # logger.info('No patient on the train loss! BREAK!\n')
            #         break

            logger.info('No validate [Epoch %d] train loss: %.4f \n' %(ep, train_loss))

        if not do_validate:
            self.save_states(self.args.model_chkp, best_dev_metric=None)
        return None

    def evaluate(self, test_loader,dev=False):
        test_pred_spans_labels = []
        test_gold_spans_labels = []
        self.model.eval()
        if dev:
            # do not write predict results when validating
            with torch.no_grad():
                    for i, batcher in enumerate(test_loader):
                        batch = batch_variable(batcher, self.vocabs)
                        batch.to_device(self.args.device)
                        pred_score = self.model(batch.bert_inp, batch.mask)
                        pred_tag_ids = self.model.tag_decode(pred_score, batch.mask)
                        seq_lens = batch.mask.sum(dim=1).tolist()
                        for j, l in enumerate(seq_lens):
                            pred_tags = self.vocabs['ner'].idx2inst(pred_tag_ids.cpu()[j][1:l].tolist())
                            gold_tags = batcher[j].ner_tags
                            pred_spans_label = extract_entity_span_label_BIO(pred_tags)
                            gold_spans_label = extract_entity_span_label_BIO(gold_tags)
                            sentence_tokens = batcher[j].tokens
                            test_pred_spans_labels.append(pred_spans_label)
                            test_gold_spans_labels.append(gold_spans_label)
                            assert len(pred_tags) == len(gold_tags)
        else:

            with open(self.predict_results_file, 'w') as f1:
                f1.write('')
            with open(self.predict_results_file, 'a') as dict_f:
                with torch.no_grad():
                    for i, batcher in enumerate(test_loader):
                        batch = batch_variable(batcher, self.vocabs)
                        batch.to_device(self.args.device)
                        pred_score = self.model(batch.bert_inp, batch.mask)
                        pred_tag_ids = self.model.tag_decode(pred_score, batch.mask)
                        seq_lens = batch.mask.sum(dim=1).tolist()
                        for j, l in enumerate(seq_lens):
                            pred_tags = self.vocabs['ner'].idx2inst(pred_tag_ids.cpu()[j][1:l].tolist())
                            gold_tags = batcher[j].ner_tags

                            pred_spans_label = extract_entity_span_label_BIO(pred_tags)
                            gold_spans_label = extract_entity_span_label_BIO(gold_tags)

                            dict_f.write('-------------' + '样本序号' +str(i*len(batcher)+j) + '-------------' + '\n')
                            sentence_tokens = batcher[j].tokens
                            dict_f.write(str(sentence_tokens) + '\n')

                            dict_f.write(str(pred_tags) + '\n')
                            dict_f.write(str(gold_tags) + '\n')

                            dict_f.write(str(pred_spans_label) + '\n')
                            dict_f.write(str(gold_spans_label) + '\n')

                            test_pred_spans_labels.append(pred_spans_label)
                            test_gold_spans_labels.append(gold_spans_label)

                            assert len(pred_tags) == len(gold_tags)
        num_true = 0
        num_pred=0
        num_gold=0
        for pred_spans_label,gold_spans_label in zip(test_pred_spans_labels,test_gold_spans_labels):
            for item in pred_spans_label:
                if item in gold_spans_label:
                    num_true+=1
            num_pred+=len(pred_spans_label)
            num_gold+=len(gold_spans_label)

        if num_pred == 0:
            precision = 0
        else:
            precision = num_true / num_pred

        if num_gold == 0:
            recall = 0
        else:
            recall = num_true / num_gold

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print('metric',dict(p=precision, r=recall, f=f1))
        return dict(p=precision, r=recall, f=f1)





def set_seeds(seed=1349):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    args = args_config()
    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    data_path = data_config('config/data_path.json')

    final_res = {'p': [], 'r': [], 'f': []}
    seed=1357
    set_seeds(seed)

    ########################################################

    train_set = load_data(data_path[args.genre]['train'])
    val_set = load_data(data_path[args.genre]['dev'])
    test_set = load_data(data_path[args.genre]['test'])

    trainer = Trainer(args, data_path,train_set=train_set,val_set=val_set,test_set=test_set)
    prf = trainer.run(do_validate=True)

    final_res['p'].append(prf['p'])
    final_res['r'].append(prf['r'])
    final_res['f'].append(prf['f'])

    logger.info('Final Result: %s' % final_res)

    #############################################################
