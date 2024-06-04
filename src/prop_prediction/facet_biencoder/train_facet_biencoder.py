import argparse
import logging
import sys
import os

import numpy as np

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
import torch.nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm, trange
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import random

from src.utils import *
from src.models import ConceptPropertyModel, ConceptPropertyFacetModel
from src.early_stop import *
from pytorch_metric_learning.samplers import MPerClassSampler


TOKENIZER_CLASS = {
    "bert-base-uncased": BertTokenizer,
    "bert-large-uncased": BertTokenizer,
    "bert-base-cased": BertTokenizer,
    "bert-large-cased": BertTokenizer,
    "roberta-base": RobertaTokenizer,
    "roberta-large": RobertaTokenizer,
    "deberta-base": DebertaTokenizer,
    "deberta-large": DebertaTokenizer,
}


def train_model(args):
    logging.info(str(args))
    # 1. read concept-prop pairs and labels
    logging.info('load in concept-property pairs')
    cp_pairs, cp_labels = read_pair_data(args.cp_fn)
    logging.info('load in property-facet pairs')
    pf_prop, pf_labels = read_pair_data(args.pf_fn)

    # repeat to align cp and pf numbers
    num_cp = len(cp_pairs)
    num_pf = len(pf_prop)
    num_facet = len(np.unique(pf_labels))
    if num_cp > num_pf:
        repeat_time = num_cp // num_pf + 1
        pf_prop = pf_prop * repeat_time
        pf_labels = pf_labels * repeat_time
    if num_cp < num_pf:
        repeat_time = num_pf // num_cp + 1
        cp_pairs = cp_pairs * repeat_time
        cp_labels = cp_labels * repeat_time

    cp_pairs = np.array(cp_pairs)
    cp_labels = np.array(cp_labels)
    pf_prop = np.array(pf_prop)
    pf_labels = np.array(pf_labels)

    # 2. split into train and dev set
    train_idx, dev_idx = train_test_split(range(min(len(pf_labels), len(cp_labels))), test_size=10000, stratify=cp_labels)

    # small data for test
    # train_idx = train_idx[:1000]
    # dev_idx = dev_idx[:100]

    # 3. dataloader
    logging.info('build dataloader')
    if args.bsz % args.m_per_class != 0 or args.m_per_class * num_facet < args.bsz:
        args.bsz = args.m_per_class * num_facet
        logging.info('modify batch size to ' + str(args.bsz))
    print(args.bsz)
    sampler = MPerClassSampler(pf_labels[train_idx], m=args.m_per_class, batch_size=args.bsz,
                               length_before_new_iter=len(pf_labels[train_idx]))
    train_dataloader = DataLoader(train_idx, batch_size=args.bsz, sampler=sampler)
    dev_dataloader = DataLoader(dev_idx, batch_size=args.bsz, shuffle=False)

    # 4. model
    logging.info('build model')
    model = ConceptPropertyFacetModel(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tokenizer = TOKENIZER_CLASS.get(args.bert_version).from_pretrained(args.bert_version)
    total_step_num = len(train_dataloader) * args.epoch
    lr_schedule = get_linear_schedule_with_warmup(optimizer, len(train_dataloader), total_step_num)

    # if model fn is given, then load it for fine-tuning
    if args.model_fn:
        model.load_state_dict(torch.load(args.model_fn))

    model_param = [args.model_prefix, args.bert_version, str(args.lr), str(args.bsz), str(args.tau)]
    if args.alpha:
        model_param.append(str(args.alpha))
    model_fn = os.path.join(args.model_dir, '_'.join(model_param) + '.pt')
    early_stopping = EarlyStopping(patience=5, verbose=False, path=model_fn, delta=1e-10)

    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)  # random batch training seed (shuffle) to ensure reproducibility

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info('use gpu')
        # n_gpu = torch.cuda.device_count()
        # if n_gpu > 1:
        #     logging.info('use multiple gpu')
        #     model = torch.nn.DataParallel(model)
        model.to('cuda')

    logging.info('start training')
    for epoch in trange(args.epoch, desc='Epoch'):
        model.train()
        train_loss = 0
        for step, batch_idx in enumerate(tqdm(train_dataloader, desc='Iteration')):
            batch_pair = cp_pairs[batch_idx]
            batch_pf_prop = pf_prop[batch_idx]
            batch_cp_label = torch.tensor(cp_labels[batch_idx]).float()
            batch_pf_label = torch.tensor(pf_labels[batch_idx]).float()

            # prepare concept token ids
            concept_sents = add_context(batch_pair[:, 0], mask_token=tokenizer.mask_token)
            concept_ids = tokenize_mask(tokenizer, concept_sents, args.max_seq_len)

            # prepare prop token ids for concept-prop pair
            cp_prop_sents = add_context(batch_pair[:, 1], mask_token=tokenizer.mask_token)
            cp_prop_ids = tokenize_mask(tokenizer, cp_prop_sents, args.max_seq_len)

            # prepare prop token ids for prop-facet pair
            pf_prop_sents = add_context(batch_pf_prop, mask_token=tokenizer.mask_token)
            pf_prop_ids = tokenize_mask(tokenizer, pf_prop_sents, args.max_seq_len)

            if use_gpu:
                concept_ids = concept_ids.to('cuda')
                cp_prop_ids = cp_prop_ids.to('cuda')
                pf_prop_ids = pf_prop_ids.to('cuda')
                batch_cp_label = batch_cp_label.to('cuda')
                batch_pf_label = batch_pf_label.to('cuda')

            optimizer.zero_grad()
            loss = model(concept_ids['input_ids'],
                         concept_ids['attention_mask'],
                         cp_prop_ids['input_ids'],
                         cp_prop_ids['attention_mask'],
                         pf_prop_ids['input_ids'],
                         pf_prop_ids['attention_mask'],
                         batch_cp_label,
                         batch_pf_label)[-1]

            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            lr_schedule.step()

            train_loss += loss.item()
        train_loss /= (step + 1)

        # validation
        model.eval()
        dev_loss = 0
        dev_logits = []
        dev_labels = []
        for step, batch_idx in enumerate(tqdm(dev_dataloader, desc='dev')):
            batch_pair = cp_pairs[batch_idx]
            batch_pf_prop = pf_prop[batch_idx]
            batch_cp_label = torch.tensor(cp_labels[batch_idx]).float()
            batch_pf_label = torch.tensor(pf_labels[batch_idx]).float()

            # prepare concept token ids
            concept_sents = add_context(batch_pair[:, 0], mask_token=tokenizer.mask_token)
            concept_ids = tokenize_mask(tokenizer, concept_sents, args.max_seq_len)

            # prepare prop token ids for concept-prop pair
            cp_prop_sents = add_context(batch_pair[:, 1], mask_token=tokenizer.mask_token)
            cp_prop_ids = tokenize_mask(tokenizer, cp_prop_sents, args.max_seq_len)

            # prepare prop token ids for prop-facet pair
            pf_prop_sents = add_context(batch_pf_prop, mask_token=tokenizer.mask_token)
            pf_prop_ids = tokenize_mask(tokenizer, pf_prop_sents, args.max_seq_len)

            if use_gpu:
                concept_ids = concept_ids.to('cuda')
                cp_prop_ids = cp_prop_ids.to('cuda')
                pf_prop_ids = pf_prop_ids.to('cuda')
                batch_cp_label = batch_cp_label.to('cuda')
                batch_pf_label = batch_pf_label.to('cuda')

            _, _, _, logits, loss = model(concept_ids['input_ids'],
                                          concept_ids['attention_mask'],
                                          cp_prop_ids['input_ids'],
                                          cp_prop_ids['attention_mask'],
                                          pf_prop_ids['input_ids'],
                                          pf_prop_ids['attention_mask'],
                                          batch_cp_label,
                                          batch_pf_label)

            dev_loss += loss.item()
            dev_logits.extend(logits.detach().cpu())
            dev_labels.extend(batch_cp_label.detach().cpu())
            torch.cuda.empty_cache()

        dev_loss /= (step + 1)

        dev_prob = torch.sigmoid(torch.tensor(dev_logits))
        dev_labels = torch.tensor(dev_labels)
        dev_pred = (dev_prob > 0.5) * 1.0
        dev_f1 = compute_scores(dev_labels, dev_pred)['binary_f1']

        logging.info("Epoch: %d | train loss: %.4f | dev loss: %.4f | dev_f1: %.4f ",
                     epoch + 1, train_loss, dev_loss, dev_f1)
        print(epoch + 1, train_loss, dev_loss, dev_f1)

        early_stopping(-dev_f1, model)

        if early_stopping.early_stop:
            logging.info('Early stop. Model trained.')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp_fn', help='directory of concept property pairs',
                        default='../../data/train_cp_pairs.tsv')
    parser.add_argument('-pf_fn', help='directory of property facet pairs',
                        default='../../data/train_pf_pairs.tsv')
    parser.add_argument('-epoch', help='number of epoch', default=100, type=int)
    parser.add_argument('-lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('-bsz', help='batch size', default=256, type=int)
    parser.add_argument('-model_dir', help='directory to store traied model', default='../trained_model/biencoder')
    parser.add_argument('-bert_version', help='bert version', default='bert-base-uncased')
    parser.add_argument('-weight_decay', help='weight decay', default=1e-2, type=float)
    parser.add_argument('-seed', help='random seed', default=88, type=int)
    parser.add_argument('-max_seq_len', help='max sequence length', default=20, type=int)
    parser.add_argument('-model_fn', help='pre-trained model file name, used for finetuning', default=None)
    parser.add_argument('-model_prefix', help='prefix of model file name', default='biencoder')
    parser.add_argument('-tau', help='tau in contrastive loss', default=0.07, type=float)
    parser.add_argument('-alpha', help='alpha used for loss balance', default=None, type=float)
    parser.add_argument('-m_per_class', help='m per class in a batch', default=5, type=int)
    args = parser.parse_args()

    log_file_path = init_logging_path('log', "facet_biencoder")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    train_model(args)
