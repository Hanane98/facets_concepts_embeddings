import argparse
import sys
import os
import pickle

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer

from src.utils import *
from src.models import ConceptPropertyFacetModel
from src.early_stop import *


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


def read_file(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data


def run(args):
    # 1. read concept or property file
    logging.info('load in concept or property names')
    if args.in_data_type == 'concept':
        concepts = np.array(read_file(args.in_fn))
        properties = np.array(['dummy_property'] * len(concepts))
    elif args.in_data_type in ['property', 'facet']:
        properties = np.array(read_file(args.in_fn))
        concepts = np.array(['dummy_concepts'] * len(properties))
    else:
        raise ValueError('in_data_type must be concept or property')

    # data loader
    dataloader = DataLoader(range(len(concepts)), batch_size=args.bsz, shuffle=False)

    # build model
    #tokenizer = TOKENIZER_CLASS.get(args.bert_version).from_pretrained(args.bert_version)
    tokenizer_class = TOKENIZER_CLASS.get(args.bert_version, BertTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.bert_version)


    model = ConceptPropertyFacetModel(args)
    model.load_state_dict(torch.load(args.model_fn, map_location='cpu'), strict=False)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info('use gpu')
        model.to('cuda')

    model.eval()
    concept_emb, property_emb = {}, {}
    for step, batch_idx in enumerate(tqdm(dataloader, desc='test')):
        # prepare concept token ids
        concept_sents = add_context(concepts[batch_idx], mask_token=tokenizer.mask_token)
        concept_ids = tokenize_mask(tokenizer, concept_sents, args.max_seq_len)

        # prepare prop token ids
        prop_sents = add_context(properties[batch_idx], mask_token=tokenizer.mask_token)
        prop_ids = tokenize_mask(tokenizer, prop_sents, args.max_seq_len)

        if use_gpu:
            concept_ids = concept_ids.to('cuda')
            prop_ids = prop_ids.to('cuda')

        con_emb, facet_prop_emb, prop_emb, _ = model(concept_ids['input_ids'],
                                                     concept_ids['attention_mask'],
                                                     prop_ids['input_ids'],
                                                     prop_ids['attention_mask'],
                                                     prop_ids['input_ids'],
                                                     prop_ids['attention_mask']
                                                     )

        if args.in_data_type == 'concept':
            for con, emb in zip(concepts[batch_idx], con_emb):
                concept_emb[con] = emb.detach().cpu().numpy()
        elif args.in_data_type == 'property':
            for prop, emb in zip(properties[batch_idx], prop_emb):
                property_emb[prop] = emb.detach().cpu().numpy()
        elif args.in_data_type == 'facet':
            for prop, emb in zip(properties[batch_idx], facet_prop_emb):
                property_emb[prop] = emb.detach().cpu().numpy()
        torch.cuda.empty_cache()

    if args.in_data_type == 'concept':
        with open(args.emb_fn, "wb") as pkl_file:
            pickle.dump(concept_emb, pkl_file)
    elif args.in_data_type in ['property', 'facet']:
        with open(args.emb_fn, "wb") as pkl_file:
            pickle.dump(property_emb, pkl_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_fn', help='directory of concepts or properties',
                        default='concepts.txt')
    parser.add_argument('-in_data_type', help='input data type: concept or property',
                        default='facet', choices=['concept', 'property', 'facet'])
    parser.add_argument('-bsz', help='batch size', default=32, type=int)
    parser.add_argument('-max_seq_len', help='max sequence length', default=32, type=int)
    parser.add_argument('-model_fn', help='pre-trained model file name, used for finetuning',
                        default='../../trained_models/facet_biencoder_bert-base-uncased_2e-05_128_0.07_pairs.pt')
    parser.add_argument('-emb_fn', help='file name of embeddings', default='embeedings/test.pkl')
    parser.add_argument('-bert_version', help='bert version', default='bert-base-uncased')
    parser.add_argument('-tau', help='tau in contrastive loss', default=0.07, type=float)
    parser.add_argument('-alpha', help='alpha used for loss balance', default=0.5, type=float)
    args = parser.parse_args()

    log_file_path = init_logging_path('log', "biencoder")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    if not os.path.exists(os.path.dirname(args.emb_fn)):
        os.makedirs(os.path.dirname(args.emb_fn))

    run(args)


