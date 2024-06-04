import argparse
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import random

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


def test_model(args):
    # 1. read concept-prop pairs and labels
    logging.info('load in concept-property-facet triples')
    cp_pairs, cp_labels = read_pair_data(args.cp_fn)
    cp_pairs = np.array(cp_pairs)
    cp_labels = np.array(cp_labels)

    # data loader
    dataloader = DataLoader(range(len(cp_labels)), batch_size=args.bsz, shuffle=False)

    # build model
    tokenizer = TOKENIZER_CLASS.get(args.bert_version).from_pretrained(args.bert_version)
    model = ConceptPropertyFacetModel(args)
    model.load_state_dict(torch.load(args.model_fn, map_location='cpu'))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info('use gpu')
        model.to('cuda')

    model.eval()
    test_logits = []
    # test_labels = []
    for step, batch_idx in enumerate(tqdm(dataloader, desc='test')):
        batch_pair = cp_pairs[batch_idx]
        # batch_label = cp_labels[batch_idx]

        # prepare concept token ids
        concept_sents = add_context(batch_pair[:, 0], mask_token=tokenizer.mask_token)
        concept_ids = tokenize_mask(tokenizer, concept_sents, args.max_seq_len)

        # prepare prop token ids
        prop_sents = add_context(batch_pair[:, 1], mask_token=tokenizer.mask_token)
        prop_ids = tokenize_mask(tokenizer, prop_sents, args.max_seq_len)

        if use_gpu:
            concept_ids = concept_ids.to('cuda')
            prop_ids = prop_ids.to('cuda')

        _, _, _, logits = model(concept_ids['input_ids'],
                                concept_ids['attention_mask'],
                                prop_ids['input_ids'],
                                prop_ids['attention_mask'],
                                prop_ids['input_ids'],
                                prop_ids['attention_mask']
                                )

        test_logits.extend(logits.detach().cpu())
        # test_labels.extend(cp_labels[batch_idx])
        torch.cuda.empty_cache()

    test_prob = torch.sigmoid(torch.tensor(test_logits))
    test_pred = (test_prob > 0.5) * 1.0
    scores = compute_scores(cp_labels, test_pred.numpy())

    with open(args.res_fn, 'w', encoding='utf-8') as f:
        for key in scores:
            f.write(key + ' : ' + str(scores[key]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp_fn', help='directory of concept property pairs',
                        default='../../data/extended_mcrae/concept_property_split/test_split_concept_property.tsv')
    parser.add_argument('-bsz', help='batch size', default=32, type=int)
    parser.add_argument('-max_seq_len', help='max sequence length', default=32, type=int)
    parser.add_argument('-model_fn', help='pre-trained model file name, used for finetuning',
                        default='../../facet_biencoder_bert-base-uncased_2e-05_32_0.07.pt')
    parser.add_argument('-res_fn', help='file name of results', default='../results/mcrae_con_prop_split.txt')
    parser.add_argument('-bert_version', help='bert version', default='bert-base-uncased')
    parser.add_argument('-tau', help='tau in contrastive loss', default=0.07, type=float)
    parser.add_argument('-alpha', help='alpha used for loss balance', default=0.5, type=float)
    args = parser.parse_args()

    log_file_path = init_logging_path('log', "biencoder")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    if not os.path.exists(os.path.dirname(args.res_fn)):
        os.makedirs(os.path.dirname(args.res_fn))

    test_model(args)
