import itertools
import logging

import os
import sys
from argparse import ArgumentParser
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../../"
sys.path.append(os.path.abspath(project_path))

import pandas as pd
from tqdm import tqdm, trange

from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from src.models import ConceptPropertyFacetModel
from src.early_stop import *
from torch.utils.data import DataLoader

from src.utils import (
    compute_scores,
    read_train_data,
    set_seed,
    read_train_and_test_data,
    add_context,
    tokenize_mask,
    set_logger
)


log = logging.getLogger()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_model(args, mask_token_id):
    # create model
    log.info("create model")
    model = ConceptPropertyFacetModel(args, mask_token_id)
    return model


def load_pretrained_model(args):
    # create model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
    model = create_model(args, tokenizer.mask_token_id)
    # load pretrained model
    model.load_state_dict(torch.load(args.model_fn))
    log.info("model loaded")
    return model, tokenizer


def train_single_epoch(model, tokenizer, train_df, train_dataloader, optimizer, lr_schedule):
    model.to(device)
    model.train()
    cp_pairs = train_df[['concept', 'property']].values
    cp_labels = train_df[['label']].values.flatten()

    train_loss = 0.0
    for step, batch_idx in enumerate(tqdm(train_dataloader, desc='Iteration')):
        batch_pair = cp_pairs[batch_idx]
        if batch_pair.ndim == 1:
            continue
        batch_cp_label = torch.tensor(cp_labels[batch_idx]).float()

        # prepare concept token ids
        concept_sents = add_context(batch_pair[:, 0], mask_token=tokenizer.mask_token)
        concept_ids = tokenize_mask(tokenizer, concept_sents, args.max_seq_len)

        # prepare prop token ids for concept-prop pair
        cp_prop_sents = add_context(batch_pair[:, 1], mask_token=tokenizer.mask_token)
        cp_prop_ids = tokenize_mask(tokenizer, cp_prop_sents, args.max_seq_len)

        concept_ids = concept_ids.to(device)
        cp_prop_ids = cp_prop_ids.to(device)
        batch_cp_label = batch_cp_label.to(device)

        optimizer.zero_grad()
        loss = model(concept_ids['input_ids'],
                     concept_ids['attention_mask'],
                     cp_prop_ids['input_ids'],
                     cp_prop_ids['attention_mask'],
                     cp_prop_ids['input_ids'],
                     cp_prop_ids['attention_mask'],
                     batch_cp_label)[-1]

        if isinstance(model, torch.nn.DataParallel):
            loss = loss.mean()
        loss.backward()
        optimizer.step()
        lr_schedule.step()

        train_loss += loss.item()
    train_loss /= (step + 1)
    return model, train_loss


def evaluate(model, tokenizer, train_df, eval_dataloader):
    model.eval()
    cp_pairs = train_df[['concept', 'property']].values
    cp_labels = train_df[['label']].values.flatten()

    eval_loss = 0.0
    eval_logits = []
    eval_labels = []
    for step, batch_idx in enumerate(tqdm(eval_dataloader, desc='eval')):
        batch_pair = cp_pairs[batch_idx]
        if batch_pair.ndim == 1:
            continue
        batch_cp_label = torch.tensor(cp_labels[batch_idx]).float()

        # prepare concept token ids
        concept_sents = add_context(batch_pair[:, 0], mask_token=tokenizer.mask_token)
        concept_ids = tokenize_mask(tokenizer, concept_sents, args.max_seq_len)

        # prepare prop token ids for concept-prop pair
        cp_prop_sents = add_context(batch_pair[:, 1], mask_token=tokenizer.mask_token)
        cp_prop_ids = tokenize_mask(tokenizer, cp_prop_sents, args.max_seq_len)

        concept_ids = concept_ids.to(device)
        cp_prop_ids = cp_prop_ids.to(device)
        batch_cp_label = batch_cp_label.to(device)

        _, _, _, logits, loss = model(concept_ids['input_ids'],
                                      concept_ids['attention_mask'],
                                      cp_prop_ids['input_ids'],
                                      cp_prop_ids['attention_mask'],
                                      cp_prop_ids['input_ids'],
                                      cp_prop_ids['attention_mask'],
                                      batch_cp_label)

        eval_loss += loss.item()
        eval_logits.extend(logits.detach().cpu())
        eval_labels.extend(batch_cp_label.detach().cpu())
        torch.cuda.empty_cache()

    eval_loss /= (step + 1)

    eval_prob = torch.sigmoid(torch.tensor(eval_logits))
    eval_labels = torch.tensor(eval_labels)
    eval_pred = (eval_prob > 0.5) * 1.0
    eval_score = compute_scores(eval_labels, eval_pred)
    return eval_loss, eval_score, cp_labels, eval_pred


def train(model, tokenizer, args, train_df, train_idx=None, dev_idx=None, fold=None):
    log.info("Initialising datasets...")

    if train_idx is None and dev_idx is None:
        # split into train and dev set
        train_idx, dev_idx = train_test_split(range(len(train_df)),
                                              test_size=0.2,
                                              stratify=train_df[['label']].values.flatten())
    train_dataloader = DataLoader(train_idx,
                                  batch_size=args.bsz,
                                  shuffle=True)

    dev_dataloader = DataLoader(dev_idx,
                                batch_size=args.bsz,
                                shuffle=False)

    # -------------------- Preparation for training  ------------------- #
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step_num = len(train_dataloader) * args.epoch
    lr_schedule = get_linear_schedule_with_warmup(optimizer, len(train_dataloader), total_step_num)

    model_param = [args.model_prefix, args.bert_version, str(args.lr), str(args.bsz), str(args.tau)]
    if args.alpha:
        model_param.append(str(args.alpha))
    model_fn = '_'.join(model_param) + '.pt'

    if fold is not None:
        model_save_path = os.path.join(
            args.model_dir,
            f"fold_{fold}_" + model_fn,
        )
    else:
        model_save_path = os.path.join(
            args.model_dir,
            model_fn,
        )

    log.info(f"best_model_path : {model_save_path}")

    early_stopping = EarlyStopping(patience=20, verbose=False, path=model_save_path, delta=1e-10)

    logging.info('start training')
    for epoch in trange(args.epoch, desc='Epoch'):
        model, train_loss = train_single_epoch(model,
                                               tokenizer,
                                               train_df,
                                               train_dataloader,
                                               optimizer,
                                               lr_schedule)

        # evaluation
        dev_loss, dev_score, _, _ = evaluate(model, tokenizer, train_df, dev_dataloader)
        dev_f1 = dev_score['binary_f1']
        logging.info("Epoch: %d | train loss: %.4f | dev loss: %.4f | dev_f1: %.4f ",
                     epoch + 1, train_loss, dev_loss, dev_f1)
        print(epoch + 1, train_loss, dev_loss, dev_f1)

        early_stopping(-dev_f1, model)

        if early_stopping.early_stop:
            logging.info('Early stop. Model trained.')
            break
    return model, model_save_path


def model_selection_cross_validation(args, concept_property_df, label_df):

    skf = StratifiedKFold(n_splits=5)
    train_df = pd.concat(
        (concept_property_df, label_df), axis=1, ignore_index=True
    )

    for fold_num, (train_index, test_index) in enumerate(
        skf.split(concept_property_df, label_df)
    ):
        log.info(f"Initialising training for fold : {fold_num + 1}")

        log.info(f"Loading the fresh model for fold : {fold_num + 1}")
        model, tokenizer = load_pretrained_model(args)

        train(model, tokenizer, args, train_df, train_idx=train_index, dev_idx=test_index)


def model_evaluation_property_cross_validation(args):

    log.info(f"Training the model with PROPERTY cross validation")
    log.info(f"Parameter 'do_cv' is : {args.do_cv}")
    log.info(f"Parameter 'cv_type' is : {args.cv_type}")

    train_and_test_df = read_train_and_test_data(args.train_file_path, args.test_file_path)

    train_and_test_df.drop("con_id", axis=1, inplace=True)
    train_and_test_df.set_index("prop_id", inplace=True)

    prop_ids = np.sort(train_and_test_df.index.unique())

    test_fold_mapping = {
        fold: test_prop_id
        for fold, test_prop_id in enumerate(np.array_split(prop_ids, 5))
    }

    log.info(f"unique prop_ids in train_and_test_df : {prop_ids}")
    log.info(f"Test Fold Mapping")
    for key, value in test_fold_mapping.items():
        log.info(f"Fold - {key} : Test Prop id length :{len(value)}")
        log.info(f"{key} : {value}")

    label, preds = [], []

    for fold, test_prop_id in test_fold_mapping.items():

        log.info("\n")
        log.info("^" * 50)
        log.info(f"Training the model on fold : {fold}")
        log.info(f"The model will be tested on prop_ids : {test_prop_id}")

        train_df = train_and_test_df.drop(index=test_prop_id, inplace=False)
        test_df = train_and_test_df[train_and_test_df.index.isin(test_prop_id)]

        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)

        log.info(
            f"For fold : {fold}, Train DF shape : {train_df.shape}, Test DF shape :{test_df.shape}"
        )

        log.info("Asserting no overlap in train and test data")

        df1 = train_df.merge(
            test_df,
            how="inner",
            on=["concept", "property", "prop_id", "label"],
            indicator=False,
        )
        df2 = train_df.merge(test_df, how="inner", on=["prop_id"], indicator=False)

        assert df1.empty
        assert df2.empty

        log.info("Assertion Passed !!!")

        train_df = train_df[["concept", "property", "label"]]
        test_df = test_df[["concept", "property", "label"]]

        load_pretrained = args.load_pretrained

        if load_pretrained:
            log.info(f"load_pretrained is : {load_pretrained}")
            log.info(f"Loading Pretrained Model ...")
            model, tokenizer = load_pretrained_model(args)
        else:
            # Untrained LM is Loaded  - for baselines results
            log.info(f"load_pretrained is : {load_pretrained}")
            tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
            model = create_model(args, tokenizer.mask_token_id)

        model, best_model_path = train(model, tokenizer, args, train_df, fold=fold)

        log.info(f"Test scores for fold :  {fold}")

        _, fold_label, fold_preds = test_best_model(args, model, test_df, best_model_path)

        label.append(fold_label)
        preds.append(fold_preds)

        log.info(f"Fold : {fold} label shape - {fold_label.shape}")
        log.info(f"Fold : {fold} preds shape - {fold_preds.shape}")

    log.info(f"\n {'*' * 50}")
    log.info(f"Test scores for all the Folds")
    label = np.concatenate(label, axis=0)
    preds = np.concatenate(preds, axis=0).flatten()

    log.info(f"All labels shape : {label.shape}")
    log.info(f"All preds shape : {preds.shape}")

    scores = compute_scores(label, preds)
    scores['test_mode'] = args.test_mode
    scores['bert_version'] = args.bert_version
    # scores['model_name'] = os.path.basename(best_model_path)

    for key, value in scores.items():
        log.info(f"{key} : {value}")

    write_results(args.res_fn, scores)


def model_evaluation_concept_property_cross_validation(args):

    log.info(f"Training the model with CONCEPT-PROPERTY cross validation")
    log.info(f"Parameter 'do_cv' is : {args.do_cv}")
    log.info(f"Parameter 'cv_type' is : {args.cv_type}")

    train_and_test_df = read_train_and_test_data(args.train_file_path, args.test_file_path)

    print("train_and_test_df")
    print(train_and_test_df)

    con_ids = np.sort(train_and_test_df["con_id"].unique())
    prop_ids = np.sort(train_and_test_df["prop_id"].unique())

    con_folds = {
        fold: test_con_id
        for fold, test_con_id in enumerate(np.array_split(np.asarray(con_ids), 3))
    }

    prop_folds = {
        fold: test_prop_id
        for fold, test_prop_id in enumerate(np.array_split(np.asarray(prop_ids), 3))
    }

    con_prop_test_fold_combination = list(itertools.product(con_folds.keys(), repeat=2))

    log.info(f"con_prop_test_fold_combination : {list(con_prop_test_fold_combination)}")

    log.info(f"Test Concept Fold Mapping")
    for key, value in con_folds.items():
        log.info(f"Fold {key} : Test Concept id Length {len(value)}")
        log.info(f"{key} : {value}")

    log.info(f"Test Property Fold Mapping")
    for key, value in prop_folds.items():
        log.info(f"Fold {key} : Test Property id Length {len(value)}")
        log.info(f"{key} : {value}")

    label, preds = [], []
    for fold, (test_con_fold, test_prop_fold) in enumerate(
        con_prop_test_fold_combination
    ):
        log.info(
            f"Fold {fold}, Test Concept fold: {test_con_fold}, Test Property Fold: {test_prop_fold}"
        )

        test_con_id = con_folds.get(test_con_fold)
        test_prop_id = prop_folds.get(test_prop_fold)

        log.info(f"Concept ids on which the model will be tested : {test_con_id}")
        log.info(f"Property ids on which the model will be tested : {test_prop_id}")

        train_and_test_df.set_index("con_id", inplace=True)

        con_id_df = train_and_test_df[train_and_test_df.index.isin(test_con_id)]
        con_id_df.reset_index(inplace=True)
        con_id_df.set_index("prop_id", inplace=True)

        test_df = con_id_df[con_id_df.index.isin(test_prop_id)]

        train_df = train_and_test_df.drop(index=test_con_id, inplace=False)
        train_df.reset_index(inplace=True)

        train_df = train_df[~train_df["property"].isin(test_df["property"].unique())]

        train_df.reset_index(inplace=True)
        test_df.reset_index(inplace=True)
        train_and_test_df.reset_index(inplace=True)

        log.info("Asserting no overlap in train and test data")

        df1 = train_df.merge(
            test_df,
            how="inner",
            on=["concept", "property", "con_id", "prop_id", "label"],
            indicator=False,
        )
        df2 = train_df.merge(test_df, how="inner", on=["con_id"], indicator=False)
        df3 = train_df.merge(test_df, how="inner", on=["prop_id"], indicator=False)

        assert df1.empty
        assert df2.empty
        assert df3.empty

        log.info("Assertion Passed !!!")

        train_df = train_df[["concept", "property", "label"]]
        test_df = test_df[["concept", "property", "label"]]

        log.info(
            f"For fold : {fold}, Train DF shape : {train_df.shape}, Test DF shape :{test_df.shape}"
        )
        log.info(f"Train DF Columns : {train_df.columns}")
        log.info(f"Test Df Columns : {test_df.columns}")

        load_pretrained = args.load_pretrained

        if load_pretrained:
            log.info(f"load_pretrained is : {load_pretrained}")
            log.info(f"Loading Pretrained Model ...")
            model, tokenizer = load_pretrained_model(args)
        else:
            # Untrained LM is Loaded  - for baselines results
            log.info(f"load_pretrained is : {load_pretrained}")
            tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
            model = create_model(args, tokenizer.mask_token_id)

        model, best_model_path = train(model, tokenizer, args, train_df, fold=fold)

        log.info(f"Test scores for fold :  {fold}")

        _, fold_label, fold_preds = test_best_model(args, model, test_df, best_model_path)

        label.append(fold_label)
        preds.append(fold_preds)

        log.info(f"Fold : {fold} label shape - {fold_label.shape}")
        log.info(f"Fold : {fold} preds shape - {fold_preds.shape}")

    log.info(f"\n {'*' * 50}")
    log.info(f"Test scores for all the Folds")

    label = np.concatenate(label, axis=0)
    preds = np.concatenate(preds, axis=0)

    log.info(f"All labels shape : {label.shape}")
    log.info(f"All preds shape : {preds.shape}")

    scores = compute_scores(label, preds)
    scores['test_mode'] = args.test_mode
    scores['bert_version'] = args.bert_version
    # scores['model_name'] = os.path.basename(best_model_path)

    for key, value in scores.items():
        log.info(f"{key} : {value}")

    write_results(args.res_fn, scores)


def test_best_model(args, model, test_df, best_model_path):

    log.info(f"\n {'*' * 50}")
    log.info(f"Testing the fine tuned model")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_version)

    # load finetuned model
    log.info(f"Testing the best model : {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))

    model.eval()
    model.to(device)

    if test_df is None:
        concept_property_df, label_df = read_train_data(args.test_file_path)
        test_df = pd.concat((concept_property_df, label_df), axis=1)

    test_dataloader = DataLoader(range(len(test_df)),
                                 batch_size=args.bsz,
                                 shuffle=False)

    _, test_scores, test_labels, test_preds = evaluate(model, tokenizer, test_df, test_dataloader)

    log.info(f"Test Metrices")

    for key, value in test_scores.items():
        log.info(f"{key} : {value}")

    test_scores['test_mode'] = args.test_mode
    test_scores['bert_version'] = args.bert_version
    test_scores['model_name'] = os.path.basename(best_model_path)

    return test_scores, test_labels, test_preds


def write_results(res_fn, test_scores):
    with open(res_fn, 'a+', encoding='utf-8') as f:
        for key in test_scores:
            f.write(key + ' : ' + str(test_scores[key]) + '\n')
        f.write('\n\n\n')


if __name__ == "__main__":
    set_seed(12345)

    parser = ArgumentParser(description="Fine tune configuration")
    parser.add_argument('-tau', help='tau in contrastive loss', default=0.07, type=float)
    parser.add_argument('-alpha', help='alpha used for loss balance', default=None, type=float)
    parser.add_argument('-bert_version', help='bert version', default='bert-base-uncased')
    parser.add_argument('-model_fn', help='pre-trained model file name, used for finetuning',
                        default="../trained_model/biencoder/finetuned_biencoder_bert-base-uncased_1e-05_32_0.07.pt")
    parser.add_argument('-epoch', help='number of epoch', default=1, type=int)
    parser.add_argument('-lr', help='learning rate', default=1e-5, type=float)
    parser.add_argument('-bsz', help='batch size', default=32, type=int)
    parser.add_argument('-model_dir', help='directory to store traied model', default='../trained_model/biencoder')
    parser.add_argument('-weight_decay', help='weight decay', default=1e-2, type=float)
    parser.add_argument('-max_seq_len', help='max sequence length', default=16, type=int)
    parser.add_argument('-model_prefix', help='prefix of model file name', default='finetuned_biencoder')
    parser.add_argument('-do_cv', help='do cross validation or not', action='store_true', default=False)
    parser.add_argument('-cv_type', help='cross validation type', default=None,
                        choices=['model_evaluation_property_split',
                                 'model_evaluation_concept_property_split'])
    parser.add_argument('-train_file_path', help='train file path',
                        default='../../data/extended_mcrae/train_mcrae.tsv')
    parser.add_argument('-test_file_path', help='test file path',
                        default='../../data/extended_mcrae/test_mcrae.tsv')
    parser.add_argument('-log_dir', help='log directory', default='mcrae_bert_logs')
    parser.add_argument('-load_pretrained', help='load pretrained or not',
                        action='store_true', default=False)
    parser.add_argument('-res_fn', help='file name of results', default='')
    parser.add_argument('-test_mode', help='test mode', choices=['con', 'pcv', 'cpcv'], default=None)

    args = parser.parse_args()
    set_logger(args)
    log.info(str(args))

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    if not os.path.exists(os.path.dirname(args.res_fn)):
        os.makedirs(os.path.dirname(args.res_fn))

    if not args.model_fn:
        raise ValueError("model_fn must be given")

    if not os.path.exists(args.model_fn):
        raise FileNotFoundError(args.model_fn + " is not found")

    if args.do_cv:
        cv_type = args.cv_type
        if cv_type == "model_selection":
            log.info(
                f"Cross Validation for Hyperparameter Tuning - that is Model Selection"
            )
            log.info("Reading Input Train File")
            concept_property_df, label_df = read_train_data(args.train_file_path)

            assert concept_property_df.shape[0] == label_df.shape[0]

            model_selection_cross_validation(args, concept_property_df, label_df)

        elif cv_type == "model_evaluation_property_split":

            log.info(f'Parameter do_cv : {args.do_cv}')
            log.info(
                "Cross Validation for Model Evaluation - Data Splited on Property basis"
            )
            log.info(f"Parameter cv_type : {cv_type}")

            model_evaluation_property_cross_validation(args)

        elif cv_type == "model_evaluation_concept_property_split":

            log.info(f'Parameter do_cv : {args.do_cv}')
            log.info(
                "Cross Validation for Model Evaluation - Data Splited on both Concept and Property basis"
            )
            log.info(f"Parameter cv_type : {cv_type}")

            model_evaluation_concept_property_cross_validation(args)

    else:
        log.info(f"Training the model without cross validation")
        log.info(f"Parameter 'do_cv' is {args.do_cv}")

        log.info("Reading Input Train File")
        concept_property_df, label_df = read_train_data(args.train_file_path)
        assert concept_property_df.shape[0] == label_df.shape[0]

        train_df = pd.concat((concept_property_df, label_df), axis=1)

        log.info(f"Train DF shape : {train_df.shape}")

        load_pretrained = args.load_pretrained

        log.info(f"load_pretrained is : {load_pretrained}")
        log.info(f"Loading Pretrained Model ...")
        model, tokenizer = load_pretrained_model(args)

        model, best_model_path = train(model, tokenizer, args, train_df)
        test_scores, _, _ = test_best_model(args, model, test_df=None, best_model_path=best_model_path)
        write_results(args.res_fn, test_scores)
