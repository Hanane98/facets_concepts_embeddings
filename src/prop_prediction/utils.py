import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import logging
import pandas as pd
import json
import torch
import numpy as np
import time


def set_logger(config):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    sub_log_directory = config.log_dir

    if sub_log_directory:
        if not os.path.exists(os.path.join("logs", config.log_dir)):
            os.makedirs(os.path.join("logs", config.log_dir))
        log_file_name = os.path.join(
            "logs",
            config.log_dir,
            f"log_{config.model_prefix}_{config.bert_version}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
        )
    else:
        log_file_name = os.path.join(
            "logs",
            f"log_{config.model_prefix}_{config.bert_version}_{time.strftime('%d-%m-%Y_%H-%M-%S')}.txt",
        )

    logging.basicConfig(
        level=logging.DEBUG,
        filename=log_file_name,
        filemode="w",
        format="%(asctime)s : %(levelname)s : %(name)s - %(message)s",
    )


log = logging.getLogger(__name__)


def init_logging_path(log_path, file_name):
    dir_log = os.path.join(log_path,f"{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    return dir_log


def add_context(words, context='means', mask_token='[MASK]'):
    sentences = []
    for w in words:
        sentences.append(' '.join([w, context, mask_token]))
    return sentences


def tokenize_mask(tokenizer, sentences, max_seq_len):
    return tokenizer(
        sentences,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def compute_scores(labels, preds):

    assert len(labels) == len(
        preds
    ), f"labels len: {len(labels)} is not equal to preds len {len(preds)}"

    scores = {
        "binary_f1": round(f1_score(labels, preds, average="binary"), 4),
        "micro_f1": round(f1_score(labels, preds, average="micro"), 4),
        "macro_f1": round(f1_score(labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(labels, preds, average="weighted"), 4),
        "accuracy": round(accuracy_score(labels, preds), 4),
        "classification report": classification_report(labels, preds, labels=[0, 1]),
        "confusion matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }

    return scores


def read_pair_data(file_name):
    data = []
    labels = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            if len(items) == 3:  # concept, property, label
                data.append([items[0], items[1]])
                labels.append(int(items[2]))
            elif len(items) == 2:  # property, facet id
                data.append(items[0])
                labels.append(int(items[1]))
    return data, labels


def read_triple_data(file_name):
    pairs = []
    concept_prop_labels = []
    prop_fact_labels = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            pairs.append([items[0], items[1]])
            concept_prop_labels.append(int(items[2]))
            prop_fact_labels.append(int(items[3]))
    return pairs, concept_prop_labels, prop_fact_labels


def read_train_data(file_name):

    data_df = pd.read_csv(
        file_name,
        sep="\t",
        header=None,
        names=["concept", "property", "label"],
    )

    data_df.drop_duplicates(inplace=True)
    data_df.dropna(inplace=True)

    # log.info(f"Concept Column is null ? : {data_df['concept'].isnull().any()}")
    # log.info(f"Property Column is null ? : {data_df['property'].isnull().any()}")
    # log.info(f"Label Column is null ? : {data_df['label'].isnull().any()}")

    data_df.dropna(subset=["concept"], inplace=True)
    data_df.dropna(subset=["property"], inplace=True)
    data_df.dropna(subset=["label"], inplace=True)

    data_df.reset_index(drop=True, inplace=True)

    # log.info(f"Total Data size {data_df.shape}")

    return data_df[["concept", "property"]], data_df[["label"]]


def read_train_and_test_data(train_file_path, test_file_path):

    train_df = pd.read_csv(
        train_file_path,
        sep="\t",
        header=None,
        names=["concept", "property", "label"],
    )

    test_df = pd.read_csv(
        test_file_path,
        sep="\t",
        header=None,
        names=["concept", "property", "label"],
    )

    train_and_test_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)

    log.info(
        f"Concept Column is null ? : {train_and_test_df['concept'].isnull().any()}"
    )
    log.info(
        f"Property Column is null ? : {train_and_test_df['property'].isnull().any()}"
    )
    log.info(f"Label Column is null ? : {train_and_test_df['label'].isnull().any()}")

    train_and_test_df.dropna(subset=["concept"], inplace=True)
    train_and_test_df.dropna(subset=["property"], inplace=True)
    train_and_test_df.dropna(subset=["label"], inplace=True)
    train_and_test_df.dropna(how="any", inplace=True)

    train_and_test_df.drop_duplicates(inplace=True)

    train_and_test_df.reset_index(inplace=True, drop=True)

    print()
    print("+++++++++++ Index Unique +++++++++++++")
    print(train_and_test_df.index.is_unique)
    print(
        f"Duplicated Index Unique : {train_and_test_df.loc[train_and_test_df.index.duplicated(), :]}"
    )
    print("train_and_test_df")
    print(train_and_test_df.head())
    print()

    train_and_test_df["con_id"] = int(-1)
    train_and_test_df["prop_id"] = int(-2)

    unique_concept = train_and_test_df["concept"].unique()
    unique_property = train_and_test_df["property"].unique()

    num_unique_concept = len(unique_concept)
    num_unique_property = len(unique_property)

    log.info(f"Train and Test Data DF shape : {train_and_test_df.shape}")

    log.info(
        f"Number of Unique Concepts in Train and Test Combined DF : {num_unique_concept}"
    )
    log.info(f"Unique Concepts in Train and Test Combined DF : {unique_concept}")

    log.info(
        f"Number of Unique Property in Train and Test Combined DF : {num_unique_property}"
    )
    log.info(f"Unique Property in Train and Test Combined DF : {unique_property}")

    con_to_id_dict = {con: id for id, con in enumerate(unique_concept)}
    con_ids = list(con_to_id_dict.values())
    log.info(f"Concept ids in con_to_id_dict : {con_ids}")

    train_and_test_df.set_index("concept", inplace=True)

    for con in unique_concept:
        train_and_test_df.loc[con, "con_id"] = con_to_id_dict.get(con)

    train_and_test_df.reset_index(inplace=True)

    log.info("Train Test DF after assigning 'con_id'")
    log.info(train_and_test_df.sample(n=10))

    assert sorted(con_ids) == sorted(
        train_and_test_df["con_id"].unique()
    ), "Assigned 'con_ids' do not match"

    #################################
    prop_to_id_dict = {prop: id for id, prop in enumerate(unique_property)}
    prop_ids = list(prop_to_id_dict.values())

    log.info(f"Number of Property ids in prop_to_id_dict : {len(prop_ids)}")
    log.info(f"Property ids in prop_to_id_dict : {prop_ids}")

    train_and_test_df.set_index("property", inplace=True)

    for prop in unique_property:
        train_and_test_df.loc[prop, "prop_id"] = prop_to_id_dict.get(prop)

    train_and_test_df.reset_index(inplace=True)

    log.info("Train Test DF after assigning 'prop_id'")
    log.info(train_and_test_df.sample(n=15))

    assert sorted(prop_ids) == sorted(
        train_and_test_df["prop_id"].unique()
    ), "Assigned 'prop_ids' do not match"

    return train_and_test_df


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

