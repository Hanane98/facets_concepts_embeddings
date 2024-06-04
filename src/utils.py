import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report


def init_logging_path(log_path, file_name):
    dir_log  = os.path.join(log_path,f"{file_name}/")
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