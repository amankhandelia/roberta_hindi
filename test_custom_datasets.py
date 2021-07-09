"""
Step1: Go inside datasets/ and execute prepare_datasets.sh (./prepare_datasets.sh) Give necessary executable permissions if needed

Step2: provide full path to dataset scripts below (/home/<user>/./...), relative path doesn't work with datasets lib.
"""

import os
import sys
import regex
from datasets import load_dataset, concatenate_datasets
# base_path = os.path.expanduser("~")
base_path = os.getcwd()
NUM_PROC = 1
DATASET_SAVE_PATH = "/home/rtx/work/dk/hf/vo"

def concatenate_hindi_text_short_summarization_corpus_row(example, cols):
    text = ""
    for col in cols:
        if example[col]:
            text += example[col]
    example["text"] = text
    return example

def preprocess_hindi_text_short_summarization_corpus(dataset):
    cols = DATASET_DICT["hindi-text-short-summarization-corpus"]["cols_to_concatenate"]
    remove_cols = DATASET_DICT["hindi-text-short-summarization-corpus"]["cols_to_remove"]
    dataset = dataset.map(lambda x: concatenate_hindi_text_short_summarization_corpus_row(x, cols), 
        remove_columns=remove_cols, num_proc=NUM_PROC)
    return dataset

def concatenate_hindi_text_short_and_large_summarization_corpus(example,cols):
    text = ""
    for col in cols:
        if example[col]:
            text += example[col]
    example["text"] = text
    return example

def preprocess_hindi_text_short_and_large_summarization_corpus(dataset):
                        
    cols = DATASET_DICT["hindi-text-short-and-large-summarization-corpus"]["cols_to_concatenate"]
    remove_cols = DATASET_DICT["hindi-text-short-and-large-summarization-corpus"]["cols_to_remove"]
    dataset = dataset.map(lambda x: concatenate_hindi_text_short_and_large_summarization_corpus(x,cols),
        remove_columns=remove_cols, num_proc=NUM_PROC)

    return dataset

def concatenate_indic_glue_wiki_ner_row(example, col):
    text = " ".join(example[col])
    example["text"] = text
    return example


def preprocess_indic_glue_wiki_ner(dataset):
    # Only one column containing list of words eg: ["hello", "world"]
    col = DATASET_DICT["indic-glue"]["cols_to_concatenate"][0]
    remove_cols = DATASET_DICT["indic-glue"]["cols_to_remove"]
    dataset = dataset.map(lambda x: concatenate_indic_glue_wiki_ner_row(x, col), 
        remove_columns=remove_cols, num_proc=NUM_PROC)
    return dataset


def concatenate_samanantar_row(example, cols):
    text = ""
    for col in cols:
        if example[col]:
            text += example[col]
    example["text"] = text
    return example


def preprocess_samanantar(dataset):
    cols = DATASET_DICT["samanantar"]["cols_to_concatenate"]
    remove_cols = DATASET_DICT["samanantar"]["cols_to_remove"]
    dataset = dataset.map(lambda x: concatenate_samanantar_row(x, cols), 
        remove_columns=remove_cols, num_proc=NUM_PROC)
    return dataset


def concatenate_oldnewspapershindi_row(example, cols):
    text = ""
    for col in cols:
        if example[col]:
            text += example[col]
    example["text"] = text
    return example


def preprocess_oldnewspapershindi(dataset):
    cols = DATASET_DICT["oldnewspapershindi"]["cols_to_concatenate"]
    remove_cols = DATASET_DICT["oldnewspapershindi"]["cols_to_remove"]
    dataset = dataset.map(lambda x: concatenate_oldnewspapershindi_row(x, cols), 
        remove_columns=remove_cols, num_proc=NUM_PROC)
    return dataset


def concatenate_hindi_wiki_articles_172k_row(example, cols):
    text = ""
    for col in cols:
        if example[col]:
            text += example[col]
    example["text"] = text
    return example


def preprocess_hindi_wiki_articles_172k(dataset):
    cols = DATASET_DICT["hindi-wikipedia-articles-172k"]["cols_to_concatenate"]
    remove_cols = DATASET_DICT["hindi-wikipedia-articles-172k"]["cols_to_remove"]
    dataset = dataset.map(lambda x: concatenate_hindi_wiki_articles_172k_row(x, cols), 
        remove_columns=remove_cols)
    return dataset


def replace_non_devanagiri(example):
    example["text"] = regex.sub(r'\P{Devanagari}+', ' ', example["text"])
    return example

# NOTE: Adjust these paths to reflect full paths appropriately
DATASET_DICT = {
    "hindi-text-short-summarization-corpus": {
        "is_custom": True,
        "path": base_path + "/datasets/hindi-text-short-summarization-corpus",
        "split_names": ["train", "test"],
        "cols_to_concatenate": ["headline", "article"],
        "cols_to_remove": ["headline", "article"],
        "configuration": None,
        "preprocess_fn": preprocess_hindi_text_short_summarization_corpus
    },
    "hindi-text-short-and-large-summarization-corpus": {
        "is_custom": True,
        "path": base_path + "/datasets/hindi-text-short-and-large-summarization-corpus",
        "split_names": ["train", "test"],
        "cols_to_concatenate": ["headline", "article"],
        "cols_to_remove": ["headline", "article","summary"],
        "configuration": None,
        "preprocess_fn": preprocess_hindi_text_short_and_large_summarization_corpus
    },
    "indic-glue": {
        "is_custom": True,
        "path": base_path + "/datasets/indic-glue",
        "split_names": ["train", "test"],
        "configuration": "wiki-ner.hi",
        "cols_to_concatenate": ["tokens"],
        "cols_to_remove": ["tokens", "ner_tags", "additional_info"],
        "preprocess_fn": preprocess_indic_glue_wiki_ner
    },
    "samanantar": {
        "is_custom": True,
        "path": base_path + "/datasets/samanantar",
        "split_names": ["train"],
        "cols_to_concatenate": ["text"],
        "cols_to_remove": [],
        "configuration": None,
        "preprocess_fn": preprocess_samanantar
    },
    "oldnewspapershindi": {
        "is_custom": True,
        "path": base_path + "/datasets/oldnewspapershindi",
        "split_names": ["train"],
        "cols_to_concatenate": ["text"],
        "cols_to_remove": ["source"],
        "configuration": None,
        "preprocess_fn": preprocess_oldnewspapershindi
    },
    "hindi-wikipedia-articles-172k": {
        "is_custom": True,
        "path": base_path + "/datasets/hindi-wikipedia-articles-172k",
        "split_names": ["train", "test"],
        "cols_to_concatenate": ["text"],
        "cols_to_remove": [],
        "configuration": None,
        "preprocess_fn": preprocess_hindi_wiki_articles_172k
    },
}

def load_and_concatenate(datasets_list, print_test_row=False):
    processed_datasets = []
    for dataset_id in datasets_list:
        if dataset_id not in DATASET_DICT:
            print("ERROR dataset config not found", dataset_id)
            sys.exit(0)

        for split_name in DATASET_DICT[dataset_id]["split_names"]:

            if not DATASET_DICT[dataset_id]["is_custom"]:
                # Load dataset with name directly
                pass
            else:
                dataset = load_dataset(DATASET_DICT[dataset_id]["path"],
                    DATASET_DICT[dataset_id]["configuration"], split=split_name)
            processed_dataset = DATASET_DICT[dataset_id]["preprocess_fn"](dataset)
            processed_datasets.append(processed_dataset)
    if print_test_row:
        print(processed_datasets)
        for d in processed_datasets:
            print(d[0])
    concatenated_dataset = concatenate_datasets(processed_datasets)
    return concatenated_dataset

datasets_list = [
    "hindi-text-short-summarization-corpus",
    "hindi-text-short-and-large-summarization-corpus",
    "indic-glue",
    "samanantar",
    "oldnewspapershindi",
    "hindi-wikipedia-articles-172k",
]

dataset = load_and_concatenate(datasets_list, print_test_row=True)
shuffle_dataset = dataset.shuffle(seed=42)

print("Removing non Devanagari characters")
shuffle_dataset.map(replace_non_devanagiri) #, num_proc=os.cpu_count() - 1)

shuffle_dataset.save_to_disk(DATASET_SAVE_PATH)
print("Total rows:", len(shuffle_dataset))
print("Sample: ", shuffle_dataset[42]["text"])