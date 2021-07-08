"""
Step1: Go inside datasets/ and execute prepare_datasets.sh (./prepare_datasets.sh) Give necessary executable permissions if needed

Step2: provide full path to dataset scripts below (/home/<user>/./...), relative path doesn't work with datasets lib.

"""

import os
import sys
from datasets import load_dataset, concatenate_datasets
# base_path = os.path.expanduser("~")
base_path = os.getcwd()


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
		remove_columns=remove_cols)
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
		remove_columns=remove_cols)
	return dataset


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
	"indic-glue": {
		"is_custom": True,
		"path": base_path + "/datasets/indic-glue",
		"split_names": ["train", "test"],
		"configuration": "wiki-ner.hi",
		"cols_to_concatenate": ["tokens"],
		"cols_to_remove": ["tokens", "ner_tags", "additional_info"],
		"preprocess_fn": preprocess_indic_glue_wiki_ner
	}
}


def load_and_concatenate(datasets_list):
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
	# print(processed_datasets)
	concatenated_dataset = concatenate_datasets(processed_datasets)
	return concatenated_dataset

datasets_list = [
	"hindi-text-short-summarization-corpus",
	"indic-glue"
]


dataset = load_and_concatenate(datasets_list)
shuffle_dataset = dataset.shuffle(seed=42)
print("Total rows:", len(shuffle_dataset))
print("Sample: ", shuffle_dataset[42])