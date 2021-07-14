import pandas as padding
from functools import partial
from datasets.dataset_dict import DatasetDict

def remove_ineligible_sentences(path_to_ineligible_cases:str, dataset:DatasetDict, split_name:str='validation'):
    
    # read csv with marking for ineligible cases
    marked_ner_examples_csv = path_to_ineligible_cases
    marked_ner_examples = pd.read_csv(marked_ner_examples_csv)
    
    # flag for the examples which are ineligible
    is_ineligible = marked_ner_examples['exclude_label'] != 0.
    
    # flag to exclude those which has not been marked yet
    is_marked = ~(marked_ner_examples['exclude_label'].isna())
    
    # creating df of ineligible examples
    ineligible_ner_examples = marked_ner_examples.loc[is_ineligible & is_marked]
    
    # turning the string object to list
    ineligible_ner_examples['tokens'] = ineligible_ner_examples['tokens'].map(eval)
    
    def identify_eligible(record, ineligible_examples):
        # return True for the records that are to be kept in the validation set
        match_count = int((ineligible_examples['tokens'].map(lambda x: x == record['tokens'])).sum())
        return match_count == 0

    # filter out the ineligible cases from the split and assign to the split
    filter_ineligble = partial(identify_eligible, ineligible_examples=ineligible_ner_examples)
    dataset[split_name] = dataset[split_name].filter(filter_ineligble)
    return dataset