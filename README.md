# RoBERTa Hindi

HuggingFace Flax-Community Project Forum link  : https://discuss.huggingface.co/t/pretrain-roberta-from-scratch-in-hindi
Pretrain RoBERTa for Hindi Language and compare the performance on various down stream tasks with existing models

### Contents:
#### Datasets:
In order to create a stable pipeline for building a large corpus of Hindi-text datasets, we brought different datasets available on HuggingFaces's datasethub and kaggle under a single wrapper `datasets` library. Dataset loading scripts for all the hindi datasets we could find are available in `datasets/` directory. 

`concatenate_datasets_and_clean.py` merges all datasets(with cleaning hooks) together into a single large dataset which could be fed seamlessly into Huggingface transformers MLM training loop. 

#### Indic-glue fix:
Just a minor fix as [WikiNER](https://github.com/huggingface/datasets/tree/master/datasets/indic_glue) hindi downstream task on HuggingFace datasets library was buggy. Need to load `indic_glue_dataset_fixed/indic_glue.py` instead to perform downstream evaluation.

#### Benchmarks: 
Using [IndicGlue](https://huggingface.co/metrics/indic_glue) benchmarks, the `benchmarks/` directory contains helpers which could be used to fine-tune & perform downstream evaluation on the following tasks (for multiple models available on modelhub as reference, and log results to wandb):
- BBCA news classification
- IITP product reviews
- IITP movie reviews
- WikiNER (Named entity recognition)

Note: We are still in the process on cleaning up some colab notebooks and will commit an end2end working script soon.

## Demo:

[Check out the Demo here!](https://huggingface.co/spaces/flax-community/roberta-hindi)

![roberta_hindi_demo](./images/roberta_hindi_demo.png)

## Example code to use our Model for Mask Filling Task:
```python
from transformers import AutoTokenizer,AutoModelForMaskedLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, from_flax=True)

nlp = pipeline('fill-mask', model=model, tokenizer=tokenizer)
masked_text = 'हम आपके <mask> यात्रा की कामना करते हैं'
result = nlp(masked_text)
result

"""
Output:
हम आपके ी यात्रा की कामना करते हैं
"""
```

## Our Results

## Contributors

## Future Work
Create the best Hindi Language Model to use in real world applications

