# RoBERTa Hindi

HuggingFace Flax-Community Project Forum link: https://discuss.huggingface.co/t/pretrain-roberta-from-scratch-in-hindi
Pretrain RoBERTa for Hindi Language and compare the performance on various down stream tasks with existing models

### Contents
#### Datasets
In order to create a stable pipeline for building a large corpus of Hindi-text datasets, we brought different datasets available on HuggingFaces's datasethub and kaggle under a single wrapper `datasets` library. Dataset loading scripts for all the hindi datasets we could find are available in `datasets/` directory. 

`concatenate_datasets_and_clean.py` merges all datasets(with cleaning hooks) together into a single large dataset which could be fed seamlessly into Huggingface transformers MLM training loop. 

#### Indic-glue Fix
Just a minor fix as [WikiNER](https://github.com/huggingface/datasets/tree/master/datasets/indic_glue) hindi downstream task on HuggingFace datasets library was buggy. Need to load `indic_glue_dataset_fixed/indic_glue.py` instead to perform downstream evaluation.

#### Benchmarks:
Using [IndicGlue](https://huggingface.co/metrics/indic_glue) benchmarks, the `benchmarks/` directory contains helpers which could be used to fine-tune & perform downstream evaluation on the following tasks (for multiple models available on modelhub as reference, and log results to wandb):
- BBCA news classification
- IITP product reviews
- IITP movie reviews
- WikiNER (Named entity recognition)

Note: We are still in the process on cleaning up some colab notebooks and will commit an end2end working script soon.

## Demo

[Check out the Demo here!](https://huggingface.co/spaces/flax-community/roberta-hindi)

![roberta_hindi_demo](./images/roberta_hindi_demo.png)

## Example code to use our Model for Mask Filling Task
```python
```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='flax-community/roberta-hindi')
>>> unmasker("हम आपके सुखद <mask> की कामना करते हैं")
[{'score': 0.3310680091381073,
  'sequence': 'हम आपके सुखद सफर की कामना करते हैं',
  'token': 1349,
  'token_str': ' सफर'},
 {'score': 0.15317578613758087,
  'sequence': 'हम आपके सुखद पल की कामना करते हैं',
  'token': 848,
  'token_str': ' पल'},
 {'score': 0.07826550304889679,
  'sequence': 'हम आपके सुखद समय की कामना करते हैं',
  'token': 453,
  'token_str': ' समय'},
 {'score': 0.06304813921451569,
  'sequence': 'हम आपके सुखद पहल की कामना करते हैं',
  'token': 404,
  'token_str': ' पहल'},
 {'score': 0.058322224766016006,
  'sequence': 'हम आपके सुखद अवसर की कामना करते हैं',
  'token': 857,
  'token_str': ' अवसर'}]
"""
```

## Training data

The RoBERTa Hindi model was pretrained on the reunion of the following datasets:
- [OSCAR](https://huggingface.co/datasets/oscar) is a huge multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the goclassy architecture.
- [mC4](https://huggingface.co/datasets/mc4) is a multilingual colossal, cleaned version of Common Crawl's web crawl corpus.
- [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/) is a natural language understanding benchmark.
- [Samanantar](https://indicnlp.ai4bharat.org/samanantar/) is a parallel corpora collection for Indic language.
- [Hindi Text Short and Large Summarization Corpus](https://www.kaggle.com/disisbig/hindi-text-short-and-large-summarization-corpus) is a collection of ~180k articles with their headlines and summary collected from Hindi News Websites.
- [Hindi Text Short Summarization Corpus](https://www.kaggle.com/disisbig/hindi-text-short-summarization-corpus) is a collection of ~330k articles with their headlines collected from Hindi News Websites.
- [Old Newspapers Hindi](https://www.kaggle.com/crazydiv/oldnewspapershindi) is a cleaned subset of HC Corpora newspapers.

## Evaluation Results

RoBERTa Hindi is evaluated on various downstream tasks. The results are summarized below.

| Task                    | Task Type            | IndicBERT | HindiBERTa | Indic Transformers Hindi BERT | RoBERTa Hindi Guj San | RoBERTa Hindi |
|-------------------------|----------------------|-----------|------------|-------------------------------|-----------------------|---------------|
| BBC News Classification | Genre Classification | **76.44**     | 66.86      | **77.6**                          | 64.9                  | 73.67         |
| WikiNER                 | Token Classification | -         | 90.68      | **95.09**                         | 89.61                 | **92.76**         |
| IITP Product Reviews    | Sentiment Analysis   | **78.01**     | 73.23      | **78.39**                         | 66.16                 | 75.53         |
| IITP Movie Reviews      | Sentiment Analysis   | 60.97     | 52.26      | **70.65**                         | 49.35                 | **61.29**         |

## Team Members
- Aman K ([amankhandelia](https://huggingface.co/amankhandelia))
- Haswanth Aekula ([hassiahk](https://huggingface.co/hassiahk))
- Kartik Godawat ([dk-crazydiv](https://huggingface.co/dk-crazydiv))
- Prateek Agrawal ([prateekagrawal](https://huggingface.co/prateekagrawal))
- Rahul Dev ([mlkorra](https://huggingface.co/mlkorra))

## Future Work
Create the best Hindi Language Model to use in real world applications.
