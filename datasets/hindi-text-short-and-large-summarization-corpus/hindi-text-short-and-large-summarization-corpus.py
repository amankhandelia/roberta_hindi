# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""disisbig/hindi-text-short-and-large-summarization-corpus"""

import ast
import csv
import datasets
# Some readme files on modelhub are large in size
csv.field_size_limit(100000000)
_CITATION = """\
"""
_DESCRIPTION = """\
disisbig/hindi-text-short-and-large-summarization-corpus
"""
_HOMEPAGE = "https://www.kaggle.com/disisbig/hindi-text-short-and-large-summarization-corpus"
_LICENSE = ""

_TRAIN_URL = "train.csv"
_TEST_URL = "test.csv"

class HindiTextShortSummarizationCorpus(datasets.GeneratorBasedBuilder):
    """disisbig/hindi-text-short-and-large-summarization-corpus"""
    VERSION = datasets.Version("1.0.0")
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "headline": datasets.Value("string"),
                    "summary" : datasets.Value("string"),
                    "article" : datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        train_path = dl_manager.download_and_extract(_TRAIN_URL)
        test_path = dl_manager.download_and_extract(_TEST_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]
    
    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f)
            for id_, row in enumerate(reader):
                
                if id_ == 0:
                    continue

                if len(row)==3:

                    yield id_, {
                        "headline": row[0],
                        "summary": row[1],
                        "article": row[2],
                    }