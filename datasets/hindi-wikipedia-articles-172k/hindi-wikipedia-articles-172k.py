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
"""disisbig/hindi-wikipedia-articles-172k"""
import ast
import csv
import os
import datasets
# Some readme files on modelhub are large in size
csv.field_size_limit(100000000)
_CITATION = """\
"""
_DESCRIPTION = """\
disisbig/hindi-wikipedia-articles-172k
"""
_HOMEPAGE = "https://www.kaggle.com/disisbig/hindi-wikipedia-articles-172k"
_LICENSE = ""

_TRAIN_URL = "train/train"
_TEST_URL = "valid/valid"

class HindiWikipediaArticles172k(datasets.GeneratorBasedBuilder):
    """disisbig/hindi-wikipedia-articles-172k"""
    VERSION = datasets.Version("1.0.0")
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
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
        for id_, filename in os.listdir(filepath):
            if ".txt" not in filename:
                continue

            with open(filename, encoding="utf-8") as f:
                yield id_, {
                    "text": f.readlines(),
                }
