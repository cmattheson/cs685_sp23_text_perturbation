import json
import os

import datasets
from datasets.tasks import TextClassification

_CITATION = None


_DESCRIPTION = """
 Arxiv Classification Dataset: a classification of Arxiv Papers (11 classes).
 It contains 11 slightly unbalanced classes, 33k Arxiv Papers divided into 3 splits: train (23k), val (5k) and test (5k).
 Copied from "Long Document Classification From Local Word Glimpses via Recurrent Attention Learning" by JUN HE LIQUN WANG LIU LIU, JIAO FENG AND HAO WU
 See: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8675939
 See: https://github.com/LiqunW/Long-document-dataset
"""

_LABELS = [
    "math.AC",
    "cs.CV", 
    "cs.AI", 
    "cs.SY", 
    "math.GR", 
    "cs.CE", 
    "cs.PL", 
    "cs.IT", 
    "cs.DS", 
    "cs.NE", 
    "math.ST"
    ]


class ArxivClassificationConfig(datasets.BuilderConfig):
    """BuilderConfig for ArxivClassification."""

    def __init__(self, **kwargs):
        """BuilderConfig for ArxivClassification.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ArxivClassificationConfig, self).__init__(**kwargs)


class ArxivClassificationDataset(datasets.GeneratorBasedBuilder):
    """ArxivClassification Dataset: classification of Arxiv Papers (11 classes)."""
    
    _DOWNLOAD_URL = "https://huggingface.co/datasets/ccdv/arxiv-classification/resolve/main/"
    _TRAIN_FILE = "train_data.txt"
    _VAL_FILE = "val_data.txt"
    _TEST_FILE = "test_data.txt"
    _LABELS_DICT = {label: i for i, label in enumerate(_LABELS)}

    BUILDER_CONFIGS = [
        ArxivClassificationConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Arxiv Classification Dataset: A classification task of Arxiv Papers (11 classes)",
        ),
        ArxivClassificationConfig(
            name="no_ref",
            version=datasets.Version("1.0.0"),
            description="Arxiv Classification Dataset: A classification task of Arxiv Papers (11 classes)",
        ),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(names=_LABELS),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            task_templates=[TextClassification(
                text_column="text", label_column="label")],
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(self._TRAIN_FILE)
        val_path = dl_manager.download_and_extract(self._VAL_FILE)
        test_path = dl_manager.download_and_extract(self._TEST_FILE)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepath": val_path}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}
            ),
        ]
    
    def _generate_examples(self, filepath):
        """Generate ArxivClassification examples."""
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                label = self._LABELS_DICT[data["label"]]
                text = data["text"]

                if self.config.name == "no_ref":
                    for _ in _LABELS:
                        text = text.replace(_, "")
                        
                yield id_, {"text": text, "label": label}
