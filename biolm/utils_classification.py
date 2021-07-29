# coding=utf-8
# Copyright 2020- The Google AI Language Team Authors and The HuggingFace Inc. team and Facebook Inc.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""processors and helpers for classification"""

import logging
import os

from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.metrics import pearson_and_spearman, simple_accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support

import numpy as np
import json

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        elif output_mode == 'multilabel_classification':
            label = [label_map[l] for l in example.label]
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if output_mode == 'multilabel_classification':
                logger.info("label: %s (id = %s)" % (example.label, str(label)))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class HOCProcessor(DataProcessor):
    """Processor for the HOC data set"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [
            'activating invasion and metastasis',
            'avoiding immune destruction',
            'cellular energetics',
            'enabling replicative immortality',
            'evading growth suppressors',
            'genomic instability and mutation',
            'inducing angiogenesis',
            'resisting cell death',
            'sustaining proliferative signaling',
            'tumor promoting inflammation'
        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = "%s-%s" % (set_type, line[0])
                text_a = line[1]
                label = line[2].split(',') if line[2] != '' else []
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MedNLIProcessor(DataProcessor):
    """Processor for the HOC data set"""

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def _read_jsonl(self, fi):
        dps = []
        for line in open(fi):
            dps.append(json.loads(line))
        return dps

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "mli_train_v1.jsonl")), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "mli_dev_v1.jsonl")), 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, "mli_test_v1.jsonl")), 'test')

    def get_labels(self):
        """See base class."""
        return ["entailment", 'neutral', "contradiction"]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for item in items:
            guid = set_type + '-' + item['pairID']
            text_a = item['sentence1']
            text_b =item['sentence2']
            label = item['gold_label']
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ChemProtProcessor(DataProcessor):
    """Processor for the HOC data set"""

    chem_pattern = '@CHEMICAL$'
    gene_pattern = '@GENE$'
    chem_gene_pattern = "@CHEMICAL-GENE$"

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', 'false']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = "%s-%s-%s" % (str(i), set_type, line[0])
                if True:#line[2] != 'false':
                    text_a = line[1]
                    text_a = text_a.replace('@CHEMICAL$', self.chem_pattern).replace(
                        '@GENE$', self.gene_pattern).replace(
                        '@CHEM-GENE$', self.chem_gene_pattern)
                    label = line[2]
                    assert label in self.get_labels()
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class GADProcessor(DataProcessor):
    """Processor for the HOC data set"""

    disease_pattern = '@DISEASE$'
    gene_pattern = '@GENE$'
    fold = 1

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, str(self.fold), f"train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, str(self.fold), f"test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir,  str(self.fold), f"test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) == 2:
                line = [i] + line
            if line[0] == 'index':
                continue

            guid = "%s-%s-%s" % (str(i), set_type, line[0])
            text_a = line[1]
            text_a = text_a.replace('@DISEASE$', self.disease_pattern).replace(
                '@GENE$', self.gene_pattern)
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class EUADRProcessor(DataProcessor):
    """Processor for the HOC data set"""

    disease_pattern = '@DISEASE$'
    gene_pattern = '@GENE$'
    fold = 1

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, str(self.fold), f"train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir,  str(self.fold), f"test.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir,  str(self.fold), f"test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if len(line) == 2:
                line = [i] + line
            if line[0] == 'index':
                continue

            guid = "%s-%s-%s" % (str(i), set_type, line[0])
            text_a = line[1]
            text_a = text_a.replace('@DISEASE$', self.disease_pattern).replace(
                '@GENE$', self.gene_pattern)
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class DDIProcessor(DataProcessor):
    """Processor for the HOC data set"""
    drug_pattern = '@DRUG$'
    drug_drug_pattern = '@DRUG-DRUG$'

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['DDI-advise', 'DDI-effect', 'DDI-int', 'DDI-mechanism', 'DDI-false']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = "%s-%s-%s" % (str(i), set_type, line[0])
                # if line[2] == 'DDI-false':
                #     continue
                text_a = line[1]
                text_a = text_a.replace('@DRUG$', self.drug_pattern).replace(
                    '@DRUG-DRUG$', self.drug_drug_pattern)
                label = line[2]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class I2B22010Processor(DataProcessor):
    """Processor for the HOC data set"""
    problem_pattern = '@PROBLEM$'
    treatment_pattern = '@TREATMENT$'
    test_pattern = '@TEST$'
    problem_problem_pattern = "@PROBLEM-PROBLEM$"
    test_problem_pattern = "@TEST-PROBLEM$"
    test_test_pattern = "@TEST-TEST$"
    treatment_test_pattern = '@TREATMENT-TEST$'
    treatment_treatment_pattern = '@TREATMENT-TREATMENT$'

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train_new.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_new.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ['PIP', 'TeCP', 'TeRP', 'TrAP', 'TrCP', 'TrIP', 'TrNAP', 'TrWP','false']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i != 0:
                guid = "%s-%s-%s" % (str(i), set_type, line[0])
                text_a = line[1]
                text_a = text_a.replace(
                    '@problem$', self.problem_pattern).replace(
                    '@treatment$', self.treatment_pattern).replace(
                    '@test$', self.test_pattern).replace(
                    "@problem-problem$", self.problem_problem_pattern).replace(
                    "@test-problem$", self.test_problem_pattern).replace(
                    "@test-test$", self.test_test_pattern).replace(
                    '@treatment-test$', self.treatment_test_pattern).replace(
                    '@treatment-treatment$', self.treatment_treatment_pattern
                )
                label = line[2]
                assert label in self.get_labels()
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


tasks_num_labels = {
    "hoc": 10,
    "mednli": 3,
    "chemprot": 6,
    "gad1": 2,
    "gad2": 2,
    "gad3": 2,
    "gad4": 2,
    "gad5": 2,
    "gad6": 2,
    "gad7": 2,
    "gad8": 2,
    "gad9": 2,
    "gad10": 2,
    "euadr1": 2,
    "euadr2": 2,
    "euadr3": 2,
    "euadr4": 2,
    "euadr5": 2,
    "euadr6": 2,
    "euadr7": 2,
    "euadr8": 2,
    "euadr9": 2,
    "euadr10": 2,
    'ddi': 5,
    "i2b22010re": 9,
}


processors = {
    "hoc": HOCProcessor,
    "mednli": MedNLIProcessor,
    "chemprot": ChemProtProcessor,
    "gad1": GADProcessor,
    "gad2": GADProcessor,
    "gad3": GADProcessor,
    "gad4": GADProcessor,
    "gad5": GADProcessor,
    "gad6": GADProcessor,
    "gad7": GADProcessor,
    "gad8": GADProcessor,
    "gad9": GADProcessor,
    "gad10": GADProcessor,
    "euadr1": EUADRProcessor,
    "euadr2": EUADRProcessor,
    "euadr3": EUADRProcessor,
    "euadr4": EUADRProcessor,
    "euadr5": EUADRProcessor,
    "euadr6": EUADRProcessor,
    "euadr7": EUADRProcessor,
    "euadr8": EUADRProcessor,
    "euadr9": EUADRProcessor,
    "euadr10": EUADRProcessor,
    "ddi": DDIProcessor,
    "i2b22010re": I2B22010Processor,
}


output_modes = {
    "hoc": "multilabel_classification",
    "mednli": "classification",
    "chemprot": "classification",
    "gad1": "classification",
    "gad2": "classification",
    "gad3": "classification",
    "gad4": "classification",
    "gad5": "classification",
    "gad6": "classification",
    "gad7": "classification",
    "gad8": "classification",
    "gad9": "classification",
    "gad10": "classification",
    "euadr1": "classification",
    "euadr2": "classification",
    "euadr3": "classification",
    "euadr4": "classification",
    "euadr5": "classification",
    "euadr6": "classification",
    "euadr7": "classification",
    "euadr8": "classification",
    "euadr9": "classification",
    "euadr10": "classification",
    "ddi": "classification",
    "i2b22010re": "classification",

}

stopping_metrics = {
    "hoc": "f",
    "mednli": "acc",
    "chemprot": "micro_f1",
    "gad1": "f1",
    "gad2": "f1",
    "gad3": "f1",
    "gad4": "f1",
    "gad5": "f1",
    "gad6": "f1",
    "gad7": "f1",
    "gad8": "f1",
    "gad9": "f1",
    "gad10": "f1",
    "euadr1": "f1",
    "euadr2": "f1",
    "euadr3": "f1",
    "euadr4": "f1",
    "euadr5": "f1",
    "euadr6": "f1",
    "euadr7": "f1",
    "euadr8": "f1",
    "euadr9": "f1",
    "euadr10": "f1",
    "ddi": "micro_f1",
    "i2b22010re": "micro_f1",

}


def multiclass_acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    macro_precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    macro_recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    macro_weighted_recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        'micro_f1': micro_f1,
        "macro_f1": macro_f1,
        "macro_weighted_f1": macro_weighted_f1,
        "macro_precision": macro_precision,
        "macro_weighted_precision": macro_weighted_precision,
        "macro_recall": macro_recall,
        "macro_weighted_recall": macro_weighted_recall,
    }


def acc_and_micro_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "micro_f1": micro_f1,
    }


def acc_p_r_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, )
    recall = recall_score(y_true=labels, y_pred=preds, )
    precision = precision_score(y_true=labels, y_pred=preds, )

    return {
        "acc": acc,
        "f1": f1,
        'precision': precision,
        'recall': recall
    }


def hoc_get_p_r_f_arrary(preds, labels, examples):
    """adapted from BLUE benchmark: https://github.com/ncbi-nlp/BLUE_Benchmark/blob/b6216f2cb9bba209ee7028fc874123d8fd5a810c/blue/eval_hoc.py """
    threshold = 0.5
    cat = 10

    test_predict_label = {}
    test_true_label = {}
    for pred, label, example in zip(preds, labels, examples):
        doc_id = example.guid.split('-')[1].split('_')[0]
        snum = int(example.guid.split('-')[1].split('_s')[1])

        ttl = test_true_label.get(doc_id, [0 for _ in range(10)])
        tpl = test_predict_label.get(doc_id, [0 for _ in range(10)])
        for ind in range(10):
            if pred[ind] > threshold:
                tpl[ind] = 1
            if label[ind] == 1:
                ttl[ind] = 1
        test_true_label[doc_id] = ttl
        test_predict_label[doc_id] = tpl

    doc_ids = list(test_true_label.keys())

    acc_list = []
    prc_list = []
    rec_list = []
    f_score_list = []
    for doc_id in doc_ids:
        label_pred_set = set()
        label_gold_set = set()

        for j in range(cat):
            if test_predict_label[doc_id][j] == 1:
                label_pred_set.add(j)
            if test_true_label[doc_id][j] == 1:
                label_gold_set.add(j)

        uni_set = label_gold_set.union(label_pred_set)
        intersec_set = label_gold_set.intersection(label_pred_set)

        tt = len(intersec_set)
        if len(label_pred_set) == 0:
            prc = 0
        else:
            prc = tt / len(label_pred_set)

        acc = tt / len(uni_set)

        rec = tt / len(label_gold_set)

        if prc == 0 and rec == 0:
            f_score = 0
        else:
            f_score = 2 * prc * rec / (prc + rec)

        acc_list.append(acc)
        prc_list.append(prc)
        rec_list.append(rec)
        f_score_list.append(f_score)

    mean_prc = np.mean(prc_list)
    mean_rec = np.mean(rec_list)

    def divide(x, y):
        return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float), where=y != 0)

    f_score = divide(2 * mean_prc * mean_rec, (mean_prc + mean_rec))
    return {'p': mean_prc, 'r': mean_rec, 'f': f_score, 'acc': np.mean(acc_list)}


def chemprot_eval(preds, labels):
    p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[0, 1, 2, 3, 4], average="micro")
    return {
        "micro_p": p,
        'micro_f1': f,
        "micro_r": r,
    }


def ddi_eval(preds, labels):
    p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[0, 1, 2, 3], average="micro")
    return {
        "micro_p": p,
        'micro_f1': f,
        "micro_r": r,
    }


def i2b22010re_eval(preds, labels):
    p,r,f,s = precision_recall_fscore_support(y_pred=preds, y_true=labels, labels=[0, 1, 2, 3, 4, 5, 6, 7], average="micro")
    return {
        "micro_p": p,
        'micro_f1': f,
        "micro_r": r,
    }


def compute_metrics(task_name, preds, labels, examples):
    assert len(preds) == len(labels) == len(examples)
    if task_name == "medsts":
        return pearson_and_spearman(preds, labels)
    elif task_name == "biosses":
        return pearson_and_spearman(preds, labels)
    elif task_name == "hoc":
        return hoc_get_p_r_f_arrary(preds, labels, examples)
    elif task_name == "mednli":
        return multiclass_acc_and_f1(preds, labels)
    elif task_name =='chemprot':
        return chemprot_eval(preds, labels)
    elif task_name == 'gad1':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad2':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad3':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad4':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad5':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad6':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad7':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad8':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad9':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'gad10':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr1':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr2':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr3':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr4':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr5':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr6':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr7':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr8':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr9':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'euadr10':
        return acc_p_r_and_f1(preds, labels)
    elif task_name == 'ddi':
        return ddi_eval(preds, labels)
    elif task_name == "i2b22010re":
        return i2b22010re_eval(preds, labels)
    else:
        raise KeyError(task_name)
