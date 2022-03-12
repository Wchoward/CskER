import sys
import os
import copy
import json
import logging
import collections
from torch.nn.functional import triplet_margin_loss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

logger = logging.getLogger(__name__)


class SentimentInputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, triples, sent_triples, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.triples = triples
        self.sent_triples = sent_triples
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentimentInputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, triples, sent_triples, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.triples = triples
        self.sent_triples = sent_triples
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class SentimentDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


def sentiment_convert_examples_to_features(
    examples,
    csk_entity_relation_vocab,
    tokenizer,
    max_length=256,
    label_list=None,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    logging.info("***** converting to features *****")
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    def _truncate(content, max_length):
        while len(content) > max_length:
            content = list(content)
            content.pop(len(content) // 2)
        return "".join(content)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            max_length=max_length,
            truncation=True,
            # truncate_first_sequence=True  # We're truncating the first sequence in priority if True
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        triples = [
            [[csk_entity_relation_vocab.get(entity, 0) for entity in tri] for tri in triples]
            for triples in example.triples
        ]  # 如果不在entity vocab里 就设置为0，也就是_NONE
        sent_triples = example.sent_triples

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        assert len(sent_triples) == max_length, "Error with input length {} vs {}".format(len(sent_triples), max_length)

        label = int(label_map[example.label])

        features.append(SentimentInputFeatures(input_ids, triples, sent_triples, attention_mask, token_type_ids, label))
    return features


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, args, csk, data_dir):
        self.args = args
        self.data_dir = data_dir
        self.NAF = csk.NAF
        self.csk_triples = csk.csk_triples
        self.csk_entities = csk.csk_entities

    @classmethod
    def _read_txt(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip(" \n"))
            return lines

    @classmethod
    def _read_csk_file(cls, input_csk_file):
        with open(input_csk_file, "r", encoding="utf-8") as f:
            lines = []
            for idx, line in enumerate(f):
                if idx % 10000 == 0:
                    print("read file line %d" % idx)
                lines.append(json.loads(line))
            return lines

    def _padding_triple(self, triple, num, l):
        newtriple = []
        triple = [[self.NAF]] + triple
        for tri in triple:
            newtriple.append(tri + [["_PAD_H", "_PAD_R", "_PAD_T"]] * (l - len(tri)))
        pad_triple = [["_PAD_H", "_PAD_R", "_PAD_T"]] * l
        return newtriple + [pad_triple] * (num - len(newtriple))


class IsearProcessor(DataProcessor):
    """Processor for Isear dataset"""

    def get_train_examples(self):
        """See base class."""
        logger.info("*" * 10 + "train dataset" + "*" * 10)
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "data_train.txt")),
            self._read_csk_file(os.path.join(self.data_dir, "trainset.txt")),
            "train",
        )

    def get_dev_examples(self):
        """See base class."""
        logger.info("*" * 10 + "val dataset" + "*" * 10)
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "data_val.txt")),
            self._read_csk_file(os.path.join(self.data_dir, "validset.txt")),
            "val",
        )

    def get_test_examples(self):
        """See base class."""
        logger.info("*" * 10 + "test testset" + "*" * 10)
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "data_test.txt")),
            self._read_csk_file(os.path.join(self.data_dir, "testset.txt")),
            "test",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6"]

    def _create_examples(self, lines, csk_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        triple_num = max([len(item["all_triples"]) for item in csk_lines]) + 1
        triple_len = max([len(tri) for item in csk_lines for tri in item["all_triples"]])
        for (i, (line, csk_line)) in enumerate(zip(lines, csk_lines)):
            line = line.replace("\n", "").split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = str(line[1])
            label = str(line[0])
            triples = self._padding_triple(
                [[self.csk_triples[x].split(", ") for x in triple] for triple in csk_line["all_triples"]],
                triple_num,
                triple_len,
            )
            sent_triples = [[x] for x in csk_line["sent_triples"]] + [[0]] * (
                self.args.max_seq_length - len(csk_line["sent_triples"])
            )
            examples.append(
                SentimentInputExample(
                    guid=guid, text_a=text_a, triples=triples, sent_triples=sent_triples, text_b=None, label=label
                )
            )
        return examples


class TecProcessor(DataProcessor):
    """Processor for Isear dataset"""

    def get_train_examples(self):
        """See base class."""
        logger.info("*" * 10 + "train dataset" + "*" * 10)
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "data_train.txt")),
            self._read_csk_file(os.path.join(self.data_dir, "trainset.txt")),
            "train",
        )

    def get_dev_examples(self):
        """See base class."""
        logger.info("*" * 10 + "val dataset" + "*" * 10)
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "data_val.txt")),
            self._read_csk_file(os.path.join(self.data_dir, "validset.txt")),
            "val",
        )

    def get_test_examples(self):
        """See base class."""
        logger.info("*" * 10 + "test testset" + "*" * 10)
        return self._create_examples(
            self._read_txt(os.path.join(self.data_dir, "data_test.txt")),
            self._read_csk_file(os.path.join(self.data_dir, "testset.txt")),
            "test",
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def _create_examples(self, lines, csk_lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        triple_num = max([len(item["all_triples"]) for item in csk_lines]) + 1
        triple_len = max([len(tri) for item in csk_lines for tri in item["all_triples"]])
        for (i, (line, csk_line)) in enumerate(zip(lines, csk_lines)):
            line = line.replace("\n", "").split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = str(line[1])
            label = str(line[0])
            triples = self._padding_triple(
                [[self.csk_triples[x].split(", ") for x in triple] for triple in csk_line["all_triples"]],
                triple_num,
                triple_len,
            )
            sent_triples = [[x] for x in csk_line["sent_triples"]] + [[0]] * (
                self.args.max_seq_length - len(csk_line["sent_triples"])
            )
            examples.append(
                SentimentInputExample(
                    guid=guid, text_a=text_a, triples=triples, sent_triples=sent_triples, text_b=None, label=label
                )
            )
        return examples


class IeceProcessor(DataProcessor):
    """Processor for Isear dataset"""

    def get_train_examples(self):
        """See base class."""
        logger.info("*" * 10 + "train dataset" + "*" * 10)
        return self._create_examples(self._read_txt(os.path.join(self.data_dir, "data_train.txt")), "train")

    def get_dev_examples(self):
        """See base class."""
        logger.info("*" * 10 + "val dataset" + "*" * 10)
        return self._create_examples(self._read_txt(os.path.join(self.data_dir, "data_val.txt")), "val")

    def get_test_examples(self):
        """See base class."""
        logger.info("*" * 10 + "test testset" + "*" * 10)
        return self._create_examples(self._read_txt(os.path.join(self.data_dir, "data_test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5", "6"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            line = line.replace("\n", "").split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = str(line[1])
            label = str(line[0])
            examples.append(SentimentInputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Smp2020Processor(DataProcessor):
    """Processor for Isear dataset"""

    def get_train_examples(self):
        """See base class."""
        logger.info("*" * 10 + "train dataset" + "*" * 10)
        return self._create_examples(self._read_txt(os.path.join(self.data_dir, "data_train.txt")), "train")

    def get_dev_examples(self):
        """See base class."""
        logger.info("*" * 10 + "val dataset" + "*" * 10)
        return self._create_examples(self._read_txt(os.path.join(self.data_dir, "data_val.txt")), "val")

    def get_test_examples(self):
        """See base class."""
        logger.info("*" * 10 + "test testset" + "*" * 10)
        return self._create_examples(self._read_txt(os.path.join(self.data_dir, "data_test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4", "5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            line = line.replace("\n", "").split("\t")
            guid = "%s-%s" % (set_type, i)
            text_a = str(line[1])
            label = str(line[0])
            examples.append(SentimentInputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


sentiment_processors = {
    "ISEAR": IsearProcessor,
    "TEC": TecProcessor,
    "IECE": IeceProcessor,
    "SMP2020": Smp2020Processor,
}

dataset_num_labels = {
    "ISEAR": 7,
    "TEC": 6,
    "IECE": 6,
    "SMP2020": 7,
}
