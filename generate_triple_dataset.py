# -*- coding:utf-8 -*-
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm, trange
import argparse
import numpy as np
from Data_Processors import *


class Config:
    def __init__(self):
        self.language = "eng"
        if self.language == "eng":
            # eng
            self.bert_model = "/home/ssm/project/wch/pretrained_models/bert_base_uncased_eng/pytorch_model.bin"
            self.bert_model_config = "/home/ssm/project/wch/pretrained_models/bert_base_uncased_eng/config.json"
            self.bert_model_voc = "/home/ssm/project/wch/pretrained_models/bert_base_uncased_eng/vocab.txt"
        elif self.language == "chn":
            # # chn
            self.bert_model = "/home/ssm/project/wch/Bert4Classification/bert/bert_base_chinese_pytorch.bin"
            self.bert_model_config = "/home/ssm/project/wch/Bert4Classification/bert/config.json"
            self.bert_model_voc = "/home/ssm/project/wch/Bert4Classification/bert/vocab.txt"

        self.data_dir = "data/ISEAR"
        self.csk_dir = "data/csk"
        self.do_lower_case = True


config = Config()
csk_triples, csk_entities, raw_vocab, kb_dict = [], [], [], []
tokenizer = BertTokenizer.from_pretrained(config.bert_model_voc, do_lower_case=config.do_lower_case)


def prepare_data(
    path,
):
    global csk_triples, csk_entities, raw_vocab, kb_dict
    with open("%s/resource.txt" % path) as f:
        d = json.loads(f.readline())
    csk_triples = d["csk_triples"]
    csk_entities = d["csk_entities"]
    raw_vocab = d["vocab_dict"]
    kb_dict = d["dict_csk"]
    return csk_triples, csk_entities, raw_vocab, kb_dict


def get_dataset_examples(
    file_path,
):
    f = open(file_path, "r", encoding="utf-8")
    data = []
    for line in f.readlines():
        line = line.replace("\n", "").split("\t")
        text_a = str(line[1])
        label = str(line[0])
        tokens_a = tokenize(text_a)
        # print(tokens_a)
        data.append(tokens_a)
    return data


def save_file(filepath, lst):
    with open(filepath, "w", encoding="utf-8") as f:
        for i in range(len(lst)):
            f.write(lst[i])
            if i != len(lst) - 1:
                f.write("\n")


def tokenize(txt):
    tokens_t = tokenizer.tokenize(txt)
    tokens = []
    tokens.append("[CLS]")
    for token in tokens_t:
        tokens.append(token)
    tokens.append("[SEP]")
    return tokens


def get_triple_json(tokens):
    """
    Args:
        tokens : [List] tokenize后的句子的list ['CLS', 'I', 'am', 'happy', 'SEP']
    """
    d = {}
    sent_triples, all_entities, all_triples = [], [], []
    idx = 0
    for token in tokens:
        entities, triples = [], []
        if token in kb_dict:
            idx += 1
            sent_triples.append(idx)
            for triple in kb_dict[token]:
                triples.append(csk_triples.index(triple))
                tgt_token = triple.split(", ")[-1] if triple.split(", ")[0] == token else triple.split(", ")[0]
                entities.append(csk_entities.index(tgt_token))
            all_triples.append(triples)
            all_entities.append(entities)
        else:
            sent_triples.append(0)
    d["sent_triples"] = sent_triples
    d["all_entities"] = all_entities
    d["all_triples"] = all_triples
    d["sentence"] = tokens
    return d


def generate_dataset(file_in_path, file_out_path):
    all_data = get_dataset_examples(file_in_path)
    out_data_lst = []
    for data in tqdm(all_data):
        out_data_lst.append(json.dumps(get_triple_json(data)))
    save_file(file_out_path, out_data_lst)


def main():
    prepare_data(config.csk_dir)
    generate_dataset(config.data_dir + "/data_train.txt", config.data_dir + "/trainset.txt")
    generate_dataset(config.data_dir + "/data_val.txt", config.data_dir + "/validset.txt")
    generate_dataset(config.data_dir + "/data_test.txt", config.data_dir + "/testset.txt")


if __name__ == "__main__":
    main()
