import argparse
import logging
import os
import json
import pickle
import random
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, set_seed
from utils import set_logger, set_seed
from transformers import BertConfig, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, AutoModel
from net.bert_base import BertForSequenceClassification
from net.bert_attention import BertForSequenceClassificationAttention
from net.bert_lstm import BertForSequenceClassificationLSTM
from net.bert_lstm_attention import BertForSequenceClassificationLSTMAttenion

from processor import sentiment_processors as processors
from processor import sentiment_convert_examples_to_features, SentimentDataset

from train_and_eval import train, test, predict, _predict


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert_base": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "bert_attention": (BertConfig, BertForSequenceClassificationAttention, BertTokenizer),
    "bert_lstm": (BertConfig, BertForSequenceClassificationLSTM, BertTokenizer),
    "bert_lstm_attention": (BertConfig, BertForSequenceClassificationLSTMAttenion, BertTokenizer),
}


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--log_dir", default="log", type=str, required=True, help="设置日志的输出目录")
    parser.add_argument(
        "--dataset",
        choices=["ISEAR", "TEC", "IECE", "SMP2020"],
        default="ISEAR",
        type=str,
        help="应用的数据集，ISEAR, TEC, IECE, SMP2020中4选1",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--pre_train_path",
        default=None,
        type=str,
        required=True,
        help="预训练模型所在的路径，包括 pytorch_model.bin, vocab.txt, bert_config.json",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")

    parser.add_argument("--max_seq_length", default=256, type=int, help="输入到bert的最大长度，通常不应该超过512")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")

    parser.add_argument("--num_train_epochs", default=20, type=int, help="epoch 数目")
    parser.add_argument("--train_batch_size", default=8, type=int, help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=512, type=int, help="验证集的batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大的梯度更新")
    parser.add_argument("--seed", type=int, default=233, help="random seed for initialization")
    # parser.add_argument("--warmup_steps", default=0, type=int,
    #                     help="让学习增加到1的步数，在warmup_steps后，再衰减到0")
    parser.add_argument(
        "--warmup_rate", default=0.00, type=float, help="让学习增加到1的步数，在warmup_steps后，再衰减到0，这里设置一个小数，在总训练步数*rate步时开始增加到1"
    )

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.dataset + "_" + args.model_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert os.path.exists(os.path.join("data", args.dataset))
    assert os.path.exists(args.pre_train_path)
    assert os.path.exists(args.output_dir)

    # 暂时不写多GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    log_dir = os.path.join(
        args.log_dir,
        args.dataset + "_" + args.model_name + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())) + ".log",
    )
    set_logger(log_dir)

    data_dir = os.path.join("data", args.dataset)

    processor = processors[args.dataset](args, data_dir)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_name = args.model_name.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_name]
    if args.do_train:
        logging.info("loading pretrained model... ...")
        config = config_class.from_pretrained(args.pre_train_path, num_labels=num_labels)
        tokenizer = tokenizer_class.from_pretrained(args.pre_train_path, do_lower_case=args.do_lower_case)
        config.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        model = model_class.from_pretrained(args.pre_train_path, config=config, args=args)
        model.to(args.device)
        logging.info("load pretrained model end... ...")
        logger.info("Training parameters %s", args)

    def convert_to_dataset(examples):
        features = sentiment_convert_examples_to_features(
            examples=examples, tokenizer=tokenizer, max_length=args.max_seq_length, label_list=label_list
        )
        return SentimentDataset(features)

    # Training
    if args.do_train:
        logging.info("loading dataset... ...")
        train_examples = processor.get_train_examples()
        train_dataset = convert_to_dataset(train_examples)
        dev_examples = processor.get_dev_examples()
        dev_dataset = convert_to_dataset(dev_examples)
        logging.info("dataset loaded...")

        train_dataset = np.array(train_dataset)
        dev_dataset = np.array(dev_dataset)

        logging.info("start training... ...")
        train(args, train_dataset, dev_dataset, model)
        logging.info("train end...")

    if args.do_eval:
        logging.info("loading trained model... ...")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        config = config_class.from_pretrained(args.output_dir, num_labels=num_labels)
        model = model_class.from_pretrained(args.output_dir, config=config, args=args)
        model.to(args.device)
        logging.info("load trained model end... ...")
        logger.info("Evaluation parameters %s", args)

    # Evaluation
    if args.do_eval:
        logging.info("loading dataset... ...")
        test_examples = processor.get_test_examples()
        test_dataset = convert_to_dataset(test_examples)
        logging.info("dataset loaded...")

        test_dataset = np.array(test_dataset)
        test_dataset = np.array(test_dataset)

        logging.info("start evaluating... ...")
        test_probs = test(args, model, test_dataset)
        logging.info("evaluate end...")


if __name__ == "__main__":
    main()
