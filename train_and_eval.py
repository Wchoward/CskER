import json
import logging
import os
import time
import pdb
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, dataloader
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import collate_batch, set_seed
from tqdm import tqdm, trange


def train(args, train_dataset, eval_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate_batch
    )
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warm up and decay)
    no_decay = ["bias", "LayerNorm.weight", "transitions"]
    bert_params = ["bert.embeddings", "bert.encoder"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay)) and any(nr in n for nr in bert_params)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and any(nr in n for nr in bert_params)
            ],
            "weight_decay": 0.0,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay)) and (not any(nr in n for nr in bert_params))
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and (not any(nr in n for nr in bert_params))
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_rate * t_total, num_training_steps=t_total
    )

    logging.info("*" * 15 + "args" + "*" * 15)
    for k, v in args.__dict__.items():
        logging.info("  {:18s} = {}".format(str(k), str(v)))
    logging.info("*" * 35)
    logging.info("***** Running training *****")
    logging.info("  Device = %s", args.device)
    logging.info("  Model name = %s", str(args.__dict__))
    logging.info("  Learning rate = %s", str(args.learning_rate))
    logging.info("  Warmup rate = %s", str(args.warmup_rate))
    logging.info("  Weight Decay = %s", str(args.weight_decay))
    logging.info("  Num examples = %d", len(train_dataset))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num Epochs = %d", args.num_train_epochs)
    logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logging.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    optimizer.step()

    best_f_score = 0.0
    best_epoch = 0

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args.seed)  # Added here for reproductibility
    for epoch in train_iterator:
        logging.info("  Epoch [{}/{}]".format(epoch + 1, args.num_train_epochs))
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(args.device)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            # logging.info('*** loss = %f ***',loss)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            logging_loss += loss.item()
            tr_loss += loss.item()

            # ???gradient_accumulation_steps??????????????????????????????????????????/????????????batch???????????????????????????gradient_accumulation_steps?????????????????????
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # ???????????????
                model.zero_grad()
                global_step += 1

                # logging.info("EPOCH = [%d/%d] global_step = %d loss = %f", epoch+1, args.num_train_epochs, global_step,logging_loss)

                # if (global_step % 50 == 0 and global_step <= 1000) or( global_step % 100 == 0 and global_step <= 5000) \
                #  or (global_step % 200 == 0):
                if global_step % 30 == 0:
                    logging.info(
                        "EPOCH = [%d/%d] global_step = %d loss = %f",
                        epoch + 1,
                        args.num_train_epochs,
                        global_step,
                        logging_loss / 30,
                    )
                    logging_loss = 0.0
                    best_f_score, best_epoch = evaluate_and_save_model(
                        args, model, eval_dataset, epoch, global_step, best_f_score, best_epoch
                    )
        if epoch - best_epoch >= 4:
            logging.info("Long time no improvement, stop train, best epoch = %f", best_epoch + 1)
            break
    logging.info("train end, best epoch = {}, best f score = {}".format(best_epoch + 1, best_f_score))


def evaluate_and_save_model(args, model, eval_dataset, epoch, global_step, best_f_score, best_epoch):
    eval_loss, label_acc, label_f_score = evaluate(args, model, eval_dataset)
    # logging.info("Evaluating EPOCH = [%d/%d] global_step = %d eval_loss = %f label_acc = %f label_f_score = %f",
    #             epoch + 1, args.num_train_epochs,global_step,eval_loss, label_acc, label_f_score)
    if label_f_score > best_f_score:
        best_f_score = label_f_score
        best_epoch = epoch
        improve = "*"
        model.save_pretrained(args.output_dir)
        # torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
        # logging.info("save the best net %s , label f score = %f",
        #              os.path.join(args.output_dir, "best_bert.bin"), best_f_score)
    else:
        improve = ""
    msg = "  Iter: {0:>6},  Val Loss: {1:>5.2}, Val F1: {2:>6.2%}, Val Acc: {3:>6.2%}, {4}"
    logging.info(msg.format(global_step, eval_loss, label_f_score, label_acc, improve))
    return best_f_score, best_epoch


def evaluate(args, model, eval_dataset, is_test=False):
    eval_ouput_dirs = args.output_dir
    if not os.path.exists(eval_ouput_dirs):
        os.makedirs(eval_ouput_dirs)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_batch
    )
    # logging.info("***** Running evaluation *****")
    # logging.info("  Num examples = %d", len(eval_dataset))
    # logging.info("  Batch size = %d", args.eval_batch_size)

    total_loss = 0.0  # loss ?????????
    total_sample_num = 0  # ???????????????

    preds = None  # ??????????????????????????????
    out_label_ids = None  # ??????????????????????????????
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in tqdm(eval_dataloader):
        model.eval()
        with torch.no_grad():
            # pdb.set_trace()
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(args.device)

            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            # ????????????????????????batch????????????batch size?????????
            total_loss += loss * list(batch.values())[0].shape[0]  # loss * ????????????
            total_sample_num += list(batch.values())[0].shape[0]  # ??????????????????
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                # pdb.set_trace()
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    loss = total_loss / total_sample_num
    preds = np.argmax(preds, axis=1)
    # print(out_label_ids.shape, preds.shape)
    # print(out_label_ids.dtype, preds.dtype)
    label_f_score = f1_score(y_true=out_label_ids, y_pred=preds, average="macro")
    label_acc = accuracy_score(out_label_ids, preds)
    model.train()
    if is_test:
        report = classification_report(y_true=out_label_ids, y_pred=preds, digits=4)
        confusion = confusion_matrix(y_true=out_label_ids, y_pred=preds)
        return loss, label_acc, label_f_score, report, confusion
    return loss, label_acc, label_f_score


def test(args, model, test_dataset):
    # test
    model.eval()
    test_loss, test_acc, label_f_score, test_report, test_confusion = evaluate(
        args, model, eval_dataset=test_dataset, is_test=True
    )
    msg = "Test Loss: {0:>5.2},  Test F1: {1:>6.2%}, Test Acc: {2:>6.2%}"
    logging.info(msg.format(test_loss, label_f_score, test_acc))
    logging.info("Precision, Recall and F1-Score...")
    logging.info(test_report)
    logging.info("Confusion Matrix...")
    logging.info(test_confusion)
    pred_probs = _predict(args, model, test_dataset)
    return pred_probs


def _predict(args, model, predict_dataset):
    model.eval()
    # ??????????????????
    predict_sampler = SequentialSampler(predict_dataset)
    # ?????????Dataloader
    predict_dataloader = DataLoader(
        predict_dataset, sampler=predict_sampler, batch_size=args.eval_batch_size, collate_fn=collate_batch
    )

    preds = None  # ????????????
    # out_label_ids = None    #???????????????
    for batch in tqdm(predict_dataloader):
        model.eval()  # ????????????
        with torch.no_grad():  # ??????????????????
            # ?????????????????? ??????????????? token_type_ids???batch[2] ?????????????????????????????? ????????????0
            inputs = {}
            for k, v in batch.items():
                inputs[k] = v.to(args.device)
            outputs = model(**inputs)  # ??????????????????
            loss, logits = outputs[0], outputs[1]  # ????????????loss???logits
            logits = torch.softmax(logits, dim=-1)
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    return preds


def predict(args, model, predict_dataset, processor, label_list):

    preds = _predict(args, model, predict_dataset)
    preds = np.argmax(preds, axis=1)  # ?????????????????????????????????????????????argmax
    results = []
    for i, pred in enumerate(preds):
        results.append({"id": i + 1, "label": label_list[pred]})
    _, data_type = os.path.split(args.data_dir)
    output_file = data_type + "_result.txt"
    with open(os.path.join(args.output_dir, output_file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
