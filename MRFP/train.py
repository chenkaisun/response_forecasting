# from transformers import BertTokenizer
# from rouge_score import rouge
# import datasets
# from transformers import BertTokenizerFast
# import os
# os.environ['WANDB_DISABLED'] = "true"
import gc
import time
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from datasets import load_metric
from torch.utils.data import DataLoader
from eval_final import Evaluate

from evaluate import *
# from torch.optim.lr_scheduler import _LRScheduler
import wandb
# from data import CustomCollator, CustomCollatorCLF  # , collate_wrapper
from data_collator import CustomCollator, CustomCollatorCLF, CustomCollatorPrimitive, CustomCollatorCLM
from collections import defaultdict
# import wandb
# from transformers import get_linear_schedule_with_warmup
from evaluate import evaluate_clf, get_scores_multilabel_clf
# from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
# from datasets import Dataset, load_dataset
# from data import pad_to_batch
from train_utils import get_scheduler, get_tensor_float
from train_utils import seed_worker
from transformers import PreTrainedTokenizerBase
from transformers import SchedulerType
# from train_utils import seed_worker
# from utils import load_file, dump_file, visualize_plot
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, TrainingArguments
from transformers.trainer import Trainer
from utils.utils import modify_dict_keys
from utils.data_utils import *
import time
from BARTScore.bart_score import BARTScorer
from constants import *


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        # loss_fct =  torch.nn.BCEWithLogitsLoss()
        loss_fct = torch.nn.MultiLabelSoftMarginLoss()
        # print("logits", get_tensor_info(logits))
        # print("self.model.num_labels", self.model.num_labels)
        loss = loss_fct(logits.view(-1, self.model.num_labels),
                        labels.float().view(-1, self.model.num_labels))
        return (loss, outputs) if return_outputs else loss


class MultiClassSingleLabelTrainer(Trainer):

    def add_attribute(self, class_weights, weighted_loss):
        self.class_weights = class_weights
        self.weighted_loss = weighted_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        # logits = logits.view(-1, self.model.num_labels)
        # loss_fct =  torch.nn.BCEWithLogitsLoss()
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights if self.weighted_loss else None)  # if self.weighted_loss else torch.nn.CrossEntropyLoss()

        # print("logits", get_tensor_info(logits))
        # print("self.model.num_labels", self.model.num_labels)
        loss = loss_fct(logits.view(-1, self.model.num_labels),
                        labels.view(-1))
        return (loss, outputs) if return_outputs else loss


@dataclass
class MetricsComputerCLF:
    tokenizer: PreTrainedTokenizerBase = None
    metric: Any = load_metric("sacrebleu")
    save_dev_location: Any = None
    save_test_location: Any = None
    input_file: Any = None
    input_data: Any = None
    print_input: Any = True
    eval_f: Any = Evaluate()
    model: Any = None
    device: Any = None
    args: Any = None

    def __call__(self, eval_preds):

        # preds, labels = eval_preds

        logits, labels = eval_preds
        labels = labels.tolist()
        # print("logits, labels ", logits, labels)
        # logits = (torch.sigmoid(get_tensor_float(logits)) > 0.5).int().tolist()
        # get softmax
        logits = torch.softmax(get_tensor_float(logits), dim=1).tolist()
        # get predictions
        preds = [np.argmax(item) for item in logits]
        max_probs = [item[preds[i]] for i, item in enumerate(logits)]

        # convert label into vec

        # labels = [id2label[label] for label in labels]

        data_file = None
        if self.args.is_labeling:
            data_file = load_file(f"{self.input_file}")
        for i, (a, b, c, d) in enumerate(zip(max_probs, preds, labels, self.input_data)):  # , d , self.input_nbs
            if self.args.is_labeling:
                data_file[d["sample_id"]][f"predicted_{self.args.label_name}"] = int(b)
                data_file[d["sample_id"]][f"prob_{self.args.label_name}"] = float(a)
            if self.args.verbose_eval:
                text = d["text"]
                sample_id = d["sample_id"]
                print(f"\n\nsample_id (index in dataset file): {sample_id}, id: {i}")
                # print(f"\n\nsample_id: {sample_id}")
                print(f"input: {text}")
                print(f"predicted: {b}, prob: {round(a, 5)}")
                print(f"label: {c}")
        if self.args.is_labeling and not self.args.debug:
            dump_file(data_file, f"{self.input_file}")

        return get_scores_multilabel_clf(preds, labels)  # todo


@dataclass
class MetricsComputer:
    tokenizer: PreTrainedTokenizerBase = None
    metric: Any = load_metric("sacrebleu")
    metric2: Any = load_metric("bertscore")
    bart_score_metric: Any = True
    rouge_metric: Any = load_metric("rouge")
    meteor_metric: Any = load_metric("meteor")
    ppl_metric: Any = load_metric("meteor")
    save_dev_location: Any = None
    save_test_location: Any = None
    # no_dl_score: bool = False

    plm: Any = ""
    mask_lens_dev: Any = None

    input_sents: Any = None
    tgt_sents: Any = None

    input_data: Any = None
    input_file: Any = None
    input_nbs: Any = None

    print_input: Any = True

    eval_f: Any = Evaluate()
    model: Any = None
    device: Any = None
    args: Any = None

    def get_eval_res(self, decoded_preds, decoded_labels):
        # Some simple post-processing
        decoded_preds, decoded_labels_special = postprocess_text(decoded_preds, decoded_labels)
        decoded_labels = [sent[0] for sent in decoded_labels_special]

        # Bleu
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels_special)
        result = {"bleu": result["score"]}

        decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels_special))

        # Rouge
        # decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        # result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels))
        rouge = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge = {key: round(value.mid.fmeasure * 100, 1) for key, value in rouge.items()}
        result.update(rouge)

        # Meteor
        meteor = self.meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor["meteor"] *= 100
        result.update(meteor)

        # BertScore
        if not self.args.debug and not self.args.no_dl_score:
            self.model.cpu()
            result_bertscore = self.metric2.compute(predictions=decoded_preds, references=decoded_labels, lang="en", batch_size=64)  # device="cpu"
            result["bertscore"] = sum(result_bertscore['f1']) / len(result_bertscore['f1'])
            result["bertscore"] *= 100
            self.model.to(self.device)

            # if self.bart_score_metric is not None:
            self.model.cpu()
            bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
            bart_scorer.load(path='BARTScore/bart.pth')
            result["bartscore"] = np.mean(bart_scorer.score(decoded_preds, decoded_labels, batch_size=32))
            bart_scorer.model.cpu()
            bart_scorer = None
            self.model.to(self.device)
        else:
            result["bertscore"] = -1
            result["bartscore"] = 1

        return result

    def __call__(self, eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        if "gpt" in self.plm.lower():
            decoded_preds, decoded_labels = [], self.tgt_sents
            for pred, eposs in zip(preds, self.mask_lens_dev):  # labels, lb
                # cur_s=self.tokenizer.batch_decode([pred[eposs:]], skip_special_tokens=False)[0].replace(self.tokenizer.bos_token, "").replace(self.tokenizer.eos_token, " ")
                # decoded_preds.append(cur_s)
                decoded_preds.append(self.tokenizer.batch_decode([pred[eposs:]], skip_special_tokens=True)[0])
        else:
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            if "t5" in self.plm.lower():
                decoded_preds = [item.replace("[label]", " [label]") for item in decoded_preds]
                decoded_labels = [item.replace("[label]", " [label]") for item in decoded_labels]
            # decoded_preds, decoded_labels = [], []
            # for pred, lb in zip(preds, labels):
            #     decoded_preds.append(self.tokenizer.batch_decode([pred], skip_special_tokens=False)[0])
            #     decoded_labels.append(self.tokenizer.batch_decode([lb], skip_special_tokens=False)[0])

        if self.args.text_only or self.args.pred_only:
            txt_decoded_preds = decoded_preds
            txt_decoded_labels = decoded_labels
        else:
            if "t5" in self.plm.lower() or "gpt" in self.plm.lower():
                txt_decoded_preds = [item.split("[label]")[-1].split("[PROFILE]")[-1].split("[POST]")[-1].strip() for item in decoded_preds]
            else:
                txt_decoded_preds = [item.split("[label]")[-1].strip() for item in decoded_preds]
            txt_decoded_labels = [item.split("[label]")[-1].strip() for item in decoded_labels]
        # breakpoint()
        # embed()

        result_txt = self.get_eval_res(txt_decoded_preds, txt_decoded_labels)
        result_txt = modify_dict_keys(result_txt, "txt_")

        # Some simple post-processing
        decoded_preds, decoded_labels_special = postprocess_text(decoded_preds, decoded_labels)
        decoded_labels = [sent[0] for sent in decoded_labels_special]

        # Bleu
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels_special)
        result = {"bleu": result["score"]}

        decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels_special))

        # F1
        # result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        # result = {"f1": result["f1"]}
        # print("result", result)
        def get_labels_from_str(decoded, special_case=False):  # , sent_labels, intensity_labels, tgts_labels
            sent_labelss, intensity_labelss, tgt_labelss = [], [], []

            for s in decoded:
                if special_case:
                    s.replace("[PROFILE]", " [label] ").replace("[POST]", " [label] ").replace("[HISTORY]", " [label] ")

                sent_label, intensity_label, tgt_label = 0, 1, s

                if s.count("[label]") == 2:  # or special_case
                    sent_label, intensity_label, tgt_label = s.split("[label]")

                    # if s.count("[label]") == 2:
                    #     sent_label, intensity_label, tgt_label = s.split("[label]")
                    # else:
                    #     if s.count("[label]")==1 and s.count("[PROFILE]")==1 :
                    #         sent_label, tgt_label = s.split("[PROFILE]")

                    sent_label, intensity_label, tgt_label = sent_label.strip(), intensity_label.strip(), tgt_label.strip()
                    try:
                        sent_label = int(sent_label)
                        if not (0 <= sent_label <= 2):
                            sent_label = 0
                    except:
                        sent_label = 0

                    try:
                        intensity_label = int(intensity_label)
                        if not (0 <= intensity_label <= 6):
                            intensity_label = 1
                    except:
                        intensity_label = 1

                sent_labelss.append(sent_label)
                intensity_labelss.append(intensity_label)
                tgt_labelss.append(tgt_label)
            return sent_labelss, intensity_labelss, tgt_labelss

        sent_preds, intensity_preds, tgt_preds = [], [], []
        sent_labels, intensity_labels, tgt_labels = [], [], []
        if self.args.label_generation:
            # accuracy
            tmp_predcted_labels = [[cs.strip() for cs in item.split()] for item in decoded_preds]  # list of str vectors
            tmp_true_labels = [[cs.strip() for cs in item.split()] for item in decoded_labels]

            ## mrr
            result['mrr'] = get_mrr(tmp_predcted_labels, tmp_true_labels, verbose=False)

            ## NDCG
            ## multilabel
            predcted_labels = convert_strlabels_to_idxlabels(tmp_predcted_labels, emotion_label_map, exclude='inconfident')
            true_labels = convert_strlabels_to_idxlabels(tmp_true_labels, emotion_label_map, exclude='inconfident')
            multilabel_scores = get_scores_multilabel_clf(predcted_labels, true_labels, verbose=False)
            result.update(multilabel_scores)
        else:
            sent_labels, intensity_labels, tgt_labels = get_labels_from_str(decoded_labels)
            sent_preds, intensity_preds, tgt_preds = get_labels_from_str(decoded_preds, special_case="t5" in self.plm.lower() or "gpt" in self.plm.lower())

            clf_score_dict = get_scores_multilabel_clf(sent_preds, sent_labels)
            for key in ["pearson", "spearman", "kappa"]:
                clf_score_dict.pop(key, None)
            # clf_score_dict=modify_dict_keys(clf_score_dict, "intensity_")
            result.update(clf_score_dict)

            clf_score_dict = get_scores_multilabel_clf(intensity_preds, intensity_labels)
            clf_score_dict = {key: value for key, value in clf_score_dict.items() if key in ["pearson", "spearman", "kappa"]}
            # clf_score_dict=modify_dict_keys(clf_score_dict, "intensity_")
            result.update(clf_score_dict)

        # Rouge
        # decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
        # result.update(self.eval_f.evaluate(live=True, cand=decoded_preds_alt, ref=decoded_labels))
        rouge = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge = {key: round(value.mid.fmeasure * 100, 1) for key, value in rouge.items()}
        result.update(rouge)

        # Meteor
        meteor = self.meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor["meteor"] *= 100
        result.update(meteor)

        # BertScore
        if not self.args.debug and not self.args.no_dl_score:
            self.model.cpu()
            result_bertscore = self.metric2.compute(predictions=decoded_preds, references=decoded_labels, lang="en", batch_size=64)  # device="cpu"
            result["bertscore"] = sum(result_bertscore['f1']) / len(result_bertscore['f1'])
            result["bertscore"] *= 100
            self.model.to(self.device)

            # if self.bart_score_metric is not None:
            self.model.cpu()
            bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
            bart_scorer.load(path='BARTScore/bart.pth')
            result["bartscore"] = np.mean(bart_scorer.score(decoded_preds, decoded_labels, batch_size=32))
            bart_scorer.model.cpu()
            bart_scorer = None
            self.model.to(self.device)
        else:
            result["bertscore"] = -1
            result["bartscore"] = 1

        # result2 = metric2.compute(predictions=decoded_preds, references=[sent[0] for sent in decoded_labels], lang="en", device="cpu")
        # result["bertscore"] = sum(result2['f1']) / len(result2['f1'])

        prediction_lens=[]
        if "gpt" in self.plm.lower():
            for pred, eposs in zip(preds, self.mask_lens_dev):  # labels, lb
                prediction_lens.append(np.count_nonzero(pred[eposs:] != self.tokenizer.pad_token_id))
        else:
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        result.update(result_txt)

        result = {k: round(v, 4) for k, v in result.items()}

        if self.print_input:
            assert len(self.input_sents) == len(decoded_preds)
            assert len(self.input_sents) == len(decoded_labels)

            outputs = []
            grouped_outputs = defaultdict(dict)
            aa = load_file(self.args.test_file)
            for i, (a, b, c, sent_pred, intensity_pred, sent_label, intensity_label, d_sample, txt_decoded_pred, txt_decoded_label) in enumerate(
                    zip(self.input_sents, decoded_preds, decoded_labels, sent_preds, intensity_preds, sent_labels, intensity_labels,
                        self.input_data.instances, txt_decoded_preds, txt_decoded_labels)):  # , d , self.input_nbs
                modified_a = a.replace("[POST]", "\n[POST]================\n\n").replace("[PROFILE]", "\n[PROFILE]================\n\n").replace(";", ";\n")

                print(f"\n =========Input========= {i}:\n {modified_a}")
                print(f"Pred: {b}")
                print(f"Ref: {c}")

                if self.args.group_output:
                    post_id, user_id, category = d_sample["extra"]["post_id"], d_sample["extra"]["user_id"], d_sample["extra"]["category"]
                    grouped_outputs[post_id]["headline"] = a.split("[POST]")[0].strip()
                    grouped_outputs[post_id][category] = b.strip()
                # print(f"Pred_Label: {sent_pred}, {intensity_pred}")
                # print(f"Ref_Label: {sent_label}, {intensity_label}")
                outputs.append({
                    "src_text": a.split("[POST]")[0].strip(),
                    "pred": b,
                    "ref": c,
                    "sample_id": d_sample["sample_id_in_orig_data"],
                    # "tgt_text": c,
                    "predicted_comment": txt_decoded_pred,
                    "predicted_sent": sent_pred,
                    "predicted_intensity": intensity_pred,
                    "label_comment": txt_decoded_label,
                    "label_sent": sent_label,
                    "label_intensity": intensity_label,
                })
                # aai=aa[d_sample["sample_id_in_orig_data"]]
                # print()

            if self.args.save_output:
                if self.args.group_output:
                    dump_file(grouped_outputs, f"{self.args.analysis_dir}{self.args.model_name.replace('-', '_')}_{self.args.load_model_path}_grouped_outputs.json")
                else:
                    dump_file(outputs, f"{self.args.analysis_dir}{self.args.model_name.replace('-', '_')}_{self.args.load_model_path}_outputs.json")
                # if self.save_dev_location:
                #     dump_file(outputs, self.save_dev_location)
                # if self.save_test_location:
                #     dump_file(outputs, self.save_test_location)

        return result


def train_clf_trainer(args, model, optimizer, tokenizer, data, id2label=None, eval_only=False, verbose=False):
    train_data, val_data, test_data = None, None, None
    if not args.eval_only:
        train_data, val_data, test_data = data

        print("\n\nlen(train_data)", len(train_data))
        print("len(val_data)", len(val_data))
        print("len(test_data)", len(test_data))
    else:
        # test_data = data
        train_data, val_data, test_data = data
        print("len(test_data)", len(test_data))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        labels = labels.tolist()
        # print("logits, labels ", logits, labels)
        # logits = (torch.sigmoid(get_tensor_float(logits)) > 0.5).int().tolist()
        # get softmax
        logits = torch.softmax(get_tensor_float(logits), dim=1).tolist()
        # get predictions
        preds = [np.argmax(item) for item in logits]

        # convert label into vec

        # labels = [id2label[label] for label in labels]

        return get_scores_multilabel_clf(preds, labels)  # todo

    "=========Train========="""
    tmp_batch_size = args.batch_size if int(args.true_batch_size) == -1 else args.true_batch_size
    num_steps_per_epoch = (len(train_data) // (tmp_batch_size * args.my_num_devices) + 1)

    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    # scheduler = get_scheduler(optimizer, args.scheduler, warmup_steps=warmup_steps, total_steps=total_steps)
    args.eval_steps = int(num_steps_per_epoch // args.num_evals_per_epoch)  # // (args.true_batch_size // (args.batch_size*args.num_devices))
    scheduler = get_scheduler(optimizer, args.scheduler, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                              eta_min=0, T_max=(int(args.num_epochs) // 4) + 1)
    print("args.eval_steps", args.eval_steps)
    print("num_steps_per_epoch", num_steps_per_epoch)
    print("total_steps", total_steps)
    print("warmup_steps", warmup_steps)

    # total_steps = (len(train_data) // args.batch_size + 1) * args.num_epochs
    # warmup_steps = int(total_steps * args.warmup_ratio)
    # print("\ntotal_steps", total_steps)
    # print("warmup_steps", warmup_steps)
    # print("strategy", args.eval_strategy)
    #
    # # args.eval_steps = total_steps // args.num_evals
    # args.eval_steps = (len(train_data) // args.batch_size + 1) // args.num_evals_per_epoch // (args.true_batch_size // args.batch_size)
    # print("eval_steps", args.eval_steps)

    # scheduler = get_scheduler(optimizer, args.scheduler, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    #                           eta_min=0, T_max=(int(args.num_epochs) // 4) + 1)
    print("args.true_batch_size // args.batch_size", args.true_batch_size // args.batch_size)
    print("args.batch_size", args.batch_size)
    training_args = TrainingArguments(
        evaluation_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=bool(args.use_amp),
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay,
        # adam_epsilon=args.adam_epsilon,
        num_train_epochs=args.num_epochs,
        learning_rate=args.plm_lr,
        seed=args.seed,
        load_best_model_at_end=True,
        # label_smoothing_factor=args.label_smoothing_factor,
        lr_scheduler_type=SchedulerType(args.scheduler),
        report_to=["wandb"] if not args.debug else [],
        metric_for_best_model=args.metric_for_best_model,
        logging_steps=args.eval_steps,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        # warmup_steps=warmup_steps,
        save_total_limit=2,
        group_by_length=True,  # todo True
        no_cuda=bool(not args.use_gpu),
        greater_is_better=True,  # todo
        # gradient_accumulation_steps=args.true_batch_size // args.batch_size,  # todo
        # debug="underflow_overflow",
        # run_name=args.exp,
        # dataloader_pin_memory=False,
        # do_train=False
    )

    # data_collator = CustomCollator(tokenizer, model=model)
    data_collator = CustomCollatorCLF(tokenizer, max_length=args.max_seq_len, collect_input=args.eval_only, collect_fields=["input_ids"])  # collate_wrapper # todo

    metrics_computer_dev = MetricsComputerCLF(tokenizer=tokenizer, args=args, input_data=val_data, input_file=args.dev_file, model=model)

    trainer = MultiClassSingleLabelTrainer(  # todo
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=metrics_computer_dev,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )
    trainer.add_attribute(deepcopy(train_data.get_class_weights()).to(args.device), args.weighted_loss)

    # trainer = Trainer(  # todo
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     optimizers=(optimizer, scheduler),
    #     compute_metrics=compute_metrics,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     data_collator=data_collator,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    # )

    if not args.eval_only:
        trainer.train()

        if not args.debug:
            import shutil
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{wandb.run.id}.pt"))
            # shutil.copy(args.model_path, os.path.join(wandb.run.dir,f"{wandb.run.id}.pt"))
            wandb.save(os.path.join(wandb.run.dir, f"{wandb.run.id}.pt"))
            # wandb.save(args.model_path)
            # wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))
        else:
            torch.save(model.state_dict(), args.model_path)
        # get best val performance
        best_metric = trainer.state.best_metric
        for hist in trainer.state.log_history[::-1]:
            if f"eval_{args.metric_for_best_model}" in hist and hist[f"eval_{args.metric_for_best_model}"] == best_metric:
                tmp_dict = {k.replace("eval_", "eval/"): v for k, v in hist.items()}
                tmp_dict["train/global_step"] = trainer.state.global_step + 1
                print("best val", tmp_dict)
                if not args.debug:
                    wandb.log(tmp_dict)  # , step=trainer.state.global_step+1
                break
        # torch.save(model.state_dict(), args.model_path)

    model.eval()
    # print("\n\n===Running Dev===")
    # if args.eval_only and args.run_dev:
    #     dev_score_dict = trainer.evaluate(metric_key_prefix="eval", eval_dataset=val_data)  # to get best dev eval metric
    #     print("dev score_dict", dev_score_dict)
    #     if not args.debug:
    #         wandb.log({k.replace("eval_", "eval/"): v for k, v in dev_score_dict.items()})

    if args.no_test:
        return
    print("\n\n===Running Test===")
    # get test  performance

    pairs = [(args.train_file, "labeling_train", train_data), (args.dev_file, "eval", val_data), (args.test_file, "test", test_data)]
    for split in args.target_splits:
        cur_file, cur_name, cur_data = pairs[int(split)]
        metrics_computer_test = MetricsComputerCLF(tokenizer=tokenizer, args=args, input_data=cur_data, input_file=cur_file)
        trainer.compute_metrics = metrics_computer_test
        score_dict = trainer.evaluate(metric_key_prefix=cur_name, eval_dataset=cur_data)  # to get best dev eval metric
        print(f"test score_dict for {cur_name}", score_dict)
        if not args.debug and cur_name != "labeling_train":
            wandb.log({k.replace(f"{cur_name}_", f"{cur_name}/"): v for k, v in score_dict.items()})
    # return score_dict


def eval_func(args, model, test_data, data_collator, tokenizer, binary=True, need_label_mapping=False):
    print("\n\n===Testing===")

    metric = load_metric("sacrebleu")  # get_metric_program("sacrebleu", slow_connection=args.slow_connection)
    metric2 = load_metric("bertscore")

    input_nbs = []
    for item in test_data:
        if 'nbs_input_ids' in item and len(item['nbs_input_ids']) and "cbr" in args.components and "bsl" not in args.model_name:
            input_nbs.append(tokenizer.batch_decode([item['nbs_input_ids'][0]])[0])
        else:
            input_nbs.append("None Used")
    metrics_computer_test = MetricsComputer(tokenizer=tokenizer, metric2=metric2,
                                            input_sents=[item['input_text'] for item in test_data], input_nbs=input_nbs,
                                            is_script_completion=args.script_completion, task_mode=args.task_mode, plm=args.plm, model=model, device=args.device, args=args)

    trainer.compute_metrics = metrics_computer_test
    score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    print("test score_dict", score_dict)
    if not args.debug:
        wandb.log({k.replace("test_", "test/"): v for k, v in score_dict.items()})

    """saving"""
    if not args.debug:
        # latest_state_file=glob.glob(os.path.join(args.model_path, "*.pt"))[-1]
        wandb.save(args.output_dir + "*checkpoint*")


def train(args, model, optimizer, tokenizer, data):
    train_data, val_data, test_data = data

    print("\n\nlen(train_data)", len(train_data))
    print("len(val_data)", len(val_data))
    print("len(test_data)", len(test_data))

    # rouge = datasets.load_metric("rouge")

    # reenable
    # if not args.debug:
    "=========Metric========="""
    # metric = load_metric("sacrebleu")  # get_metric_program("sacrebleu", slow_connection=args.slow_connection)
    # metric2 = load_metric("bertscore")

    # print("len(train_data.instances)", len(train_data.instances))
    # print("len(val_data.instances)", len(val_data.instances))

    # rouge=None
    # def compute_metrics(pred):
    #
    #     labels_ids = pred.label_ids
    #     pred_ids = pred.predictions
    #
    #     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    #     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    #     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #
    #     rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
    #
    #     return {
    #         "rouge2_precision": round(rouge_output.precision, 4),
    #         "rouge2_recall": round(rouge_output.recall, 4),
    #         "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    #     }
    "=========Scheduler========="""
    # breakpoint()
    tmp_batch_size = args.batch_size if int(args.true_batch_size) == -1 else args.true_batch_size
    num_steps_per_epoch = (len(train_data) // (tmp_batch_size * args.my_num_devices) + 1)

    total_steps = num_steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    # scheduler = get_scheduler(optimizer, args.scheduler, warmup_steps=warmup_steps, total_steps=total_steps)
    args.eval_steps = int(num_steps_per_epoch // args.num_evals_per_epoch)  # // (args.true_batch_size // (args.batch_size*args.num_devices))
    scheduler = get_scheduler(optimizer, args.scheduler, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
                              eta_min=0, T_max=(int(args.num_epochs) // 4) + 1)
    "=========Train========="""
    print("\ntotal_steps", total_steps)
    # args.eval_steps = 1000#total_steps // 4
    print("eval_steps", args.eval_steps)
    print("warmup_steps", warmup_steps)
    print("strategy", args.eval_strategy)
    print("args.plm_lr", args.plm_lr)
    """don not use scheduler at start according to exp"""
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy=args.eval_strategy,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=bool(args.use_amp),
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay,
        # adam_epsilon=args.adam_epsilon,
        num_train_epochs=args.num_epochs,
        learning_rate=args.plm_lr,
        seed=args.seed,
        load_best_model_at_end=True,
        label_smoothing_factor=args.label_smoothing_factor,
        lr_scheduler_type=SchedulerType(args.scheduler),
        report_to=["wandb"] if not args.debug else [],
        metric_for_best_model=args.metric_for_best_model,
        logging_steps=args.eval_steps,
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        group_by_length=True,  # todo True
        no_cuda=bool(not args.use_gpu),
        greater_is_better=bool(args.greater_is_better),  # todo
        gradient_accumulation_steps=tmp_batch_size // args.batch_size,  # todo
        # debug="underflow_overflow",
        # run_name=args.exp,
        # dataloader_pin_memory=False,
        # do_train=False
    )

    mask_lens_dev = []
    data_collator = CustomCollator(tokenizer, model=model, max_length=args.max_seq_len, use_special_tag=args.use_special_tag, verbose=args.debug,
                                   args=args) if "gpt" not in args.model_name \
        else CustomCollatorCLM(tokenizer, model=model, max_length=args.max_seq_len, verbose=args.debug, mask_lens_dev=mask_lens_dev, args=args)

    metrics_computer_dev = MetricsComputer(tokenizer=tokenizer, input_sents=[item['src_text'] for item in val_data], tgt_sents=[item['tgt_text'] for item in val_data],
                                           plm=args.plm, model=model, mask_lens_dev=mask_lens_dev, input_data=val_data,
                                           device=args.device, args=args, save_dev_location=args.save_dev_location)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, scheduler),
        compute_metrics=metrics_computer_dev,  # compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    # print("args.model_path", args.model_path)`
    # score_dict = trainer.evaluate(metric_key_prefix="dev", eval_dataset=test_data)  # to get best dev eval metric
    # print("\n\n===Testing Degbuggging===")
    # input_nbs = []
    # metrics_computer_test = MetricsComputer(tokenizer=tokenizer, metric2=metric2,
    #                                         input_sents=[item['input_text'] for item in test_data], input_nbs=input_nbs,
    #                                         is_script_completion=args.script_completion, task_mode=args.task_mode, plm=args.plm, model=model, device=args.device, args=args)
    #
    # trainer.compute_metrics = metrics_computer_test
    # score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    # print("test score_dict", score_dict)
    # print("\n\n===Testing Degbuggging===")

    if not args.eval_only:
        trainer.train()
        print("Trained")
        # torch.save(model.state_dict(), args.model_path)
        # trainer.evaluate()  # to get best dev eval metric
        # torch.save(model, "model/states/best_one.pt")

        # all_ckpts=list(glob("model/states/checkpoint-*"))
        # all_ckpts_ids=np.array([int(item.split("checkpoint-")[-1]) for item in all_ckpts])
        # print("all_ckpts_ids",all_ckpts_ids)
        # best_ckpt = all_ckpts[all_ckpts_ids.argsort()[-1]]
        # print("best_ckpt", best_ckpt)
        # model = model.from_pretrained(best_ckpt).to(args.device)

        # model = torch.load("model/states/best_one.pt")

        # test data
        # model.load_state_dict(torch.load(args.model_path))
        # model.eval()
        # # embed()
        # evaluate(args, model, test_data, tokenizer, metric2=metric2)

        # get test  performance
        # torch.save(model.state_dict(), args.model_path)
        if not args.debug:
            import shutil
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"{wandb.run.id}.pt"))
            # shutil.copy(args.model_path, os.path.join(wandb.run.dir,f"{wandb.run.id}.pt"))
            wandb.save(os.path.join(wandb.run.dir, f"{wandb.run.id}.pt"))
            # wandb.save(args.model_path)
            # wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))
        else:
            torch.save(model.state_dict(), args.model_path)
        # get best val performance
        gib_metric_names = {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "bleu",
                            "bertscore", "bartscore", "meteor", "pearson", "kappa", "spearson",
                            "rouge1", "rouge2", "rougeL", "rougeLsum",
                            "mrr", "accuracy", "mif1", "maf1", "mirecall", "marecall", "miprecision", "maprecision"}
        for met in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "bleu", "bertscore", "bartscore", "meteor", "rouge1", "rouge2", "rougeL", "rougeLsum"]:
            gib_metric_names.add("txt_" + met)

        lib_metric_names = {"loss"}
        general_metric_names = {"eval_gen_len", "eval_steps_per_second", "step", "epoch", "eval_samples_per_second", "eval_runtime", }

        best_metric = trainer.state.best_metric
        tmp_dict = None

        valid_general_metric_names = None
        valid_gib_metric_names = None
        valid_lib_metric_names = None
        for hist in trainer.state.log_history:
            if f"eval_{args.metric_for_best_model}" in hist:
                if tmp_dict is None:
                    tmp_dict = hist.copy()
                else:
                    if valid_general_metric_names is None: valid_general_metric_names = set([m for m in general_metric_names if f"eval_{m}" in hist])
                    if valid_gib_metric_names is None: valid_gib_metric_names = set([m for m in gib_metric_names if f"eval_{m}" in hist])
                    if valid_lib_metric_names is None: valid_lib_metric_names = set([m for m in lib_metric_names if f"eval_{m}" in hist])
                    if hist[f"eval_{args.metric_for_best_model}"] == best_metric:
                        for m in valid_general_metric_names:
                            tmp_dict[m] = hist[m]
                    for m in valid_gib_metric_names:
                        m = f"eval_{m}"
                        tmp_dict[m] = max(tmp_dict[m], hist[m])
                    for m in valid_lib_metric_names:
                        m = f"eval_{m}"
                        tmp_dict[m] = min(tmp_dict[m], hist[m])
        # for hist in trainer.state.log_history:
        #     if f"eval_{args.metric_for_best_model}" in hist:
        #         if tmp_dict is None:
        #             tmp_dict = hist.copy()
        #         else:
        #             if hist[f"eval_{args.metric_for_best_model}"] == best_metric:
        #                 for m in general_metric_names:
        #                     tmp_dict[m] = hist[m]
        #             for m in gib_metric_names:
        #                 m = f"eval_{m}"
        #                 tmp_dict[m] = max(tmp_dict[m], hist[m])
        #             for m in lib_metric_names:
        #                 m = f"eval_{m}"
        #                 tmp_dict[m] = min(tmp_dict[m], hist[m])
        if not tmp_dict:
            print("tmp_dict is None")
            embed()

        tmp_dict = {k.replace("eval_", "eval/"): v for k, v in tmp_dict.items()}
        tmp_dict["train/global_step"] = trainer.state.global_step + 1
        print("best val", tmp_dict)
        if not args.debug:
            wandb.log(tmp_dict)  # , step=trainer.state.global_step+1
        # for hist in trainer.state.log_history[::-1]:
        #     if f"eval_{args.metric_for_best_model}" in hist and hist[f"eval_{args.metric_for_best_model}"] == best_metric:
        #         tmp_dict = {k.replace("eval_", "eval/"): v for k, v in hist.items()}
        #         tmp_dict["train/global_step"] = trainer.state.global_step + 1
        #         print("best val", tmp_dict)
        #         if not args.debug:
        #             wandb.log(tmp_dict)  # , step=trainer.state.global_step+1
        #         break

        # get test  performance
        torch.save(model.state_dict(), args.model_path)
        if not args.debug: wandb.save(os.path.join(wandb.run.dir, "best_model.pt"))

    print("\n\n===Running Dev===")
    if args.eval_only and args.run_dev:
        mask_lens_dev.clear()
        dev_score_dict = trainer.evaluate(metric_key_prefix="eval", eval_dataset=val_data)  # to get best dev eval metric
        print("dev score_dict", dev_score_dict)
        if not args.debug:
            wandb.log({k.replace("eval_", "eval/"): v for k, v in dev_score_dict.items()})

    if args.no_test:
        return

    # clean gpu memory
    # model = model.cpu()
    # time.sleep(5)

    """testing"""
    print("\n\n===Testing===")
    mask_lens_dev.clear()
    metrics_computer_test = MetricsComputer(tokenizer=tokenizer, input_sents=[item['src_text'] for item in test_data],
                                            plm=args.plm, model=model, tgt_sents=[item['tgt_text'] for item in test_data], input_data=test_data,
                                            device=args.device, args=args, save_test_location=args.save_test_location, mask_lens_dev=mask_lens_dev)
    trainer.compute_metrics = metrics_computer_test
    score_dict = trainer.evaluate(metric_key_prefix="test", eval_dataset=test_data)  # to get best dev eval metric
    print("test score_dict", score_dict)
    if not args.debug:
        wandb.log({k.replace("test_", "test/"): v for k, v in score_dict.items()})

    # """saving"""
    # if not args.debug:
    #     # latest_state_file=glob.glob(os.path.join(args.model_path, "*.pt"))[-1]
    #     wandb.save(args.output_dir + "*checkpoint*")

    # if self.args.visualize_scatterplot:
    #     tmp = {}
    #     if "rouge" in self.args.scatterplot_metrics:
    #         rouges = [self.rouge_metric.compute(predictions=[decoded_pred], references=[decoded_label]) for i, (decoded_pred, decoded_label) in
    #                   enumerate(tqdm(zip(decoded_preds, decoded_labels)))]
    #         for j, item in enumerate(tqdm(rouges)):
    #             for key, value in item.items():
    #                 item[key] = round(value.mid.fmeasure * 100, 1)
    #
    #         tmp["rouges"] = rouges
    #         dump_file(tmp, f"{self.args.analysis_dir}{self.args.model_name}_results.json")
    #
    #     if "bleu" in self.args.scatterplot_metrics:
    #         bleus = []
    #         for j, (decoded_pred, decoded_label) in enumerate(tqdm(zip(decoded_preds, decoded_labels_special))):
    #             print(decoded_pred, decoded_label)
    #             bleus.append(self.metric.compute(predictions=[decoded_pred], references=[decoded_label if decoded_label else "."]))
    #
    #         tmp["bleus"] = bleus
    #         # embed()
    #         BLEUs = []
    #         # BLEUs = [self.eval_f.evaluate(live=True, cand={j:decoded_preds_alt[j]}, ref=[decoded_label]) for j, decoded_label in tqdm(enumerate(decoded_labels_special), desc="BLEUs")]
    #         decoded_preds_alt = {i: item for i, item in enumerate(decoded_preds)}  # dict style
    #         for j, decoded_label in tqdm(enumerate(decoded_labels_special), desc="BLEUs"):
    #             BLEUs.append(self.eval_f.evaluate(live=True, cand={0: decoded_preds_alt[j]}, ref=[decoded_label]))
    #         tmp["BLEUs"] = BLEUs
    #         # embed()
    #         dump_file(tmp, f"{self.args.analysis_dir}{self.args.model_name}_results.json")
    #
    #     if "meteor" in self.args.scatterplot_metrics:
    #         meteors = [self.meteor_metric.compute(predictions=[decoded_pred], references=[decoded_label]) for decoded_pred, decoded_label in zip(decoded_preds, decoded_labels)]
    #         for item in meteors:
    #             for key, value in item.items():
    #                 item[key] = value * 100
    #         tmp["meteors"] = meteors
    #
    #     # print("tmp", tmp)
    #     # embed()
    #     dump_file(tmp, f"{self.args.analysis_dir}{self.args.model_name}_results.json")
