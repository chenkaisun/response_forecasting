from torch.utils.data import DataLoader
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    precision_score, recall_score, cohen_kappa_score
import numpy as np
# from data import collate_wrapper
# from pprint import pprint as pp
# from sklearn.metrics import accuracy_score
# from train_utils import seed_worker

from datasets import load_dataset, load_metric
# import numpy as np
# from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding
# from datasets import Dataset, load_dataset
# from torch.utils.data.dataloader import DataLoader
# from glob import glob
# from data import pad_to_batch# , CustomBatch, collate_wrapper
from data_collator import CustomCollator# , CustomBatch, collate_wrapper
from train_utils import get_tensor_long, get_tensor_float
from pprint import pprint as pp
from tqdm import tqdm
from eval_final import Evaluate
from rouge import Rouge
import wandb
from IPython import embed
from utils.utils import *
from utils.data_utils import *



from scipy.stats import pearsonr, spearmanr

def get_prf(targets, preds, average="micro", verbose=False):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def convert_to_dict(ls):
    # ls = [self[i] for i in range(len(self))]
    keys = list(ls[0].keys())
    ress = {}
    for key in keys:
        ress[key] = [i[key] for i in ls]

    # print("\nres.keys()", res.keys())
    return ress
#
# def gen_metric_compute(preds, labels):
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
#     result = {"bleu": result["score"]}
#
#     result2 = metric2.compute(predictions=decoded_preds, references=[sent[0] for sent in decoded_labels], lang="en")
#     result["bertscore"] = result2['f1'] / len(result2['f1'])


# class GenerationEvaluator:
#     def __init__(self):
#         self.sacrebleu= load_metric("sacrebleu")
#         self.bertscore= load_metric("bertscore")
#         self.general=Evaluate()
#
#     def compute(self, preds, references, dev=True):
#
#         ## BLEU's
#
#         preds_alt = {i: item for i, item in enumerate(preds)}  # dict style
#         references_alt = [item[0] for item in references]  # no streaming style
#         final_scores = self.general.evaluate(live=True, cand=preds_alt, ref=references)
#         # ## Rouge
#         # rouge = Rouge()
#         # # final_scores.update(rouge.get_scores(preds, references_alt, avg=True))
#         # print(rouge.get_scores(preds, references_alt, avg=True))
#         ## SacreBleu
#         result = self.sacrebleu.compute(predictions=preds, references=references)
#         final_scores["sacrebleu"] = result['score']
#
#         result = self.bertscore.compute(predictions=preds, references=[sent[0] for sent in references], lang="en")
#         final_scores["bertscore"] = sum(result2['f1']) / len(result['f1'])
#         # final_scores['epoch']=0
#         return final_scores

        # if get_scores:
        #     return final_scores

def get_mrr(preds, labels, verbose=False):
    total_score = 0
    score_cnt = 0
    for i, (pl, tl) in enumerate(zip(preds, labels)):
        assert len(tl), breakpoint()
        if tl:
            score_cnt += 1
            if tl[0] in pl:
                total_score += 1 / (pl.index(tl[0]) + 1)

    return round(total_score / score_cnt * 100, 4)

def get_scores_multilabel_clf(preds, labels, verbose=False):
    # print("get_scores_multilabel_clf")
    # print("logits, labels", logits, labels)
    # preds = (torch.sigmoid(get_tensor_float(logits)) > 0.5).int().tolist()

    score_dict = {}
    mi_precision, mi_recall, mi_f1 = get_prf(labels, preds, average="micro", verbose=False)
    ma_precision, ma_recall, ma_f1 = get_prf(labels, preds, average="macro", verbose=False)
    #get pearson and spearman

    pearson = pearsonr(preds, labels)[0]
    # print(list(zip(preds, labels)))
    spearman = spearmanr(preds, labels)[0]
    kappa = cohen_kappa_score(preds, labels, weights="quadratic")
    if np.isnan(pearson):
        pearson = 0
    if np.isnan(spearman):
        spearman = 0
    if np.isnan(kappa):
        kappa = 0
    score_dict.update({
        "mif1": mi_f1,
        "maf1": ma_f1,
        "accuracy": accuracy_score(labels, preds),
        "miprecision": mi_precision,
        "mirecall": mi_recall,
        "maprecision": ma_precision,
        "marecall": ma_recall,
        "pearson": pearson,
        "spearman": spearman,
        "kappa": kappa,
    })
    for key in score_dict:
        score_dict[key] = round(score_dict[key]*100, 4)
    return score_dict





def get_scores_binary_clf(preds, labels, num_decimals=4):
    # preds = np.argmax(logits, axis=-1)

    score_dict = {}
    try:
        precision, recall, f1 = get_prf(labels, preds, average="binary", verbose=False)
    except Exception as e:
        print(e)
        embed()
    score_dict.update({
        "f1": round(f1, num_decimals),
        "accuracy":  round(accuracy_score(labels, preds), num_decimals),
        "precision":  round(precision, num_decimals),
        "recall":  round(recall, num_decimals),
    })
    return score_dict


def evaluate(args, model, data, tokenizer, no_scoring=False, metric2=None):
    data_collator=CustomCollator(tokenizer=tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)
    # m_names=["sacrebleu", "bleu", "rouge"]
    # metrics = [load_metric(m_name) for m_name in m_names]
    metric = load_metric("sacrebleu")

    def generate_summary(batch):
        input_ids, attention_mask=batch['input_ids'], batch['attention_mask']
        decoded_input_display = tokenizer.batch_decode(input_ids, skip_special_tokens=False)

        input_ids, attention_mask = input_ids.to(args.device), attention_mask.to(args.device)

        if "bsl" in args.model_name:
            outputs = model.generate(input_ids, attention_mask=attention_mask)
        else:
            outputs = model.generate(input_ids, top_p=0.92, attention_mask=attention_mask, g_data=batch['g_data'], g_data2=batch['g_data2'],
                                 token2nodepos=batch['token2nodepos'], event_position_ids=batch['event_position_ids'])

        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["labels"] = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)


        # print("batch[labels]", batch["labels"])
        output_str, batch["labels"] = postprocess_text(output_str, batch["labels"])
        print("\n\nComparisons")
        # print("lbs", batch["labels"])
        for i, (a, b, c) in enumerate(zip(decoded_input_display, output_str, batch["labels"])):

            # if a.count("[SEP]")==1:
            #     print("\n\n\n")
            print("\n\n[Current Steps]: ", a.replace(tokenizer.cls_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.sep_token, "\n"+" "*18).strip(), )
            print("[Predicted Next Step]: ", b, )
            print("[True Next Step]: ", c[0])

        batch["output_str"] = output_str
        return batch

    eval_loader = DataLoader(data, collate_fn=data_collator, batch_size=args.batch_size)
    # inputs=[]
    preds=[]
    references=[]
    print("\n\nComparisons")
    for i, batch in enumerate(tqdm(eval_loader)):
        tmp=generate_summary(batch)
        # inputs+=tmp["input_ids"]
        preds+=tmp["output_str"]
        references+=tmp["labels"]

    if no_scoring: return


    print("\n\nScoring...")
    # breakpoint()
    preds_alt = {i:item for i,item in enumerate(preds) } # dict style
    references_alt = [item[0] for item in references] #no streaming style

    ## BLEU's
    eval_f = Evaluate()
    final_scores = eval_f.evaluate(live=True, cand=preds_alt, ref=references)

    ## Rouge
    # rouge = Rouge()
    # # final_scores.update(rouge.get_scores(preds, references_alt, avg=True))
    # print(rouge.get_scores(preds, references_alt, avg=True))
    ## SacreBleu
    result = metric.compute(predictions=preds, references=references)
    final_scores["sacrebleu"]=result['score']

    print("references[0]", references[0])
    result2=metric2.compute(predictions=preds, references=[sent[0] for sent in references], lang="en", device="cpu")
    final_scores["bertscore"]=sum(result2['f1'])/len(result2['f1'])
    # final_scores['epoch']=0


    for k in final_scores:
        wandb.run.summary["test/"+k] = final_scores[k]
    # wandb.log(final_scores, step)
    print("final_scores", final_scores)
    return final_scores
    ##Meteor

    #
    #
    #
    # # result = metric.compute(predictions=pred_labels, references=true_labels)
    # result = {"bleu": result["score"]}
    #
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in inputs]# results['input_ids']
    # result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    # print("test result", result)
    # return result
    # print(rouge.compute(predictions=results["pred_summary"], references=results["ref"], rouge_types=["rouge2"])[
    #           "rouge2"].mid)


def get_prf(targets, preds, average="micro", verbose=False):
    precision = precision_score(targets, preds, average=average, zero_division=0)
    recall = recall_score(targets, preds, average=average, zero_division=0)
    f1 = f1_score(targets, preds, average=average, zero_division=0)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1


def evaluate_clf(args, model, data, split="dev", binary=False, data_collator=None, verbose=False, tokenizer=None, need_label_mapping=False):
    # print("Evaluate")

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator, drop_last=False)
    preds = []
    targets = []
    target_labels = []
    ids = []
    original_text_list = []

    id2label = data.id2label
    # print("id2label", id2label)
    cc = {'authority': 0, 'betrayal': 1, 'care': 2, 'cheating': 3, 'degradation': 4,
          'fairness': 5, 'harm': 6, 'loyalty': 7, 'non-moral': 8, 'purity': 9, 'subversion': 10}

    cls_id2label = {0: 'non-moral', 1: "care/harm", 2: "cheating/fairness", 3: "loyalty/betrayal", 4: "authority/subversion", 5: "purity/degradation"}

    def convert_labels(lab):
        new_lab = [lab[8]] + [0] * 5  # non moral
        new_lab[1] = int(lab[2] or lab[6])  # care harm
        new_lab[2] = int(lab[3] or lab[5])  # cheat fair
        new_lab[3] = int(lab[1] or lab[7])  # loyalty
        new_lab[4] = int(lab[0] or lab[10])  # subversion
        new_lab[5] = int(lab[4] or lab[9])  # degradation
        # print("lab", lab)
        # print("new_lab", new_lab)
        return new_lab

    model.eval()
    for batch in tqdm(dataloader):

        inputs = send_to_device(batch, exclude=['ids'], device=args.device)
        inputs["in_train"] = False

        with torch.no_grad():
            pred = model(**inputs)['logits']
            try:
                cur_preds = [[id2label[i] for i, e in enumerate(vec) if e != 0] for vec in (torch.sigmoid(pred) > 0.5).int().tolist()]

                pred = (torch.sigmoid(get_tensor_float(pred)) > 0.5).int().tolist()
                # pred = pred.tolist()

                preds.extend(pred)

                ids.extend(inputs['ids'])
                cur_labels = inputs['labels'].tolist()

                targets.extend(cur_labels)

                assert len(inputs['ids']) == len(cur_preds), print("len not equal")
                # original_text_list.extend(cur_text)
                if verbose:
                    cur_text = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                    cur_tweet_ids = [data[id]["tweet_id"] for id in inputs['ids']]
                    cur_labelnames = [[id2label[i] for i, e in enumerate(vec) if e != 0] for vec in cur_labels]
                    for i, (id, tweet_id, text, pred_vec, target_vec) in enumerate(zip(inputs['ids'], cur_tweet_ids, cur_text, cur_preds, cur_labelnames)):
                        if pred_vec != target_vec:
                            print(id, tweet_id, text.strip(), "\npred:", pred_vec, " label:", target_vec, "\n")
            except Exception as e:
                print(e)
                embed()

            # print("pred", pred)
            # print("targets", targets)
            # print("ids", ids)

    """convert labels"""
    if need_label_mapping:
        preds = [convert_labels(tmp) for tmp in preds]
        targets = [convert_labels(tmp) for tmp in targets]

    preds = np.array(preds)

    score_dict = {}
    score_dicts = []

    if binary:
        # precision, recall, f1 = get_prf(targets, preds, average="binary", verbose=True)
        # score_dict.update({
        #     "precision": precision,
        #     "recall": recall,
        #     "f1": f1,
        # })
        # acc = accuracy_score(targets, preds)
        # score_dict["accuracy"] = acc
        targets = np.array(targets)
        # print("preds.shape", preds.shape)
        # print("id2label",id2label)
        for cls in range(preds.shape[-1]):
            # print("cls", id2label[cls])
            sub_preds, sub_targets = preds[:, cls], targets[:, cls]
            # print("sub_preds, sub_targets",sub_preds, sub_targets)
            # embed()
            score_dict.update(modify_dict_keys(get_scores_binary_clf(sub_preds, sub_targets), prefix=id2label[cls]+"_"))
            # print(score_dict)
            score_dicts.append(score_dict)

    else:
        score_dict = get_scores_multilabel_clf(preds, targets)
        # precision, recall, score2 = get_prf(targets, preds, average="macro", verbose=True)
        # mi_precision, mi_recall, mi_f1 = get_prf(targets, preds, average="micro", verbose=True)
        # ma_precision, ma_recall, ma_f1 = get_prf(targets, preds, average="macro", verbose=True)
        # score_dict.update({"mi_precision": mi_precision,
        #                    "mi_recall": mi_recall,
        #                    "mi_f1": mi_f1,
        #                    "ma_precision": ma_precision,
        #                    "ma_recall": ma_recall,
        #                    "ma_f1": ma_f1,
        #                    })
        # acc = accuracy_score(targets, preds)
        # score_dict["accuracy"] = acc
        # print("score_dict", score_dict)
        score_dicts.append(score_dict)
    if verbose: print("subset", args.subset)
    # args.log  ger.debug(f"score_dict")
    # args.logger.debug(score_dict)

    # score_dict = {(split + "/" + k): item for k, item in score_dict.items()}

    # if not args.debug and split == "test":
    #     for k in score_dict:
    #         wandb.run.summary[k] = score_dict[k]

    # output = None
    return score_dict, [list(item) for item in zip(ids, preds, targets)]

def generate_steps(args, model, data, tokenizer, data_collator=None, no_scoring=False):
    pass
