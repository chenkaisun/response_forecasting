import numpy as np
from torch.utils.data import Dataset

import TweetNormalizer
# from copy import deepcopy
# from datasets import load_dataset
from transformers.data.data_collator import DataCollatorWithPadding
# from copy import deepcopy
# import csv
# import scipy
from utils.utils import *
from utils.data_utils import *

if module_exists("torch_geometric"):
    from torch_geometric.data import Batch, Data
from copy import deepcopy, copy
from multiprocessing import Pool
from tqdm import tqdm
from train_utils import get_tensor_long, get_tensor_float
from collections import defaultdict, OrderedDict
from dataclasses import dataclass

from typing import Any, List, Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers import PreTrainedTokenizerBase
import re
import nltk
from nltk.corpus import stopwords
import spacy
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import preprocessor as p
from constants import TASK_SETTINGS
from sklearn.utils.class_weight import compute_class_weight


def fill_array_at_positions(length, positions, null_val=0, val=1, ):
    label_vec = [null_val] * length
    for pos in positions:
        label_vec[pos] = val
    return label_vec


class PrimitiveDataset(Dataset):

    def __init__(self, args, filename, tokenizer=None, in_train=False):

        print("\nLoading Dataset...")

        "=============Loading Cache============="""
        # print("filename", filename)
        args.cache_filename = os.path.splitext(filename)[0] + ("primitive_d") + ".pkl"
        if args.use_cache and os.path.exists(args.cache_filename):
            print("Loading Cached Data...", args.cache_filename)
            self.instances = load_file(args.cache_filename)
            return

        "=============Loading============="""
        print("loading", filename)
        self.original_data = load_file(filename)

        self.instances = []
        # print("in_train and args.augment_with_translation", in_train and args.augment_with_translation)
        # print("(not in_train and do_foreign_eval)", (not in_train and args.do_foreign_eval))

        for idx, sample in enumerate(self.original_data):
            # if "non-moral" in sample["labels"]:
            #     continue
            text = sample["text"]
            if 'doc_pos' in sample:
                # post_text=self.original_data[sample["doc_pos"]]["text"]
                # text = f"'{post_text}'. {text}"
                text = text

            self.instances.append({"text": text,
                                   "sample_id": idx,
                                   "tweet_id": sample["tweet_id"] if "tweet_id" in sample else None,
                                   "input_ids": tokenizer.tokenize(text),
                                   })
        if args.cache_filename:
            dump_file(self.instances, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


class PrimitivePredictionDataset(Dataset):

    def __init__(self, args, filename, tokenizer=None, labels=None, label_map=None, in_train=False, extra_data=None):

        print("\nLoading Dataset...")

        self.SEP = tokenizer.sep_token_id
        self.CLS = tokenizer.cls_token_id
        self.BOS = tokenizer.bos_token_id
        self.EOS = tokenizer.eos_token_id
        self.SEP_TOKEN = tokenizer.sep_token
        self.CLS_TOKEN = tokenizer.cls_token
        self.BOS_TOKEN = tokenizer.bos_token
        self.EOS_TOKEN = tokenizer.eos_token

        if self.SEP is None:
            self.SEP = self.EOS
            self.SEP_TOKEN = self.EOS_TOKEN

        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label_map = label_map
        print("\nself.label2id", self.label2id)
        print("len self.label2id", len(self.label2id))

        if not args.database_io:
            "=============Loading Cache============="""
            # print("filename", filename)
            args.cache_filename = os.path.splitext(filename)[0] + ".pkl"
            if args.use_cache and os.path.exists(args.cache_filename):
                print("Loading Cached Data...", args.cache_filename)
                self.instances = load_file(args.cache_filename)
                return

            "=============Loading============="""
            print("loading", filename)
            self.original_data = load_file(filename)
            # if isinstance(self.original_data, dict):
            #     self.original_data
            if ".csv" in filename:
                json_str = self.original_data.to_json(orient="records")
                self.original_data = json.loads(json_str)

            tmp = []
            if args.personal_response_pred:
                user_data, post_data, history_data, graph_data = extra_data
                print("personal_response_pred")

                cnt_empty_user_attributes = 0
                num_empty_desc = 0
                for i, sample in enumerate(tqdm(self.original_data)):  # this indicates ith path after breaking out all articles into individual paths
                    user_id = sample["author_id"]
                    post_id = sample["conversation_id"]
                    if str(user_id) not in user_data:
                        user_desc = ""
                        # continue
                    else:
                        user_desc = user_data[str(user_id)]['description']
                        if not user_desc.strip():
                            num_empty_desc += 1
                            if args.skip_empty_profile: continue

                    # pretrain
                    # in reply to account id, author id, use for recording conversation
                    # geo
                    # timestamp

                    # history
                    # geo TODO
                    # entities/context
                    # cat top 20, max is max seq len - user desc and news
                    # topic as retrieval
                    if str(user_id) not in history_data:
                        history_text = ""
                        # continue
                    else:
                        history_posts = [item["text"] for item in history_data[str(user_id)][:5]]
                        history_text = ";".join(history_posts)

                    if not user_desc and not history_text:
                        cnt_empty_user_attributes += 1
                        continue

                    if "predicted" not in sample:
                        continue

                    # filter out inactive users
                    post_text = post_data[str(post_id)]['text']
                    post_text = preprocess_tweet_local(post_text)

                    tgt_text = sample["text"]
                    # p.set_options(p.OPT.URL)
                    # # skip the ones with url
                    # if p.clean(tgt_text)!=tgt_text:
                    #     continue
                    if find_URLS(tgt_text):
                        continue
                    tgt_text = p.clean(tgt_text)

                    src_text = f"{post_text} [POST] {user_desc} [PROFILE] {history_text}"  # {tokenizer.sep_token}
                    if args.is_labeling:

                        # # tgt_text
                        # if args.config == "label_sent":
                        #     tgt_text = preprocess_tweet_local(tgt_text)
                        #     if not tgt_text: continue

                        tmp.append({"text": tgt_text, })
                    else:
                        tmp.append({"text": src_text, "label": sample["predicted"], })
                        if args.use_intensity_for_sentiment:
                            if sample["predicted"] == 3:
                                tmp[-1]['label'] = 1
                            else:
                                tmp[-1]['label'] = 0 if int(sample["predicted"]) < 3 else 2

                    tmp[-1]['sample_id'] = i
                    tmp[-1]['orig_comment'] = tgt_text
                print("num_empty_desc", num_empty_desc)
                print("cnt_empty_user_attributes", cnt_empty_user_attributes / len(self.original_data))
            elif args.sent_clf:
                for i, sample in enumerate(tqdm(self.original_data)):
                    tgt_text = sample["text"]
                    if args.is_labeling:
                        # tgt_text
                        tgt_text = preprocess_tweet_local(tgt_text)
                        if not tgt_text: continue
                        tmp.append({"text": tgt_text})
                        tmp[-1]['sample_id'] = i

            self.original_data = tmp
            print("total", len(self.original_data))

        else:
            raw_data = retrieve_raw_tweets_from_db()
            # self.original_data = [{'tweet_text':item["content_text"],"labels":["care"],'tweet_id':item['uiuc_message_id'] } for item in raw_data]
            self.original_data = [{'tweet_text': item.content_text, "labels": ["care"], 'tweet_id': item.uiuc_message_id} for item in raw_data]
            # breakpoint()

        self.instances = []
        # print("in_train and args.augment_with_translation", in_train and args.augment_with_translation)
        # print("(not in_train and do_foreign_eval)", (not in_train and args.do_foreign_eval))

        class_labels = []
        for idx, sample in enumerate(tqdm(self.original_data)):
            # if "non-moral" in sample["labels"]:
            #     continue
            text = sample["text"]
            # if 'doc_pos' in sample:
            #     post_text=self.original_data[sample["doc_pos"]]["text"]
            #     text = f"'{post_text}'. {text}"

            """can modify to include number"""

            # text = normalizeTweet(text)
            if not args.is_labeling:
                label_vec = fill_array_at_positions(length=len(self.labels),
                                                    positions=[self.label2id[label] for label in sample["label"]]) if isinstance(sample["label"], list) else self.label2id[
                    sample["label"]]  # label_map[label]
            else:
                label_vec = 0
                # fill_array_at_positions(length=len(self.labels), positions=[]) if isinstance(sample["label"], list) else
            # print("lbs", sample["labels"])
            # print("label_vec", label_vec)
            self.instances.append({"text": text,
                                   "id": idx,
                                   "sample_id": sample["sample_id"],
                                   # "tweet_id": sample["tweet_id"] if "tweet_id" in sample else None,
                                   "labels": label_vec,
                                   'orig_comment': sample["orig_comment"] if "orig_comment" in sample else "",
                                   "input_ids": tokenizer.tokenize(text),
                                   "in_train": in_train
                                   })
            # if in_train and args.augment_with_translation:
            #     text = sample["tweet_text_fr"]
            #     self.instances.append({"text": text,
            #                            "id": idx,
            #                            "tweet_id": sample["tweet_id"],
            #                            "labels": label_vec,
            #                            "input_ids": tokenizer.tokenize(text),
            #                            })
        # if in_train:
        #     class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=np.array(class_labels))
        #     self.class_weights = torch.tensor(class_weights, dtype=torch.float)

        if args.cache_filename:
            dump_file(self.instances, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

    def get_class_weights(self):
        class_labels = [sample["labels"] for sample in self.instances]
        if np.unique(class_labels).shape[0] < len(self.labels):
            print("\n\n\nclass_weights dim smaller than label dim")
            class_labels = np.arange(len(self.labels))
        class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=np.array(class_labels))
        return torch.tensor(class_weights, dtype=torch.float)

    @classmethod
    def collect_labels(cls, files, path):
        if path_exists(path):
            return load_file(path)

        labels = set()
        for filename in files:
            data = load_file(filename)
            for idx, sample in enumerate(data):
                for m in sample["annotations"]:  # todo
                    labels |= m["labels"]
        labels = sorted(labels)
        dump_file(labels, path)

        return labels


def find_URLS(string):
    # findall() has been used
    # with valid conditions for urls in string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return [x[0] for x in url]


class PrimitiveGenerationDataset(Dataset):

    def __init__(self, args, src_file, tokenizer=None, in_train=False, extra_data=None):

        super().__init__()
        """========Init========="""

        self.tokenizer = tokenizer
        self.instances = []
        if not src_file.strip():
            return

        # Special Tokens
        self.SEP = tokenizer.sep_token_id
        self.CLS = tokenizer.cls_token_id
        self.BOS = tokenizer.bos_token_id
        self.EOS = tokenizer.eos_token_id
        self.SEP_TOKEN = tokenizer.sep_token
        self.CLS_TOKEN = tokenizer.cls_token
        self.BOS_TOKEN = tokenizer.bos_token
        self.EOS_TOKEN = tokenizer.eos_token

        if self.SEP is None:
            self.SEP = self.EOS
            self.SEP_TOKEN = self.EOS_TOKEN

        self.max_seq_len = args.max_seq_len
        # print("self.max_seq_len", self.max_seq_len)

        # is_t5 = "t5" in args.plm
        is_gpt = "gpt" in args.plm.lower()

        """========Load Cache========="""
        args.cache_filename = os.path.splitext(src_file)[0] + "_" + args.plm_class + "_" + args.data_mode + \
                              (f"_gprimitive") + \
                              ("_prp" if args.personal_response_pred else "") + \
                              ("_lb" if args.is_labeling else "") + \
                              ("_sc" if args.sent_clf else "") + \
                              ("_po" if args.pred_only else "") + \
                              ("_int4sent" if args.use_intensity_for_sentiment else "") + \
                              ("_skipemprof" if args.skip_empty_profile else "") + \
                              ".pkl"  #
        save_file = args.cache_filename
        print('\nReading data from {}.'.format(src_file))
        if os.path.exists(save_file) and args.use_cache:
            self.instances = load_file(save_file)
            print('load processed data from {}.'.format(save_file))
            return

        data_samples = load_file(src_file)
        if ".csv" in src_file:
            json_str = data_samples.to_json(orient="records")
            data_samples = json.loads(json_str)
        if args.debug:
            data_samples = data_samples[:100]
        # p.set_options(p.OPT.MENTION)

        if args.comment_generation:
            # restructuring
            tmp = []
            for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
                sum_likes = sum([x["like"] for x in sample["direct_replies"]])
                num_distinct = len(set([x["like"] for x in sample["direct_replies"]]))
                if num_distinct == 0: continue

                cumulated_likes = 0
                cumulated_likes_to_record = 0
                prev_recorded_num_likes = -1
                num_values_below = 0
                sorted_by_likes = sorted(sample["direct_replies"], key=lambda x: x["like"], reverse=False)
                for j, reply in enumerate(sample["direct_replies"] if args.min_num_likes == -1 else sorted_by_likes):
                    if reply["like"] > prev_recorded_num_likes:
                        prev_recorded_num_likes = reply["like"]
                        num_values_below = j
                        cumulated_likes_to_record = cumulated_likes + reply["like"]
                    cumulated_likes += reply["like"]
                    if args.min_num_likes == -1 or reply["like"] >= args.min_num_likes:
                        percentile = str(round(num_values_below / len(sample["direct_replies"]) * 100, 0)) if num_distinct > 0 else 0
                        percentile_str = ". [" + percentile + "th percentile]"
                        src_text = sample["text"] if args.min_num_likes == -1 else sample["text"] + percentile_str
                        # if reply["like"] >0:
                        #     breakpoint()
                        tmp.append({"src_text": src_text,
                                    "tgt_text": reply["text"],
                                    "tweet_id": sample["tweet_id"],
                                    "post_tweet_id": sample["tweet_id"],
                                    "reply_tweet_id": reply["tweet_id"],
                                    "num_likes": reply["like"]
                                    })
            data_samples = tmp
        elif args.label_generation:
            print("label generation")
            tmp = []
            for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
                src_text = sample["text"]
                sorted_keys = sort_key_by_value(sample["response_labels"][args.label_category], reverse=True)
                tmp.append({"src_text": src_text,
                            "tgt_text": " ".join(sorted_keys),
                            "tweet_id": sample["tweet_id"],
                            "post_tweet_id": sample["tweet_id"],
                            })
            data_samples = tmp
        elif args.personal_response_pred:
            # user_data, post_data, graph_data = extra_data
            print("personal_response_pred")
            tmp = process_samples(data_samples, args, extra_data=extra_data)
            data_samples = tmp

        print("restructured")
        maxlens = 0

        for i, sample in enumerate(tqdm(data_samples)):  # this indicates ith path after breaking out all articles into individual paths
            if "src_text" not in sample:
                src_text, tgt_text = sample["text"], None
            else:
                src_text, tgt_text = sample["src_text"], sample["tgt_text"]
            if len(tgt_text.split()) > 50: continue

            # for gpt
            tmp_max_seq_len = self.max_seq_len - 2
            tmp_max_seq_len_tgt = 90
            tmp_max_seq_len_src = tmp_max_seq_len - tmp_max_seq_len_tgt

            model_inputs = self.tokenizer(src_text, padding=True, max_length=tmp_max_seq_len_src, truncation=True)
            # self.max_seq_len if not is_gpt else tmp_max_seq_len_src

            if tgt_text is not None:
                with self.tokenizer.as_target_tokenizer():
                    # tgt_text = tgt_text.lower()
                    tgt = self.tokenizer(tgt_text, padding=True, max_length=tmp_max_seq_len_tgt, truncation=True)
                    # self.max_seq_len if not is_gpt else

                    # if tgt["input_ids"].count(self.tokenizer.encode('[label]')[0])!=2:
                    #     breakpoint()

                if is_gpt:
                    input_ids, attention_mask = model_inputs["input_ids"], model_inputs["attention_mask"]
                    if in_train:
                        model_inputs["labels"] = [-100] * len(input_ids) + tgt['input_ids']  # tgt['input_ids']
                        model_inputs["labels"] = model_inputs["labels"][:tmp_max_seq_len] + [tokenizer.eos_token_id]
                        model_inputs["input_ids"] += tgt['input_ids']  # tgt['input_ids']
                        model_inputs["attention_mask"] += tgt['attention_mask']  # tgt['input_ids']
                        # model_inputs["labels"] = model_inputs["labels"][:tmp_max_seq_len] + [tokenizer.eos_token_id]
                        model_inputs["input_ids"] = model_inputs["input_ids"][:tmp_max_seq_len] + [tokenizer.eos_token_id]
                        model_inputs["attention_mask"] = model_inputs["attention_mask"][:tmp_max_seq_len] + [1]
                    else:
                        model_inputs["labels"] = [-100] * len(input_ids)
                        model_inputs["labels"] = model_inputs["labels"][:tmp_max_seq_len_src]
                        model_inputs["input_ids"] = model_inputs["input_ids"][:tmp_max_seq_len_src]
                        model_inputs["attention_mask"] = model_inputs["attention_mask"][:tmp_max_seq_len_src]
                        # tgt_text=self.tokenizer.decode(tgt['input_ids']).replace("  "," ")

                else:
                    model_inputs["labels"] = tgt['input_ids']
                    # for key in ['input_ids', 'attention_mask']:
                    #     model_inputs["decoder_" + key] = tgt[key]
                    # model_inputs["decoder_" + "token_type_ids"] = [0] * len(tgt["input_ids"])

            # # don't do over long inputs
            # exceed_max_len = False
            # maxlens = max(len(tgt['input_ids']), maxlens)
            # if max(len(tgt['input_ids']), len(model_inputs['input_ids'])) >= self.max_seq_len - 2:
            #     exceed_max_len = True

            self.instances.append({
                'tokenizer': tokenizer,
                'src_text': src_text,
                'tgt_text': tgt_text,
                "sample_id": i,
                "extra": sample["extra"],
                "sample_id_in_orig_data": sample["sample_id"],
                'exceed_max_len': -1,  # exceed_max_len,
                "in_train": in_train
            })
            self.instances[-1].update(model_inputs)

        """encode different parts into functions"""
        # save data
        print("maxlens", maxlens)
        if args.cache_filename:
            dump_file(self.instances, save_file)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance


###


def process_samples(original_data, args, extra_data=None):
    tmp = []
    if args.personal_response_pred:
        user_data, post_data, history_data, graph_data = extra_data

        print("personal_response_pred")

        cnt_empty_user_attributes = 0
        num_empty_desc = 0

        custom_posts = [
            "I am a student",
        ]
        for i, sample in enumerate(tqdm(original_data)):  # this indicates ith path after breaking out all articles into individual paths
            user_id = sample["author_id"]
            post_id = sample["conversation_id"]

            """user_desc"""
            if str(user_id) not in user_data:
                user_desc = ""
                # continue
            else:
                user_desc = user_data[str(user_id)]['description']

            if not user_desc.strip():
                num_empty_desc += 1
                if args.skip_empty_profile: continue

            # pretrain
            # in reply to account id, author id, use for recording conversation
            # geo
            # timestamp

            "only use nonreply"

            # history
            # geo TODO
            # entities/context
            # cat top 20, max is max seq len - user desc and news
            # topic as retrieval

            """history"""
            if str(user_id) not in history_data:
                history_text = ""
                # continue
            else:
                history_posts = [item["text"] for item in history_data[str(user_id)][:50] if int(item["tweet_id"]) != int(sample["tweet_id"])]
                history_text = ";".join(history_posts)

            """check"""
            if not user_desc and not history_text:
                cnt_empty_user_attributes += 1
                continue

            if "predicted" not in sample or "predicted_intensity" not in sample:
                continue

            # filter out inactive users
            post_text = post_data[str(post_id)]['text']
            post_text = preprocess_tweet_local(post_text)

            tgt_text = sample["text"]
            # p.set_options(p.OPT.URL)
            # # skip the ones with url
            # if p.clean(tgt_text)!=tgt_text:
            #     continue
            if find_URLS(tgt_text):
                continue
            tgt_text = p.clean(tgt_text)
            if not tgt_text.strip():
                continue

            uid_str = str(user_id)
            if "1" not in args.user_attributes:
                user_desc = " "
            if "2" not in args.user_attributes:
                history_text = ""
            if "3" not in args.user_attributes:
                uid_str = " "

            src_text = f"{post_text} [POST] {user_desc} [PROFILE] {history_text}"  # {tokenizer.sep_token}[LABEL_SEP] {uid_str} [UID]

            if args.task_mode == "clf":
                if args.is_labeling:
                    tmp.append({"text": tgt_text, })
                else:
                    tmp.append({"text": src_text, "label": sample["predicted"], })
                    if args.use_intensity_for_sentiment:
                        if sample["predicted"] == 3:
                            tmp[-1]['label'] = 1
                        else:
                            tmp[-1]['label'] = 0 if int(sample["predicted"]) < 3 else 2
                tmp[-1]['orig_comment'] = tgt_text
            elif args.task_mode == "gen":
                if args.pred_only:
                    tmp_tgt_text = f" {sample['predicted']} [label] {sample['predicted_intensity']} [label]  "
                elif args.text_only:
                    tmp_tgt_text = "EMPTY" if not tgt_text.strip() else tgt_text
                else:
                    tmp_tgt_text = f" {sample['predicted']} [label] {sample['predicted_intensity']} [label] {tgt_text}"

                tmp.append({"src_text": src_text,
                            "tgt_text": tmp_tgt_text,
                            # "extra": (sample["predicted"], sample["predicted_intensity"], user_id, post_id)
                            "extra": {
                                "predicted": sample["predicted"],
                                "predicted_intensity": sample["predicted_intensity"],
                                "category": user_data[str(user_id)]["category"] if str(user_id) in user_data and "category" in user_data[str(user_id)] else " ",
                                "user_id": user_id,
                                "post_id": post_id
                            }
                            })
            tmp[-1]['sample_id'] = i
        print("num_empty_desc", num_empty_desc)
        print("cnt_empty_user_attributes", cnt_empty_user_attributes / len(original_data))
    return tmp
