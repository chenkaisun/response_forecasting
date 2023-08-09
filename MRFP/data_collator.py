import numpy as np
from torch.utils.data import Dataset

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
from collections import defaultdict
from dataclasses import dataclass

from typing import Any, List, Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers import PreTrainedTokenizerBase
@dataclass
class CustomCollatorCLF(DataCollatorWithPadding):
    collect_input: Optional[Any] = False
    collect_fields: Optional[List] = None

    def __call__(self, features):
        input_features = {}
        input_features.update(self.tokenizer([f["text"] for f in features], truncation=True,
                                             max_length=self.max_length, return_tensors='pt', padding=True))
        input_features["labels"] = get_tensor_long([f["labels"] for f in features])
        # input_features["ids"] = [f["id"] for f in features]  # get_tensor_long([f["id"] for f in features])
        # self.tokenizer.decode(input_features["input_ids"][0])
        # input_features["tweet_ids"] = [f["tweet_id"] for f in features]  # get_tensor_long([f["id"] for f in features])
        # input_features["input_ids"] = [f["input_ids"] for f in features]
        ## The tweet id here means the id in the dataset fr evall, not actual tweet id

        # print("input_features",input_features)
        # input_features.pop("token_type_ids")
        # print("self.max_length", self.max_length)

        # print("input_features", input_features)
        return input_features

@dataclass
class CustomCollatorPrimitive():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # features=deepcopy(feats)

        # encoder_features = [{'input_ids': feat['input_ids'],
        #                      'attention_mask': feat['attention_mask'],
        #                      } for feat in features]
        # input_features = self.tokenizer.pad(
        #     encoder_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )
        input_features={}
        key_name_for_text='text' if 'text' in features[0] else 'src_text'

        input_features.update(self.tokenizer([f[key_name_for_text] for f in features], truncation=True,
                                             max_length=self.max_length, return_tensors='pt', padding=True))

        # category_encoder_features = [{'input_ids': feat['category_input_ids'],
        #                               'attention_mask': feat['category_attention_mask'],
        #                               } for feat in features]
        # input_features.update(modify_dict_keys(self.tokenizer.pad(
        #     category_encoder_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # ), prefix="category_"))

        input_features.update({
            "texts": [feat[key_name_for_text] for feat in features],
            'sample_ids': [feat['sample_id'] for feat in features]
        })

        return input_features


@dataclass
class CustomCollatorCLM():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    verbose: Optional[bool] = False
    args: Optional[Any] = None
    mask_lens_dev: Optional[Any] = None


    def __call__(self, features):
        sample_ids = [feature["sample_id"] for feature in features]
        if self.verbose:
            print("sample id", sample_ids)

        """labels"""
        labels = [feature["labels"].copy() for feature in features] if "labels" in features[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)  # max len of decoder input ids
            padding_side = self.tokenizer.padding_side
            for j, lb in enumerate(labels):
                remainder = [self.label_pad_token_id] * (max_label_length - len(lb))
                labels[j] = (
                    lb + remainder if padding_side == "right" else remainder + lb
                )

        """encoder features"""
        encoder_features = [{'input_ids': feat['input_ids'],
                             'attention_mask': feat['attention_mask'],
                             } for feat in features]
        input_features = {}
        input_features.update(self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ))
        if "labels" in features[0]:
            input_features['labels'] = get_tensor_long(labels)

        bsz, slen = input_features['input_ids'].shape
        if not features[0]["in_train"] and "gpt" in self.args.plm.lower():
            self.mask_lens_dev.extend([slen]*bsz)

        return input_features




@dataclass
class CustomCollator():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    verbose: Optional[bool] = False
    use_special_tag: Optional[bool] = False
    args: Optional[Any] = None
    def __call__(self, features):
        sample_ids = [feature["sample_id"] for feature in features]
        if self.verbose:
            print("sample id", sample_ids)
        # print("sample id", sample_ids)
        """labels"""
        if "clf_label" not in features[0]:
            labels = [feature["labels"].copy() for feature in features] if "labels" in features[0].keys() else None
            if labels is not None:
                max_label_length = max(len(l) for l in labels)  # max len of decoder input ids
                if "t5-" not in self.args.plm:
                    for j, lb in enumerate(labels):
                        labels[j] = lb[1:]
                padding_side = self.tokenizer.padding_side
                # print("self.label_pad_token_id",self.label_pad_token_id)
                for j, lb in enumerate(labels):
                    remainder = [self.label_pad_token_id] * (max_label_length - len(lb))
                    labels[j] = (
                        lb + remainder if padding_side == "right" else remainder + lb
                    )


        """encoder features"""
        encoder_features = [{'input_ids': feat['input_ids'],
                             'attention_mask': feat['attention_mask'],'labels': labels[j]
                             } for j, feat in enumerate(features)]

        input_features = {}
        """agg features"""
        input_features.update(self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ))

        # """CLF or not decode"""
        # decoder_features = [{'input_ids': feat['decoder_input_ids'],
        #                      'attention_mask': feat['decoder_attention_mask'],
        #                      'labels': labels[j],
        #                      } for j, feat in enumerate(features)]
        # decoder_features = self.tokenizer.pad(
        #     decoder_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )
        # print("input_features[labels]", input_features['labels'])

        # input_features['decoder_input_ids'] = decoder_features['input_ids']
        # input_features['decoder_attention_mask'] = decoder_features['attention_mask']

        # input_features['labels'] = decoder_features['labels']


        # not used right now

        bsz, slen = input_features['input_ids'].shape

        if self.verbose:
            if -1 in sample_ids:
                breakpoint()
        # """=============batch entities============="""
        # if "t5-" in self.args.plm:
        #     input_features.pop('decoder_input_ids')
        #     input_features.pop('decoder_attention_mask')
        # # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=input_features["labels"])
            input_features["decoder_input_ids"] = decoder_input_ids

        return input_features








@dataclass
class CustomCollatorRET():
    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    has_concepts: bool = False

    def __call__(self, features):
        # features=deepcopy(feats)

        encoder_features = [{'input_ids': feat['input_ids'],
                             'attention_mask': feat['attention_mask'],
                             } for feat in features]
        input_features = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        category_encoder_features = [{'input_ids': feat['category_input_ids'],
                                      'attention_mask': feat['category_attention_mask'],
                                      } for feat in features]
        input_features.update(modify_dict_keys(self.tokenizer.pad(
            category_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ), prefix="category_"))

        title_encoder_features = [{'input_ids': feat['title_input_ids'],
                                   'attention_mask': feat['title_attention_mask'],
                                   } for feat in features]
        input_features.update(modify_dict_keys(self.tokenizer.pad(
            title_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ), prefix="title_"))

        title2_encoder_features = [{'input_ids': feat['title2_input_ids'],
                                    'attention_mask': feat['title2_attention_mask'],
                                    } for feat in features]
        input_features.update(modify_dict_keys(self.tokenizer.pad(
            title2_encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        ), prefix="title2_"))

        input_features.update({
            'sample_ids': [feat['sample_id'] for feat in features]
        })

        return input_features
