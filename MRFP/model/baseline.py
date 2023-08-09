from pprint import pprint as pp
import pickle as pkl
from collections import deque, defaultdict, Counter
import json
from IPython import embed
import random
import time
import os
import numpy as np
import argparse
import gc
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from transformers import EncoderDecoderModel, BertModel

class BaselineEncoder(BertModel):

    def forward(self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                g_data=None,
                g_data2=None,
                token2nodepos=None):
        return super(BaselineEncoder, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_data=g_data,
                g_data2=g_data2,
                token2nodepos=token2nodepos )
