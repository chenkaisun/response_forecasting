# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BART model. """
import gc
import math
import random
import warnings
from typing import Optional, Tuple
from IPython import embed
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from model.activations import ACT2FN
from model.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from model.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputCustom,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from model.modeling_utils import PreTrainedModel
from ..utils import logging
from .configuration_bart import BartConfig
from ..cse import EventReasoningModule
from ..model_utils import get_tensor_info
from ..deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from model.gnn import GNN


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/bart-large"
_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # print("SEEBartLearnedPositionalEmbedding fwd")
        # print("past_key_values_length", past_key_values_length)
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)

    def forward_event_tag(self, positions=None):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        return super().forward(positions + self.offset)


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            # print('key_value_states',key_value_states)
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # print('key_value_states',key_value_states)
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class SEEBartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.do_attend_to_nbs=False
        self.nb_decoder_attention_heads=config.decoder_attention_heads
        self.nb_attention_dropout=config.attention_dropout
        # self.encoder_attn = SEEBartAttention(
        #     self.embed_dim,
        #     config.decoder_attention_heads,
        #     dropout=config.attention_dropout,
        #     is_decoder=True,
        # )
        self.attend_to_nbs=False



        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    def add_modules(self, args):
        # print("self.embed_dim",self.embed_dim)
        self.fc_nbs = nn.Linear((args.num_nbs+1)*(self.embed_dim), self.embed_dim)
        # self.nb_attn = BartAttention(
        #     self.embed_dim,
        #     self.nb_decoder_attention_heads,
        #     dropout=self.nb_attention_dropout,
        #     is_decoder=True,
        # )
        self.attend_to_nbs=True
        ##* donwstream, so init differently

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        nbs_hidden_states_list: Optional[torch.Tensor] = None,
        nbs_attention_mask_list: Optional[torch.Tensor] = None,
            nbs_msk=None,

    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None

            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            if self.attend_to_nbs:

                tmp_attn_list=[]
                for nb_i in range(len(nbs_attention_mask_list)):
                    # nb_attn(
                    attended_hidden_states, _, _ = self.encoder_attn(
                        hidden_states=residual,
                        key_value_states=nbs_hidden_states_list[nb_i],
                        attention_mask=nbs_attention_mask_list[nb_i],
                        layer_head_mask=None,
                        past_key_value=None,
                        output_attentions=False,
                    )
                    tmp_attn_list.append(attended_hidden_states)
                    attended_hidden_states=None

                hidden_states=torch.cat([hidden_states]+tmp_attn_list, dim=-1)
                tmp_attn_list=None

                hidden_states=self.fc_nbs(hidden_states)


            """nb attn"""
            # hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            #     hidden_states=hidden_states,
            #     key_value_states=encoder_hidden_states,
            #     attention_mask=encoder_attention_mask,
            #     layer_head_mask=cross_attn_layer_head_mask,
            #     past_key_value=cross_attn_past_key_value,
            #     output_attentions=output_attentions,
            # )


            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (:obj:`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )


BART_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are decoder input IDs? <../glossary.html#decoder-input-ids>`__

            Bart uses the :obj:`eos_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            For translation and summarization training, :obj:`decoder_input_ids` should be provided. If no
            :obj:`decoder_input_ids` is provided, the model will create this tensor by shifting the :obj:`input_ids` to
            the right for denoising pre-training following the paper.
        decoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
            Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`:
            :obj:`attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
            `optional`) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the
            cross-attention of the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
            shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, args=None, pretrained_concept_emb=None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # print("self.max_source_positions",self.max_source_positions)

        # self.evt_reasoner=EventReasoningModule()

        self.init_weights()
    def add_modules(self, args, pretrained_concept_emb):
        self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
        ##* donwstream, so init differently


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        # cur_modelname=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        ##*
        hidden_states = inputs_embeds + embed_pos

        # print("Encoder self.submodule_1.components", self.submodule_1.components)
        # if "evttag" in self.submodule_1.components: hidden_states+=self.embed_positions.forward_event_tag(event_position_ids)

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # # print("hidden_states", get_tensor_info(hidden_states))
        # if "submodule_1" in dir(self):
        #     hidden_states = self.submodule_1(hidden_states,
        #                                      g_data,
        #                                      g_data2,
        #                                      token2nodepos)  # (batch size, dim_mode)

        # print("\nhidden_states", hidden_states)
        # print("return_dict", return_dict)


        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)


        # print("encoder_states", encoder_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
class BartEncoderForClassification(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None, args=None, pretrained_concept_emb=None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)


        # print("self.max_source_positions",self.max_source_positions)

        # self.evt_reasoner=EventReasoningModule()

        self.init_weights()
    def add_modules(self, args, pretrained_concept_emb):
        self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
        self.combiner = nn.Linear(args.plm_hidden_dim, args.out_dim)
        ##* donwstream, so init differently


    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        # print("new resized embeddings")
        if new_num_tokens is None:
            return old_embeddings

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim, padding_idx=old_embeddings.padding_idx).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        # cur_modelname=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        ##*

        # print("in Encoder")
        # embed()
        # g_data = g_data.to(attention_mask.device)
        # g_data2 = g_data2.to(attention_mask.device)
        if g_data is not None:
            g_data = g_data.to(attention_mask)
            g_data2 = g_data2.to(attention_mask)
            token2nodepos = token2nodepos.to(attention_mask)  # .cuda()#.to(attention_mask)
            event_position_ids = event_position_ids.to(attention_mask)  # .cuda()#.to(attention_mask)

        if aggregate_ipids is not None:
            # print("aggregate_input_ids", aggregate_input_ids.device)

            # aggregate_input_ids = aggregate_input_ids.to(attention_mask)
            # aggregate_attention_mask = aggregate_attention_mask.to(attention_mask)
            input_ids=aggregate_ipids
            attention_mask=aggregate_atmsk

        # else:
        #     print("aggregate_ipids is none")

        # print("input_ids.shape",input_ids.shape)
        # try:
        #     # g_data = g_data.cuda()
        #     # g_data2 = g_data2.cuda()
        #     # g_data = g_data.to(self.device)
        #     # g_data2 = g_data2.to(self.device)
        #     g_data = g_data.to(attention_mask)
        #     g_data2 = g_data2.to(attention_mask)
        #     token2nodepos = token2nodepos.to(attention_mask)#.cuda()#.to(attention_mask)
        #     event_position_ids = event_position_ids.to(attention_mask)#.cuda()#.to(attention_mask)
        #     # pass
        #
        # except:
        #     print("in Encoder")
        #
        #     embed()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)
        ##*
        hidden_states = inputs_embeds + embed_pos

        # print("Encoder self.submodule_1.components", self.submodule_1.components)
        # if "evttag" in self.submodule_1.components: hidden_states+=self.embed_positions.forward_event_tag(event_position_ids)

        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # print("hidden_states", get_tensor_info(hidden_states))
        if "submodule_1" in dir(self):
            hidden_states = self.submodule_1(hidden_states,
                                             g_data,
                                             g_data2,
                                             token2nodepos)  # (batch size, dim_mode)
        if "combiner" in dir(self):
            output = self.combiner(torch.cat(hidden_states, dim=-1))
        # print("output", get_tensor_info(output))

        # print("output", output)
        return {
            "logits": output
        }
        # print("\nhidden_states", hidden_states)
        # print("return_dict", return_dict)


        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)


        # print("encoder_states", encoder_states)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        # print("in BartDecoder")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:


                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
class _SEEBartEncoder(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        # self.dropout = config.dropout
        # self.layerdrop = config.encoder_layerdrop
        #
        # embed_dim = config.d_model
        # self.padding_idx = config.pad_token_id
        # self.max_source_positions = config.max_position_embeddings
        # self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        #
        # if embed_tokens is not None:
        #     self.embed_tokens = embed_tokens
        # else:
        #     self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        #
        # self.embed_positions = BartLearnedPositionalEmbedding(
        #     config.max_position_embeddings,
        #     embed_dim,
        # )
        # self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.encoder = BartEncoder(config, embed_tokens) #, args=args, pretrained_concept_emb=pretrained_concept_emb

        # print("self.max_source_positions",self.max_source_positions)

        # self.evt_reasoner=EventReasoningModule()

        # self.init_weights()
    # def add_modules(self, args, pretrained_concept_emb):
    #     self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
    #     ##* donwstream, so init differently

    def encode_gather(self, text_embeddings, piece_idxs , idxs, masks, token_num, token_len):

        idxs=idxs.to(piece_idxs)
        masks=masks.to(text_embeddings)
        batch_size = text_embeddings.shape[0]

        # idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, text_embeddings.shape[-1]) # + 1
        idxs=idxs.unsqueeze(-1).expand(batch_size, -1, text_embeddings.shape[-1])

        # masks = text_embeddings.new(masks).unsqueeze(-1)
        masks =masks.unsqueeze(-1)

        text_embeddings = torch.gather(text_embeddings, 1, idxs) * masks
        text_embeddings = text_embeddings.view(batch_size, token_num, token_len,text_embeddings.shape[-1])
        text_embeddings = text_embeddings.sum(2)  # max_seq_len in words
        return text_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        token_lens=None,
        nbs_ipids=None,
        nbs_atmsk=None,
        token2nodeid=None,
        nbs_g_data=None,
        nbs_g_data_extra=None,
        token2rootnodeid=None,
        root_indices=None,
        root_masks=None,
        nbs_root_indices=None,
        nbs_root_masks=None,
        nbs_msk=None
    ):

        nbs_hidden_states_list = []
        nbs_attention_mask_list = []

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_data=g_data,
            g_data2=g_data2,
            token2nodepos=token2nodepos,
            event_position_ids=event_position_ids,
            aggregate_ipids=aggregate_ipids,
            aggregate_atmsk=aggregate_atmsk,
        )

        text_embeddings = encoder_outputs[0]
        # encoder_outputs[0]=None # to remove ref
        encoder_outputs['last_hidden_state'] = None  # to remove ref

        """====Encode Original Graph===="""
        """init graph embeds from token embeds"""

        graph_embeddings = text_embeddings

        print('mdl fwd')

        if root_indices is not None:
            print('root_indices is not None')

            tmp = []
            for i in range(graph_embeddings.shape[0]):
                tmp.append(torch.index_select(graph_embeddings[i], 0, root_indices[i]) * root_masks[i].unsqueeze(1))
            graph_embeddings = torch.stack(tmp, dim=0)
            tmp = None

        if g_data is not None:
            print('g_data is not None')

            graph_embeddings = self.encode_gather(text_embeddings, input_ids, g_data.idxs,
                                                  g_data.masks, g_data.max_token_num[0].item(), g_data.max_token_len[0].item())

            """gnn conv"""
            g_data.x = torch.tensor([])
            g_data.idxs = torch.tensor([])
            g_data.masks = torch.tensor([])
            g_data.token_num = torch.tensor([])
            g_data.token_len = torch.tensor([])  # these are not useful anymore
            g_data = g_data.to(attention_mask)

            max_node_num_batch = graph_embeddings.shape[1]
            # print("max_node_num_batch",max_node_num_batch)

            graph_embeddings = graph_embeddings.view(-1, graph_embeddings.shape[-1])
            graph_embeddings = self.gnn(graph_embeddings, g_data)
            g_data = None  # clr ref

            graph_embeddings = graph_embeddings.view(text_embeddings.shape[0], max_node_num_batch, graph_embeddings.shape[-1])

        # if max_node_num_batch > 3:
        #     graph_embeddings = graph_embeddings[:, 3:, :]
        # print("graph_embeddings1",get_tensor_info(graph_embeddings))
        # g_data=None # to remove ref

        """matching nbs"""
        if self.use_nbs and nbs_ipids is not None:

            summ = None
            for i in range(nbs_ipids.shape[0]):
                # print("nb",i)

                """====Encode Neighbor Text===="""
                # nb_features=text_embeddings

                nb_features = self.encoder(
                    input_ids=nbs_ipids[i],
                    attention_mask=nbs_atmsk[i],
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict)[0]
                # print("nb_features",get_tensor_info(nb_features))

                """====Encode Neighbor Graph===="""
                nb_graph_embeddings = nb_features

                if nbs_root_indices is not None:
                    print('nbs_root_indices is not None', nbs_root_indices is not None)
                    tmp = []
                    nb_root_indices = nbs_root_indices[i].to(root_indices)
                    nb_root_masks = nbs_root_masks[i].to(root_masks)
                    for j in range(nb_graph_embeddings.shape[0]):
                        tmp.append(torch.index_select(nb_graph_embeddings[j], 0, nb_root_indices[j]) * nb_root_masks[j].unsqueeze(1))
                    nb_graph_embeddings = torch.stack(tmp, dim=0)
                    tmp = None
                if g_data is not None:
                    nb_g_data = nbs_g_data[i]
                    nb_graph_embeddings = self.encode_gather(nb_features, nbs_ipids[i], nb_g_data.idxs,
                                                             nb_g_data.masks, nb_g_data.max_token_num[0].item(), nb_g_data.max_token_len[0].item())
                    nb_features = None  # clear reference, no need neighbor text information

                    """gnn"""
                    nb_g_data.x = torch.tensor([])
                    nb_g_data.idxs = torch.tensor([])
                    nb_g_data.masks = torch.tensor([])
                    nb_g_data.token_num = torch.tensor([])
                    nb_g_data.token_len = torch.tensor([])  # these are not useful anymore
                    nb_g_data = nb_g_data.to(attention_mask)  # now only has edge

                    max_node_num_batch = nb_graph_embeddings.shape[1]
                    nb_graph_embeddings = nb_graph_embeddings.view(-1, nb_graph_embeddings.shape[-1])
                    nb_graph_embeddings = self.gnn(nb_graph_embeddings, nb_g_data)
                    # print("nb_graph_embeddings1",get_tensor_info(nb_graph_embeddings))
                    nbs_g_data[i] = None  # clear reference
                    # nb_g_data=None# clear reference

                    nb_graph_embeddings = nb_graph_embeddings.view(text_embeddings.shape[0], max_node_num_batch,
                                                                   nb_graph_embeddings.shape[-1])
                """====MultiHead Attention from Original===="""

                # if max_node_num_batch>3:
                #     nb_graph_embeddings=nb_graph_embeddings[:,3:,:]

                attended_emb = self.submodule_1(text_embeddings=text_embeddings,
                                                graph_embeddings=graph_embeddings,
                                                nb_graph_embeddings=nb_graph_embeddings,
                                                token2nodeid=token2nodeid,
                                                token2rootnodeid=token2rootnodeid,
                                                nbs_msk=nbs_msk[i])  # (batch size, dim_mode)
                if self.decattn:
                    nbs_hidden_states_list.append(nb_graph_embeddings)
                    nbs_attention_mask_list.append(nbs_atmsk[i])
                nb_graph_embeddings = None  # clear reference

                if summ is None:
                    summ = attended_emb
                else:
                    summ += attended_emb  # text_embeddings
            encoder_outputs['last_hidden_state'] = summ + text_embeddings
        else:
            print("no component")
            encoder_outputs['last_hidden_state'] = text_embeddings

        return BaseModelOutputCustom(
            last_hidden_state=encoder_outputs['last_hidden_state'],
            hidden_states=encoder_outputs['hidden_states'],
            attentions=encoder_outputs['attentions'],
            nbs_hidden_states_list=nbs_hidden_states_list,
        )

        # return encoder_outputs
class SEEBartEncoder(BartEncoder):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)

        # self.dropout = config.dropout
        # self.layerdrop = config.encoder_layerdrop
        #
        # embed_dim = config.d_model
        # self.padding_idx = config.pad_token_id
        # self.max_source_positions = config.max_position_embeddings
        # self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        #
        # if embed_tokens is not None:
        #     self.embed_tokens = embed_tokens
        # else:
        #     self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)
        #
        # self.embed_positions = BartLearnedPositionalEmbedding(
        #     config.max_position_embeddings,
        #     embed_dim,
        # )
        # self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        # self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # self.encoder = BartEncoder(config, embed_tokens) #, args=args, pretrained_concept_emb=pretrained_concept_emb

        # print("self.max_source_positions",self.max_source_positions)

        # self.evt_reasoner=EventReasoningModule()
        self.use_nbs=False
        self.decattn=False
        self.addemb=False
        self.gnn=None
        self.submodule_1=None
        print("enc init")

        # self.init_weights()
    # def add_modules(self, args, pretrained_concept_emb):
    #     self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
    #     ##* donwstream, so init differently

    def encode_gather(self, text_embeddings, piece_idxs , idxs, masks, token_num, token_len):

        idxs=idxs.to(piece_idxs)
        masks=masks.to(text_embeddings)
        batch_size = text_embeddings.shape[0]

        # idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, text_embeddings.shape[-1]) # + 1
        idxs=idxs.unsqueeze(-1).expand(batch_size, -1, text_embeddings.shape[-1])

        # masks = text_embeddings.new(masks).unsqueeze(-1)
        masks =masks.unsqueeze(-1)

        text_embeddings = torch.gather(text_embeddings, 1, idxs) * masks
        text_embeddings = text_embeddings.view(batch_size, token_num, token_len,text_embeddings.shape[-1])
        text_embeddings = text_embeddings.sum(2)  # max_seq_len in words
        return text_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        token_lens=None,
        nbs_ipids=None,
        nbs_atmsk=None,
        token2nodeid=None,
        nbs_g_data=None,
        nbs_g_data_extra=None,
        token2rootnodeid=None,
        root_indices=None,
        root_masks=None,
        nbs_root_indices=None,
        nbs_root_masks=None,
        nbs_msk=None,
        ent_lists=None,
        entities=None,
        triggers=None,
        text_graphs=None,
    ):

        nbs_hidden_states_list = []
        # nbs_attention_mask_list = []

        encoder_outputs = super(SEEBartEncoder, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_data=g_data,
            g_data2=g_data2,
            token2nodepos=token2nodepos,
            event_position_ids=event_position_ids,
            aggregate_ipids=aggregate_ipids,
            aggregate_atmsk=aggregate_atmsk,
        )

        text_embeddings = encoder_outputs[0]
        # encoder_outputs[0]=None # to remove ref
        encoder_outputs['last_hidden_state'] = None  # to remove ref

        """====Encode Original Graph===="""
        """init graph embeds from token embeds"""

        graph_embeddings = text_embeddings

        # print('mdl fwd')

        if root_indices is not None:
            print('root_indices is not None')

            tmp = []
            for i in range(graph_embeddings.shape[0]):
                # noinspection PyTypeChecker
                tmp.append(torch.index_select(graph_embeddings[i], 0, root_indices[i]) * root_masks[i].unsqueeze(1))
            graph_embeddings = torch.stack(tmp, dim=0)
            tmp = None

        if entities is None or g_data is not None or root_indices is not None:
            breakpoint()
        if g_data is not None:
            print('g_data is not None')
            breakpoint()

            graph_embeddings = self.encode_gather(text_embeddings, input_ids, g_data.idxs,
                                                  g_data.masks, g_data.max_token_num[0].item(), g_data.max_token_len[0].item())

            """gnn conv"""
            g_data.x = torch.tensor([])
            g_data.idxs = torch.tensor([])
            g_data.masks = torch.tensor([])
            g_data.token_num = torch.tensor([])
            g_data.token_len = torch.tensor([])  # these are not useful anymore
            g_data = g_data.to(attention_mask)

            max_node_num_batch = graph_embeddings.shape[1]
            # print("max_node_num_batch",max_node_num_batch)

            graph_embeddings = graph_embeddings.view(-1, graph_embeddings.shape[-1])
            graph_embeddings = self.gnn(graph_embeddings, g_data)
            g_data = None  # clr ref

            graph_embeddings = graph_embeddings.view(text_embeddings.shape[0], max_node_num_batch, graph_embeddings.shape[-1])

        # if max_node_num_batch > 3:
        #     graph_embeddings = graph_embeddings[:, 3:, :]
        # print("graph_embeddings1",get_tensor_info(graph_embeddings))
        # g_data=None # to remove ref
        """matching nbs"""
        if self.use_nbs and nbs_ipids is not None:
            # print("self.use_nbs and nbs_ipids is not None",self.use_nbs and nbs_ipids is not None)
            print('nbs_g_data is not None')
            breakpoint()
            summ = text_embeddings
            for i in range(nbs_ipids.shape[0]):
                # print("nb",i)

                """====Encode Neighbor Text===="""
                # nb_features=text_embeddings

                nb_features = super(SEEBartEncoder, self).forward(
                    input_ids=nbs_ipids[i],
                    attention_mask=nbs_atmsk[i],
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict)[0]
                # print("nb_features",get_tensor_info(nb_features))

                """====Encode Neighbor Graph===="""
                nb_graph_embeddings = nb_features

                if nbs_root_indices is not None:
                    print('nbs_root_indices is not None', nbs_root_indices is not None)
                    tmp = []
                    nb_root_indices = nbs_root_indices[i].to(root_indices)
                    nb_root_masks = nbs_root_masks[i].to(root_masks)
                    for j in range(nb_graph_embeddings.shape[0]):
                        # noinspection PyTypeChecker
                        tmp.append(torch.index_select(nb_graph_embeddings[j], 0, nb_root_indices[j]) * nb_root_masks[j].unsqueeze(1))
                    nb_graph_embeddings = torch.stack(tmp, dim=0)
                    tmp = None
                if g_data is not None:
                    nb_g_data = nbs_g_data[i]
                    nb_graph_embeddings = self.encode_gather(nb_features, nbs_ipids[i], nb_g_data.idxs,
                                                             nb_g_data.masks, nb_g_data.max_token_num[0].item(), nb_g_data.max_token_len[0].item())
                    nb_features = None  # clear reference, no need neighbor text information

                    """gnn"""
                    nb_g_data.x = torch.tensor([])
                    nb_g_data.idxs = torch.tensor([])
                    nb_g_data.masks = torch.tensor([])
                    nb_g_data.token_num = torch.tensor([])
                    nb_g_data.token_len = torch.tensor([])  # these are not useful anymore
                    nb_g_data = nb_g_data.to(attention_mask)  # now only has edge

                    max_node_num_batch = nb_graph_embeddings.shape[1]
                    nb_graph_embeddings = nb_graph_embeddings.view(-1, nb_graph_embeddings.shape[-1])
                    nb_graph_embeddings = self.gnn(nb_graph_embeddings, nb_g_data)
                    # print("nb_graph_embeddings1",get_tensor_info(nb_graph_embeddings))
                    nbs_g_data[i] = None  # clear reference
                    # nb_g_data=None# clear reference

                    nb_graph_embeddings = nb_graph_embeddings.view(text_embeddings.shape[0], max_node_num_batch,
                                                                   nb_graph_embeddings.shape[-1])
                """====MultiHead Attention from Original===="""

                # if max_node_num_batch>3:
                #     nb_graph_embeddings=nb_graph_embeddings[:,3:,:]


                if self.addemb:
                    print("self.addemb")
                    attended_emb = self.submodule_1(text_embeddings=text_embeddings,
                                                    graph_embeddings=graph_embeddings,
                                                    nb_graph_embeddings=nb_graph_embeddings,
                                                    token2nodeid=token2nodeid,
                                                    token2rootnodeid=token2rootnodeid,
                                                    nbs_msk=nbs_msk[i])  # (batch size, dim_mode)
                        # nbs_attention_mask_list.append(nbs_atmsk[i])
                    tmp_nb_msk = nbs_msk[i].unsqueeze(-1).expand(-1, text_embeddings.shape[1]).unsqueeze(-1)
                    summ = summ + attended_emb*tmp_nb_msk  # text_embeddings
                    attended_emb=None

                if self.decattn:
                    print("self.decattn")
                    tmp_nb_msk = nbs_msk[i].unsqueeze(-1).expand(-1, nb_graph_embeddings.shape[1]).unsqueeze(-1)
                    # print("tmp_nb_msk", tmp_nb_msk)
                    # print("nb_graph_embeddings", nb_graph_embeddings)
                    # breakpoint()
                    nb_graph_embeddings=nb_graph_embeddings * tmp_nb_msk
                    nbs_hidden_states_list.append(nb_graph_embeddings)
                tmp_nb_msk = None
                nb_graph_embeddings = None  # clear reference
            encoder_outputs['last_hidden_state'] = summ
        elif entities is not None:
            # summ = torch.zeros_like(text_embeddings)

            summ = self.submodule_1(text_embeddings=text_embeddings, entities=entities, text_graphs=text_graphs, input_ids=input_ids, triggers=triggers, attn_mask=attention_mask)  # (batch size, dim_mode)
            # for j, sample in enumerate(ent_lists):
            #     for ent in sample:
            #         # print("ent",ent)
            #         tmp_dict=ent.copy()
            #         for key in ['mention_idxs','verb_idxs','token2nodepos']: #'max_token_num','max_token_len',
            #             tmp_dict[key] = ent[key].to(input_ids)
            #         for key in ['verb_masks','mention_masks','mention_existence_mask','verb_existence_mask']: # orig mask is whether this verb at a timespoint  exists
            #             tmp_dict[key] = ent[key].to(text_embeddings)
            #
            #         tmp_txt_emb=text_embeddings[j].unsqueeze(0)
            #         tmp_input_ids=input_ids[j].unsqueeze(0)
            #         graph_embeddings = self.encode_gather(tmp_txt_emb, tmp_input_ids, tmp_dict['mention_idxs'],
            #                                               tmp_dict['mention_masks'], tmp_dict['mention_max_token_num'], tmp_dict['mention_max_token_len'])
            #         graph_embeddings_verb = self.encode_gather(tmp_txt_emb, tmp_input_ids, tmp_dict['verb_idxs'],
            #                                               tmp_dict['verb_masks'], tmp_dict['verb_max_token_num'], tmp_dict['verb_max_token_len'])
            #
            #         tmp_mention_existence_mask=tmp_dict['mention_existence_mask'].unsqueeze(0).unsqueeze(-1)
            #         tmp_verb_existence_mask=tmp_dict['verb_existence_mask'].unsqueeze(0).unsqueeze(-1)
            #         graph_embeddings=graph_embeddings*(tmp_mention_existence_mask)
            #         tmp_mention_existence_mask=None
            #
            #         graph_embeddings_verb=graph_embeddings_verb*(tmp_verb_existence_mask)
            #         tmp_verb_existence_mask=None
            #
            #         graph_embeddings=torch.cat([graph_embeddings, graph_embeddings_verb], dim=-1)
            #         graph_embeddings_verb=None
            #
            #         tmp_token2nodepos=tmp_dict['token2nodepos'].unsqueeze(0)
            #         tmp_dict=None
            #
            #         attended_emb = self.submodule_1(text_embeddings=tmp_txt_emb,
            #                                         graph_embeddings=graph_embeddings,
            #                                         token2nodeid=tmp_token2nodepos)  # (batch size, dim_mode)
            #         tmp_token2nodepos=None
            #
            #         attended_emb=attended_emb.squeeze(0)
            #         summ[j]+=attended_emb
            #         attended_emb=None
            #         # summ = summ + attended_emb  # text_embeddings
            #
            #
            #         # text_embeddings[j]+=attended_emb
            #         # tmp_dict=None
            #         # for key in ['idxs','token2nodeid','masks']: #'max_token_num','max_token_len',
            #         #     ent[key] = None
            #         # ent_lists[j]=None

            encoder_outputs['last_hidden_state'] = summ#+text_embeddings
            # print("summ",summ.shape)


        else:
            print("no component")
            breakpoint()
            encoder_outputs['last_hidden_state'] = text_embeddings
        # print("nbs_hidden_states_list",nbs_hidden_states_list)
        return BaseModelOutputCustom(
            # last_hidden_state=encoder_outputs[0],
            last_hidden_state=encoder_outputs['last_hidden_state'],
            hidden_states=encoder_outputs['hidden_states'] if 'hidden_states' in encoder_outputs else None,
            attentions=encoder_outputs['attentions'] if 'attentions' in encoder_outputs else None,
            nbs_hidden_states_list=nbs_hidden_states_list,
        )

        # return encoder_outputs


class SEEBartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # self.layers = nn.ModuleList([BartDecoderLayer(config)]+[SEEBartDecoderLayer(config)]+[BartDecoderLayer(config) for _ in range(config.decoder_layers-2)])
        # self.layers = nn.ModuleList([SEEBartDecoderLayer(config)]+[SEEBartDecoderLayer(config)]+[SEEBartDecoderLayer(config) for _ in range(config.decoder_layers-2)])
        self.layers = nn.ModuleList([SEEBartDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        nbs_hidden_states_list = None,
        nbs_attention_mask_list= None,
        nbs_msk=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        # print("in BartDecoder")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        if nbs_hidden_states_list is not None and nbs_attention_mask_list is not None:
            nbs_attention_mask_list = [_expand_mask(nbs_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]) for nbs_attention_mask in nbs_attention_mask_list]


        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            # print("dec layer",idx)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    nbs_hidden_states_list = nbs_hidden_states_list,
                    nbs_attention_mask_list = nbs_attention_mask_list,
                    nbs_msk=nbs_msk,
                )

            hidden_states = layer_outputs[0]
            # print("head_mask",head_mask)
            # print("past_key_value",past_key_value)
            # print("cross_attn_head_mask",cross_attn_head_mask)
            # print("use_cache",use_cache)
            # print("output_attentions",output_attentions)
            # print("layer_outputs")

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )
@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class SEEBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig): #, args=None,pretrained_concept_emb=None
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.encoder = BartEncoder(config, self.shared) #, args=args, pretrained_concept_emb=pretrained_concept_emb
        self.encoder = SEEBartEncoder(config, self.shared) #, args=args, pretrained_concept_emb=pretrained_concept_emb

        self.decoder = BartDecoder(config, self.shared)
        # self.decoder = SEEBartDecoder(config, self.shared)
        # self.use_nbs=False
        # self.decattn=False

        self.init_weights()
    def add_modules(self, args, pretrained_concept_emb):

        if "match" in args.components or "est" in args.components:
            self.encoder.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
        args.g_dim=args.plm_hidden_dim
        self.encoder.use_nbs = "cbr" in args.components
        self.encoder.decattn = "decattn" in args.components

        self.encoder.addemb = "addemb" in args.components
        print("self.encoder.use_nbs",self.encoder.use_nbs)
        print("self.encoder.decattn",self.encoder.decattn)
        if "decattn" in args.components:
            if args.decattn_layer_idx=="all":
                for ly in self.decoder.layers:
                    ly.add_modules(args)
            else:
                self.decoder.layers[int(args.decattn_layer_idx)].add_modules(args)
        if "graph" in args.components:
            self.encoder.gnn = GNN(args=args, input_dim=args.g_dim, gnn_type=args.gnn_type, num_gnn_layers=args.num_gnn_layers,
                       encode_edge_features=True,pool_type=args.pool_type, global_pooling=False, )


        ##* donwstream, so init differently

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


    def encode_gather(self, text_embeddings, piece_idxs , idxs, masks, token_num, token_len):

        idxs=idxs.to(piece_idxs)
        masks=masks.to(text_embeddings)
        batch_size = text_embeddings.shape[0]

        # idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, text_embeddings.shape[-1]) # + 1
        idxs=idxs.unsqueeze(-1).expand(batch_size, -1, text_embeddings.shape[-1])

        # masks = text_embeddings.new(masks).unsqueeze(-1)
        masks =masks.unsqueeze(-1)

        text_embeddings = torch.gather(text_embeddings, 1, idxs) * masks
        text_embeddings = text_embeddings.view(batch_size, token_num, token_len,text_embeddings.shape[-1])
        text_embeddings = text_embeddings.sum(2)  # max_seq_len in words
        return text_embeddings
    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data_extra=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        token_lens=None,
        nbs_ipids=None,
        nbs_atmsk=None,
        token2nodeid=None,
        nbs_g_data=None,
        nbs_g_data_extra=None,
        token2rootnodeid=None,
        root_indices=None,
        root_masks=None,
        nbs_root_indices=None,
        nbs_root_masks=None,
        nbs_msk=None,
        ent_lists=None,
        entities = None,
        triggers = None,
        text_graphs = None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """====encoder===="""

        # nbs_hidden_states_list = []
        # nbs_attention_mask_list = []

        if aggregate_ipids is not None:
            input_ids=aggregate_ipids
            attention_mask=aggregate_atmsk

        # print("encoder_outputs is None", encoder_outputs is None)
        already_have_input=encoder_outputs is not None
        # if encoder_outputs is not None:
        #     """check anything becomes nothing"""
        #     print("encoder_outputs is not None")
        # if not self.training:
        #     breakpoint()

        if encoder_outputs is None:

            """====Encode Original Text===="""


            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                g_data=g_data,
                g_data2=g_data2,
                token2nodepos=token2nodepos,
                event_position_ids=event_position_ids,
                aggregate_ipids=aggregate_ipids,
                aggregate_atmsk=aggregate_atmsk,
                token_lens=token_lens,
                nbs_ipids=nbs_ipids,
                nbs_atmsk=nbs_atmsk,
                token2nodeid=token2nodeid,
                nbs_g_data=nbs_g_data,
                nbs_g_data_extra=nbs_g_data_extra,
                token2rootnodeid=token2rootnodeid,
                root_indices=root_indices,
                root_masks=root_masks,
                nbs_root_indices=nbs_root_indices,
                nbs_root_masks=nbs_root_masks,
                nbs_msk=nbs_msk,
                ent_lists=ent_lists,
                entities=entities,
                triggers=triggers,
                text_graphs=text_graphs,
            )

            # encoder_outputs = self.encoder(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     head_mask=head_mask,
            #     inputs_embeds=inputs_embeds,
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            #     g_data=g_data,
            #     g_data2=g_data2,
            #     token2nodepos=token2nodepos,
            #     event_position_ids=event_position_ids,
            #     aggregate_ipids=aggregate_ipids,
            #     aggregate_atmsk=aggregate_atmsk,
            # )
            #
            # text_embeddings=encoder_outputs[0]
            # # encoder_outputs[0]=None # to remove ref
            # encoder_outputs['last_hidden_state']=None # to remove ref
            #
            # """====Encode Original Graph===="""
            # """init graph embeds from token embeds"""
            #
            #
            # graph_embeddings=text_embeddings
            #
            # print('mdl fwd')
            #
            # if root_indices is not None:
            #     print('root_indices is not None')
            #
            #     tmp=[]
            #     for i in range(graph_embeddings.shape[0]):
            #         tmp.append(torch.index_select(graph_embeddings[i], 0, root_indices[i])*root_masks[i].unsqueeze(1))
            #     graph_embeddings=torch.stack(tmp, dim=0)
            #     tmp = None
            #
            #
            # if g_data is not None:
            #     print('g_data is not None')
            #
            #     graph_embeddings = self.encode_gather(text_embeddings, input_ids, g_data.idxs,
            #                                      g_data.masks, g_data.max_token_num[0].item(), g_data.max_token_len[0].item())
            #
            #     """gnn conv"""
            #     g_data.x=torch.tensor([])
            #     g_data.idxs=torch.tensor([])
            #     g_data.masks=torch.tensor([])
            #     g_data.token_num=torch.tensor([])
            #     g_data.token_len=torch.tensor([]) # these are not useful anymore
            #     g_data = g_data.to(attention_mask)
            #
            #     max_node_num_batch=graph_embeddings.shape[1]
            #     # print("max_node_num_batch",max_node_num_batch)
            #
            #     graph_embeddings=graph_embeddings.view(-1, graph_embeddings.shape[-1])
            #     graph_embeddings = self.gnn(graph_embeddings, g_data)
            #     g_data=None # clr ref
            #
            #     graph_embeddings = graph_embeddings.view(text_embeddings.shape[0], max_node_num_batch, graph_embeddings.shape[-1])
            #
            # # if max_node_num_batch > 3:
            # #     graph_embeddings = graph_embeddings[:, 3:, :]
            # # print("graph_embeddings1",get_tensor_info(graph_embeddings))
            # # g_data=None # to remove ref
            #
            # """matching nbs"""
            # if self.use_nbs and nbs_ipids is not None:
            #
            #
            #     summ=None
            #     for i in range(nbs_ipids.shape[0]):
            #         # print("nb",i)
            #
            #
            #         """====Encode Neighbor Text===="""
            #         # nb_features=text_embeddings
            #
            #         nb_features = self.encoder(
            #             input_ids=nbs_ipids[i],
            #             attention_mask=nbs_atmsk[i],
            #             head_mask=head_mask,
            #             inputs_embeds=inputs_embeds,
            #             output_attentions=output_attentions,
            #             output_hidden_states=output_hidden_states,
            #             return_dict=return_dict)[0]
            #         # print("nb_features",get_tensor_info(nb_features))
            #
            #         """====Encode Neighbor Graph===="""
            #         nb_graph_embeddings=nb_features
            #
            #         if nbs_root_indices is not None:
            #             print('nbs_root_indices is not None',nbs_root_indices is not None)
            #             tmp = []
            #             nb_root_indices=nbs_root_indices[i].to(root_indices)
            #             nb_root_masks=nbs_root_masks[i].to(root_masks)
            #             for j in range(nb_graph_embeddings.shape[0]):
            #                 tmp.append(torch.index_select(nb_graph_embeddings[j], 0, nb_root_indices[j]) * nb_root_masks[j].unsqueeze(1))
            #             nb_graph_embeddings = torch.stack(tmp, dim=0)
            #             tmp=None
            #         if g_data is not None:
            #             nb_g_data = nbs_g_data[i]
            #             nb_graph_embeddings=self.encode_gather(nb_features, nbs_ipids[i] , nb_g_data.idxs,
            #                                            nb_g_data.masks, nb_g_data.max_token_num[0].item(), nb_g_data.max_token_len[0].item())
            #             nb_features = None # clear reference, no need neighbor text information
            #
            #             """gnn"""
            #             nb_g_data.x = torch.tensor([])
            #             nb_g_data.idxs = torch.tensor([])
            #             nb_g_data.masks = torch.tensor([])
            #             nb_g_data.token_num = torch.tensor([])
            #             nb_g_data.token_len = torch.tensor([])  # these are not useful anymore
            #             nb_g_data = nb_g_data.to(attention_mask) # now only has edge
            #
            #             max_node_num_batch = nb_graph_embeddings.shape[1]
            #             nb_graph_embeddings=nb_graph_embeddings.view(-1, nb_graph_embeddings.shape[-1])
            #             nb_graph_embeddings=self.gnn(nb_graph_embeddings, nb_g_data)
            #             # print("nb_graph_embeddings1",get_tensor_info(nb_graph_embeddings))
            #             nbs_g_data[i]=None# clear reference
            #             # nb_g_data=None# clear reference
            #
            #             nb_graph_embeddings = nb_graph_embeddings.view(text_embeddings.shape[0], max_node_num_batch,
            #                                                      nb_graph_embeddings.shape[-1])
            #         """====MultiHead Attention from Original===="""
            #
            #         # if max_node_num_batch>3:
            #         #     nb_graph_embeddings=nb_graph_embeddings[:,3:,:]
            #
            #         attended_emb = self.submodule_1(text_embeddings=text_embeddings,
            #                                          graph_embeddings=graph_embeddings,
            #                                          nb_graph_embeddings=nb_graph_embeddings,
            #                                          token2nodeid=token2nodeid,
            #                                         token2rootnodeid=token2rootnodeid,
            #                                         nbs_msk=nbs_msk[i])  # (batch size, dim_mode)
            #         if self.decattn:
            #             nbs_hidden_states_list.append(nb_graph_embeddings)
            #             nbs_attention_mask_list.append(nbs_atmsk[i])
            #         nb_graph_embeddings=None # clear reference
            #
            #         if summ is None:
            #             summ = attended_emb
            #         else:
            #             summ += attended_emb #text_embeddings
            #     encoder_outputs['last_hidden_state']=summ+text_embeddings
            # else:
            #     print("no component")
            #     encoder_outputs['last_hidden_state'] = text_embeddings

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutputCustom):
            # breakpoint()
            encoder_outputs = BaseModelOutputCustom(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
                nbs_hidden_states_list=encoder_outputs['nbs_hidden_states_list'] if 'nbs_hidden_states_list' in encoder_outputs else None,
            )

        # print("encoder_outputs[0]", encoder_outputs[0].shape)
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # print()
        """!!test if chagning positions in encoder_outputs, orederdict """
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # nbs_hidden_states_list = encoder_outputs["nbs_hidden_states_list"],
            # nbs_attention_mask_list = nbs_atmsk,
            # nbs_msk=nbs_msk,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig): #, args=None,pretrained_concept_emb=None
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared) #, args=args, pretrained_concept_emb=pretrained_concept_emb
        # self.decoder = BartDecoder(config, self.shared)
        self.decoder = SEEBartDecoder(config, self.shared)

        self.init_weights()
    def add_modules(self, args, pretrained_concept_emb):
        self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
        ##* donwstream, so init differently

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        token_lens=None,
        self_graph=None,
        nbs_graph=None,
    ):


        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """====encoder===="""

        if g_data is not None:
            g_data = g_data.to(attention_mask)
            g_data2 = g_data2.to(attention_mask)
            token2nodepos = token2nodepos.to(attention_mask)  # .cuda()#.to(attention_mask)
            event_position_ids = event_position_ids.to(attention_mask)  # .cuda()#.to(attention_mask)
        if aggregate_ipids is not None:
            # print("aggregate_input_ids", aggregate_input_ids.device)
            # aggregate_input_ids = aggregate_input_ids.to(attention_mask)
            # aggregate_attention_mask = aggregate_attention_mask.to(attention_mask)
            input_ids=aggregate_ipids
            attention_mask=aggregate_atmsk
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                g_data=g_data,
                g_data2=g_data2,
                token2nodepos=token2nodepos,
                event_position_ids=event_position_ids,
                aggregate_ipids=aggregate_ipids,
                aggregate_atmsk=aggregate_atmsk,
            )
            # print("encoder_outputs[1]",encoder_outputs[1])
            if "submodule_1" in dir(self):
                encoder_outputs[0] = self.submodule_1(encoder_outputs[0],
                                                 g_data,
                                                 g_data2,
                                                 token2nodepos,
                                                 token_lens=token_lens, piece_idxs=input_ids)  # (batch size, dim_mode)




        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # print("encoder_outputs[0]", encoder_outputs[0].shape)
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.",
    BART_START_DOCSTRING,
)
class BartModelForClassification(BartPretrainedModel):
    def __init__(self, config: BartConfig): #, args=None,pretrained_concept_emb=None
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared) #, args=args, pretrained_concept_emb=pretrained_concept_emb
        # self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        # self.decoder.embed_tokens = self.shared

    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        # print("new resized embeddings")
        if new_num_tokens is None:
            return old_embeddings

        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
                old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim, padding_idx=old_embeddings.padding_idx).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        else:
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings
    def get_encoder(self):
        return self.encoder

    # def get_decoder(self):
    #     return self.decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        # if decoder_input_ids is None and decoder_inputs_embeds is None:
        #     decoder_input_ids = shift_tokens_right(
        #         input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        #     )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                g_data=g_data,
                g_data2=g_data2,
                token2nodepos=token2nodepos,
                event_position_ids=event_position_ids,
                aggregate_ipids=aggregate_ipids,
                aggregate_atmsk=aggregate_atmsk,

            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        return encoder_outputs
        # print("encoder_outputs[0]", encoder_outputs[0].shape)
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

@add_start_docstrings(
    "The TPE Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class SEEBartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig): #, args=None,pretrained_concept_emb=None
        super().__init__(config)
        self.model = SEEBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # def _get_resized_embeddings(
    #     self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    # ) -> nn.Embedding:
    #     """
    #     Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
    #     initialized vectors at the end. Reducing the size will remove vectors from the end
    #
    #     Args:
    #         old_embeddings (:obj:`torch.nn.Embedding`):
    #             Old embeddings to be resized.
    #         new_num_tokens (:obj:`int`, `optional`):
    #             New number of tokens in the embedding matrix.
    #
    #             Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
    #             vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
    #             :obj:`torch.nn.Embedding`` module of the model without doing anything.
    #
    #     Return:
    #         :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
    #         :obj:`new_num_tokens` is :obj:`None`
    #     """
    #     # print("new resized embeddings")
    #     if new_num_tokens is None:
    #         return old_embeddings
    #
    #     if is_deepspeed_zero3_enabled():
    #         import deepspeed
    #
    #         with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=None):
    #             old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    #     else:
    #         old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    #
    #     if old_num_tokens == new_num_tokens:
    #         return old_embeddings
    #
    #     if not isinstance(old_embeddings, nn.Embedding):
    #         raise TypeError(
    #             f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
    #             f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
    #         )
    #
    #     # Build new embeddings
    #     new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim, padding_idx=old_embeddings.padding_idx).to(
    #         self.device, dtype=old_embeddings.weight.dtype
    #     )
    #
    #     # initialize all new embeddings (in particular added tokens)
    #     self._init_weights(new_embeddings)
    #
    #     # Copy token embeddings from the previous weights
    #
    #     # numbers of tokens to copy
    #     n = min(old_num_tokens, new_num_tokens)
    #     if is_deepspeed_zero3_enabled():
    #         import deepspeed
    #
    #         with deepspeed.zero.GatheredParameters(old_embeddings.weight, modifier_rank=0):
    #             if torch.distributed.get_rank() == 0:
    #                 new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    #     else:
    #         new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    #
    #     return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        g_data=None,
        g_data_extra=None,
        g_data2=None,
        token2nodepos=None,
        event_position_ids=None,
        aggregate_ipids=None,
        aggregate_atmsk=None,
        token_lens=None,
        nbs_ipids=None,
        nbs_atmsk=None,
        token2nodeid=None,
        nbs_g_data=None,
        nbs_g_data_extra=None,
        token2rootnodeid=None,
        root_indices=None,
        root_masks=None,
        nbs_root_indices=None,
        nbs_root_masks=None,
            nbs_msk=None,
            ent_lists=None,
        entities=None,
        triggers=None,
        text_graphs=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        # print("event_position_ids is None", event_position_ids is None)
        # print("in BartForConditionalGeneration")
        # if g_data is not None: print("g_data", get_tensor_info(g_data.x))
        # if g_data2 is not None: print("g_data2", get_tensor_info(g_data2.x))
        # if event_position_ids is not None: print("event_position_ids.shape", event_position_ids.shape)

        # gc.collect()

        # def view_model_param_state(model, rq=True):
        #     for name, param in model.named_parameters():
        #         if rq and param.requires_grad:
        #             print(name, " | ", param.requires_grad, " | ", param.data.size())
        #
        # view_model_param_state(self, rq=True)



        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            g_data=g_data,
            g_data_extra=g_data_extra,
            g_data2=g_data2,
            token2nodepos=token2nodepos,
            event_position_ids=event_position_ids,
            aggregate_ipids=aggregate_ipids,
            aggregate_atmsk=aggregate_atmsk,
            token_lens=token_lens,
            nbs_ipids=nbs_ipids,
            nbs_atmsk=nbs_atmsk,
            token2nodeid=token2nodeid,
            nbs_g_data=nbs_g_data,
            nbs_g_data_extra=nbs_g_data_extra,
        token2rootnodeid=token2rootnodeid,
        root_indices=root_indices,
        root_masks=root_masks,
        nbs_root_indices=nbs_root_indices,
        nbs_root_masks=nbs_root_masks,
            nbs_msk=nbs_msk,
            ent_lists=ent_lists,
            entities=entities,
            triggers=triggers,
            text_graphs=text_graphs,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        # print("self.final_logits_bias", self.final_logits_bias)
        # print("self.final_logits_bias.shape", self.final_logits_bias.shape)
        # print("lm_logits.shape",lm_logits.shape)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # add KLD

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        tmp_dict={
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)


        }
        # tmp_dict.update(kwargs)
        return tmp_dict

        # return {
        #     "input_ids": None,  # encoder_outputs is defined. input_ids not needed
        #     "encoder_outputs": encoder_outputs,
        #     "past_key_values": past,
        #     "decoder_input_ids": decoder_input_ids,
        #     "attention_mask": attention_mask,
        #     "head_mask": head_mask,
        #     "decoder_head_mask": decoder_head_mask,
        #     "cross_attn_head_mask": cross_attn_head_mask,
        #     "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        #
        #
        # }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

# @add_start_docstrings(
#     "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
# )
# class BartForConditionalGeneration(BartPretrainedModel):
#     base_model_prefix = "model"
#     _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]
#
#     def __init__(self, config: BartConfig, args=None,pretrained_concept_emb=None):
#         super().__init__(config)
#         self.model = BartModel(config, args=args, pretrained_concept_emb=pretrained_concept_emb)
#         self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
#         self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
#
#         self.init_weights()
#
#     def get_encoder(self):
#         return self.model.get_encoder()
#
#     def get_decoder(self):
#         return self.model.get_decoder()
#
#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings
#
#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings
#
#     @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
#     @add_end_docstrings(BART_GENERATION_EXAMPLE)
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         decoder_input_ids=None,
#         decoder_attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         encoder_outputs=None,
#         past_key_values=None,
#         inputs_embeds=None,
#         decoder_inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#             g_data=None,
#             g_data2=None,
#             token2nodepos=None,
#             event_position_ids=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
#             Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
#             config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
#             (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
#
#         Returns:
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         if labels is not None:
#             if decoder_input_ids is None:
#                 decoder_input_ids = shift_tokens_right(
#                     labels, self.config.pad_token_id, self.config.decoder_start_token_id
#                 )
#
#         outputs = self.model(
#             input_ids,
#             attention_mask=attention_mask,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=encoder_outputs,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             decoder_inputs_embeds=decoder_inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             g_data=g_data,
#             g_data2=g_data2,
#             token2nodepos=token2nodepos,
#             event_position_ids=event_position_ids,
#         )
#         lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
#
#         masked_lm_loss = None
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
#
#         if not return_dict:
#             output = (lm_logits,) + outputs[1:]
#             return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
#
#         return Seq2SeqLMOutput(
#             loss=masked_lm_loss,
#             logits=lm_logits,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )
#
#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         past=None,
#         attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         use_cache=None,
#         encoder_outputs=None,
#         **kwargs
#     ):
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]
#
#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "decoder_head_mask": decoder_head_mask,
#             "cross_attn_head_mask": cross_attn_head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }
#
#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
#
#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past

