import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from utils.utils import freeze_net
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import numpy as np
# from torch import optim

def index_select_from_embeddings(embeddings, lens=None, indices_list=None, batch_embeddings=False, merge=False, is_torch_embedding=False):
    """
    :param embeddings: (B, L, D)
    :param indices_list: (B, L)
    :return: (B, L, D)
    """
    tmp = []
    if not indices_list:
        indices_list = [torch.arange(lens[i], dtype=torch.long, device=embeddings.device) for i in range(len(lens))]
    else:
        indices_list = [torch.as_tensor(indices_list[i], dtype=torch.long, device=embeddings.device) for i in range(len(indices_list))]
    for i, indices in enumerate(indices_list):
        tmp.append(torch.index_select(embeddings[i] if batch_embeddings else embeddings, 0, indices))
    return torch.cat(tmp, dim=0) if merge else torch.stack(tmp, dim=0)

def encode_factors(text_embeddings, input_ids, info_dict_orig, single_batch_multi_target=False):
    info_dict = info_dict_orig.copy()
    for key in ['idxs']:  # 'max_token_num','max_token_len',
        info_dict[key] = info_dict[key].to(input_ids)
    for key in ['masks']:  # orig mask is whether this verb at a timespoint  exists
        info_dict[key] = info_dict[key].to(text_embeddings)

    # if single_batch_multi_target:
    #     tmp = []
    #     for i in range(info_dict['idxs'].shape[0]):  # ith entity
    #         tmp.append(encode_gather(text_embeddings.unsqueeze(0), input_ids.unsqueeze(0), info_dict['idxs'][i].unsqueeze(0), info_dict['masks'][i].unsqueeze(0),
    #                                  info_dict['max_token_num'], info_dict['max_token_len']))
    #     return torch.cat(tmp, dim=0)

    return encode_gather(text_embeddings, input_ids, info_dict['idxs'], info_dict['masks'], info_dict['max_token_num'], info_dict['max_token_len'])


def encode_factors2(text_embeddings, input_ids, info_dict_orig):
    info_dict = info_dict_orig.copy()
    for key in ['idxs']:  # 'max_token_num','max_token_len',
        info_dict[key] = info_dict[key].to(input_ids)
    for key in ['masks']:  # orig mask is whether this verb at a timespoint  exists
        info_dict[key] = info_dict[key].to(text_embeddings)
    tmp = []
    for i in range(info_dict['idxs'].shape[0]):  # ith entity
        tmp.append(encode_gather(text_embeddings.unsqueeze(0), input_ids.unsqueeze(0), info_dict['idxs'][i].unsqueeze(0), info_dict['masks'][i].unsqueeze(0),
                                 info_dict['max_token_num'], info_dict['max_token_len']))
    return torch.cat(tmp, dim=0)




def extract_unmasked_embeddings(embeddings, token2nodeid, batch_embeddings=False):


    # if single_batch_src and batch_embeddings:
    #     token_has_node_mask = (token2nodeid != -1).long()
    #     updated_mask = token2nodeid * token_has_node_mask  # Bx seqlen # -1's  arefilled with 0
    #     # updated_mask=updated_mask.float()
    #     tmp = []
    #     for i in range(embeddings.shape[0]):
    #         tmp.append(torch.index_select(embeddings if not batch_embeddings else embeddings[i], 0, updated_mask[i]) * token_has_node_mask[i].unsqueeze(1))
    #     return torch.stack(tmp, dim=0)
    token_has_node_mask = (token2nodeid != -1).long()
    updated_mask = token2nodeid * token_has_node_mask  # Bx seqlen # -1's  arefilled with 0
    # updated_mask=updated_mask.float()
    tmp = []
    for i in range(token2nodeid.shape[0]):
        tmp.append(torch.index_select(embeddings if not batch_embeddings else embeddings[i], 0, updated_mask[i]) * token_has_node_mask[i].unsqueeze(1))
    return torch.stack(tmp, dim=0)

def encode_gather(text_embeddings, piece_idxs , idxs, masks, token_num, token_len):
    """
    :param text_embeddings: (B, L, D)
    :param text_embeddings:
    :param piece_idxs:
    :param idxs:
    :param masks:
    :param token_num:
    :param token_len:
    :return: batch_size x max_token_num x D
    """
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

def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len



class ModifiedBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        print("config.max_position_embeddings", config.max_position_embeddings)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.event_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("event_position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, event_position_ids=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
            event_position_embeddings = self.event_position_embeddings(event_position_ids)
            embeddings += event_position_embeddings


        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)

def get_tensor_info(tensor):
    return f"Shape: {tensor.shape} | Type: {tensor.type()} | Device: {tensor.device}"


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)

def get_loss_fn(loss_name):
    if loss_name == "mse":
        return torch.nn.MSELoss()
    elif loss_name == "bce":
        return torch.nn.BCELoss()
    elif loss_name == "bce_logit":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == "ce":
        return torch.nn.CrossEntropyLoss()
    elif loss_name == "kl":
        return torch.nn.KLDivLoss()
    elif loss_name == "nll":
        return torch.nn.NLLLoss()
    else:
        assert False, f"loss_name {loss_name} not valid"


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class CrossModalAttention(nn.Module):
    def __init__(self, reduction='mean', m1_dim=0, m2_dim=0, final_dim=800):
        super().__init__()
        # self.epsilon = epsilon
        self.reduction = reduction
        self.temprature = 3
        # self.aggregate = True
        self.l_filter1 = torch.nn.Linear(m1_dim, final_dim)
        self.l_filter2 = torch.nn.Linear(m2_dim, final_dim)

        self.l1 = torch.nn.Linear(m1_dim, final_dim)
        self.l2 = torch.nn.Linear(m2_dim, final_dim)

    def forward(self, m1, m2, aggregate=True):
        """
        :param m1: vectors of a modality
        :param m2: same as above
        :return: attended embeddings
        """
        # print("m1", get_tensor_info(m1))
        # print("m2", get_tensor_info(m2))
        # print("m1", m1)
        # print("m2", m2)

        # norms1, norms2 = torch.norm(m1, dim=-1, p=2, keepdim=True), torch.norm(m2, dim=-1, p=2, keepdim=True)
        # # print("norms1, norms2", norms1, norms2)
        #
        # norm_prod_mat = norms1.matmul(norms2.t())
        # # print("norm_prod_mat.shape", norm_prod_mat.shape)
        raw_prod = m1.matmul(m2.t())
        # # print("raw_prod", get_tensor_info(raw_prod))
        # # print("raw_prod", raw_prod)
        #
        # cos_sim_mat = raw_prod / norm_prod_mat
        # # print("cos_sim_mat", cos_sim_mat)
        # c_plus = torch.relu(cos_sim_mat)
        # # print("c_plus", c_plus)
        #
        # # c_hat=torch.tensor()
        # # sum_each_row = c_plus.pow(2).sum(dim=-1, keepdim=True).sqrt()
        # # sum_each_row=torch.sqrt(torch.sum(torch.pow(c_hat, 2), dim=-1, keepdim=True))
        # c_hat = c_plus / torch.norm(c_plus, dim=-1, p=2, keepdim=True)  # / sum_each_row
        # print("c_hat", c_hat)

        # print("c_hat", get_tensor_info(c_hat))
        # + torch.tensor(1e-7, device=c_hat.device)
        # alpha = torch.softmax(self.temprature * (c_hat.t()+torch.tensor(1e-7, device=c_hat.device)), dim=-1)  # .t()
        # print("self.temprature*(c_hat.t()+torch.tensor(1e-7, device=c_hat.device)",
        #       self.temprature * (c_hat.t() + torch.tensor(1e-7, device=c_hat.device)))
        alpha2 = torch.softmax(raw_prod, dim=-1)
        alpha1 = torch.softmax(raw_prod.t(), dim=-1)  # + torch.tensor(1e-9, device=m1.device)
        # print("alpha", alpha)

        attended_m1 = alpha1.matmul(m1)
        attended_m2 = alpha2.matmul(m2)
        # print("attended_m1.mean(0)", get_tensor_info(attended_m1.mean(0)))

        # return attended_m1.mean(0)
        # print("attended_m1", get_tensor_info(attended_m1))
        # print("attended_m1", attended_m1)
        # return F.tanh(self.l1(attended_m1.sum(0)))
        if aggregate:
            attended_m1 = attended_m1
            attended_m2 = attended_m2

            # m2 = m2.mean(0)
            # print("numnan", torch.sum(torch.isnan(torch.cat([attended_m1, m2], dim=-1))))

            attended_m1 = F.dropout(attended_m1, 0.1, training=self.training)
            # m2 = F.dropout(m2, 0.1, training=self.training)
            attended_m2 = F.dropout(attended_m2, 0.1, training=self.training)
            # filter = torch.sigmoid(self.l_filter(torch.cat([attended_m1, m2], dim=-1)))  # .sum(0)
            # filter1 = torch.sigmoid(self.l_filter1(torch.cat([attended_m2], dim=-1)).mean(0))  # .sum(0)
            # filter2 = torch.sigmoid(self.l_filter2(torch.cat([attended_m1], dim=-1)).mean(0))  # .sum(0)
            filter1 = torch.sigmoid(self.l_filter1(attended_m2).mean(0))  # .sum(0)
            filter2 = torch.sigmoid(self.l_filter2(attended_m1).mean(0))  # .sum(0)
            # print("attended_m2", get_tensor_info(attended_m2))
            # print("attended_m2", get_tensor_info(attended_m2))

            # print("dropout attended_m1", attended_m1)
            # print("dropout m2", m2)
            # print("filter", filter)
            # return attended_m1.sum(0)

            transformed_m1 = torch.tanh(self.l1(attended_m1)).mean(0)
            transformed_m2 = torch.tanh(self.l2(attended_m2)).mean(0)
            # print("transformed_m1", transformed_m1)
            # print("transformed_m2", transformed_m2)
            # print("filter", get_tensor_info(filter))
            # print("transformed_m1", get_tensor_info(transformed_m1))
            # print("transformed_m2", get_tensor_info(transformed_m2))

            return transformed_m1 * filter1, transformed_m2 * filter2

        return attended_m1, m2


def encode(self, input_ids, attention_mask):
    config = self.config
    if config.transformer_type == "bert":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id]
    elif config.transformer_type == "roberta":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id, config.sep_token_id]
    sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
    return sequence_output, attention


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention

class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn

class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


# emb as no cuda
class CustomizedEmbedding2(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                print("freeze_ent_emb", )
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale



class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                print("freeze_ent_emb", )
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale
