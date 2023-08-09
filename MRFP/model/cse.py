from torch.nn import Linear, GELU, Tanh
import torch.utils.data
# from transformers.models.bert.modeling_bert import ModifiedBertModel
# from transformers.models.bert_generation.modeling_bert_generation import BertGenerationEncoder
from .gnn import GNN
from model.model_utils import *
from copy import deepcopy
from IPython import embed
import gc
from transformers import AutoModel

import torch.utils.data
import torch.nn.utils.rnn as R
# from .model.bart.modeling_bart import BartEncoder

from torch.nn import Sigmoid, ReLU

from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer, MultiheadAttention, TransformerEncoder

from .bert.modeling_bert import BertModel


class WMG_Network(nn.Module):
    def __init__(self, observation_space, action_space):
        super(WMG_Network, self).__init__()
        pass


class EventReasoningModule2(nn.Module):

    def __init__(self, args=None, pretrained_concept_emb=None):
        super().__init__()
        # print("args.g_dim", args.g_dim)
        self.components = args.components

        self.batch_size = args.batch_size
        self.num_sents_per_sample = args.num_nbs + 1

        # self.extra_emb = nn.Embedding(2, args.g_dim, 0)
        # self.extra_emb = nn.Embedding(2, args.plm_hidden_dim, 0)
        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device, requires_grad=False)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)
        # self.rand_emb = nn.Embedding(2, args.plm_hidden_dim)

        # self.activation = GELU()
        self.activation = Tanh()
        # self.pooler = MultiheadAttPoolLayer(args.n_attention_head, args.plm_hidden_dim, args.g_dim)
        # self.fc = MLP(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim, args.plm_hidden_dim, 1, dropout=args.dropoutf, layer_norm=True)
        # self.fc=Linear(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim)
        # self.cross_attn=TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head)

        # self.fc1 = Linear((args.g_dim + args.plm_hidden_dim) if "wm" in self.components else args.plm_hidden_dim, args.g_dim)  # args.plm_hidden_dim
        # self.fc2=Linear(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim)

        final_dim = args.plm_hidden_dim
        # self.fc3 = Linear(final_dim, args.plm_hidden_dim)

        self.dropout_e = nn.Dropout(args.dropouti)
        self.dropout_fc = nn.Dropout(args.dropoutf)
        self.plm_hidden_dim = args.plm_hidden_dim

        # self.wm = KnowMemSpace(args, pretrained_concept_emb=pretrained_concept_emb)
        # self.wm = GNN(args, gnn_type=args.gnn_type, num_gnn_layers=args.num_gnn_layers,
        #               encode_node_features=True, encode_edge_features=False,
        #               pool_type=args.pool_type, global_pooling=False, pretrained_concept_emb=pretrained_concept_emb)
        # self.gnn = GNN(args, input_dim=args.g_dim, gnn_type=args.gnn_type2, num_gnn_layers=args.num_gnn_layers2,
        #                 # input_dim=args.plm_hidden_dim,
        #                 encode_edge_features=True,
        #                 pool_type=args.pool_type, global_pooling=False, )

        if "cbr" in self.components and "maxmat" not in self.components and "nomat" not in self.components:
            self.multihead_attn = nn.MultiheadAttention(args.plm_hidden_dim, num_heads=8, kdim=args.plm_hidden_dim, vdim=args.plm_hidden_dim, batch_first=True)  # //2
        if "sim" in self.components:
            self.sim_fc = Linear(args.plm_hidden_dim * 2, 1)

        if "maxmat" in self.components and "nomat" not in self.components:
            self.key_map = Linear(args.plm_hidden_dim, args.plm_hidden_dim)
            self.query_map = Linear(args.plm_hidden_dim, args.plm_hidden_dim)
            self.value_map = Linear(args.plm_hidden_dim, args.plm_hidden_dim)

        if "est" in self.components:
            self.lstm = nn.LSTM(input_size=args.plm_hidden_dim, hidden_size=args.plm_hidden_dim, batch_first=True, dropout=args.dropouti, bidirectional=True)  # //2
            self.fc2 = Linear(args.plm_hidden_dim * 2, args.plm_hidden_dim)
            self.fc2_1 = Linear(args.plm_hidden_dim * 2, args.plm_hidden_dim)
            self.fc_transform_orig = Linear(args.plm_hidden_dim, args.plm_hidden_dim)

            # self.lstm_dropout = nn.Dropout(self.dropout_e)

            # self.init_range = args.init_range
            # if args.init_range > 0:
            #     self.apply(self._init_weights2)

        # self.q_proj=Linear(args.plm_hidden_dim, args.plm_hidden_dim)
        # self.k_proj=Linear(args.plm_hidden_dim, args.plm_hidden_dim)
        # self.v_proj=Linear(args.plm_hidden_dim, args.plm_hidden_dim)

    def _init_weights2(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # aggregate_input_ids = None,
    # aggregate_attention_mask = None

    def encode(self, text_embeddings, piece_idxs, token_lens):

        batch_size = text_embeddings.shape[0]
        idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
        masks = text_embeddings.new(masks).unsqueeze(-1)
        text_embeddings = torch.gather(text_embeddings, 1, idxs) * masks
        text_embeddings = text_embeddings.view(batch_size, token_num, token_len, self.bert_dim)
        text_embeddings = text_embeddings.sum(2)  # max_seq_len in words
        return text_embeddings

    def forward(self, text_embeddings=None, graph_embeddings=None, nb_text_embeddings=None, nb_graph_embeddings=None, token2nodeid=None, token2rootnodeid=None, nbs_msk=None):
        """

        :param text_embeddings: BxNxD
        :param g_data:
        :param g_data2:
        :param token2nodepos:
        :param token_lens:
        :param piece_idxs:
        :param nbs_msk: B
        :return:
        """
        # word2node
        """Use Data Instance to avoid being converted to cuda"""

        if self.args.use_graph:
            if self.args.use_graph_data:
                g_data = graph_embeddings
        pass

        # if self.args.use_graph:


class EventReasoningModule(nn.Module):

    def __init__(self, args=None, pretrained_concept_emb=None):
        super().__init__()
        # print("args.g_dim", args.g_dim)
        self.components = args.components

        self.batch_size = args.batch_size
        self.num_sents_per_sample = args.num_nbs + 1
        self.propagate_factor_embeddings = args.propagate_factor_embeddings
        self.ablation = args.ablation
        self.use_token_tag= args.use_token_tag
        self.cat_text_embed= args.cat_text_embed


        # self.extra_emb = nn.Embedding(2, args.g_dim, 0)
        # self.extra_emb = nn.Embedding(2, args.plm_hidden_dim, 0)
        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device, requires_grad=False)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)
        # self.rand_emb = nn.Embedding(2, args.plm_hidden_dim)
        self.debug=args.debug

        # self.activation = GELU()
        self.activation = Tanh()
        # self.pooler = MultiheadAttPoolLayer(args.n_attention_head, args.plm_hidden_dim, args.g_dim)
        # self.fc = MLP(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim, args.plm_hidden_dim, 1, dropout=args.dropoutf, layer_norm=True)
        # self.fc=Linear(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim)
        # self.cross_attn=TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head)

        # self.fc1 = Linear((args.g_dim + args.plm_hidden_dim) if "wm" in self.components else args.plm_hidden_dim, args.g_dim)  # args.plm_hidden_dim
        # self.fc2=Linear(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim)

        final_dim = args.plm_hidden_dim

        self.dropout_e = nn.Dropout(args.dropouti)
        self.dropout_fc = nn.Dropout(args.dropoutf)
        self.plm_hidden_dim = args.plm_hidden_dim

        if "cbr" in self.components and "maxmat" not in self.components and "nomat" not in self.components:
            self.multihead_attn = nn.MultiheadAttention(args.plm_hidden_dim, num_heads=8, kdim=args.plm_hidden_dim, vdim=args.plm_hidden_dim, batch_first=True)  # //2
        if "sim" in self.components:
            self.sim_fc = Linear(args.plm_hidden_dim * 2, 1)

        if "maxmat" in self.components and "nomat" not in self.components:
            self.key_map = Linear(args.plm_hidden_dim, args.plm_hidden_dim)
            self.query_map = Linear(args.plm_hidden_dim, args.plm_hidden_dim)
            self.value_map = Linear(args.plm_hidden_dim, args.plm_hidden_dim)

        if "est" in self.components:

            self.internal_dim=args.plm_hidden_dim if not self.use_token_tag else args.plm_hidden_dim*2

            self.factor_transform = nn.Sequential(
                nn.Dropout(args.dropouti),
                Linear(self.internal_dim, self.internal_dim),
                # GELU(),
                Tanh()
            )
            self.lstm = nn.LSTM(input_size=self.internal_dim,
                                hidden_size=self.internal_dim,
                                batch_first=True, dropout=args.dropouti, bidirectional=True)  # //2
            # self.fc2 = Linear(args.plm_hidden_dim * 2, args.plm_hidden_dim)
            self.fc2_1 = Linear(self.internal_dim * 2, self.internal_dim)
            # self.fc_transform_orig = Linear(args.plm_hidden_dim, args.plm_hidden_dim)
            # self.factor_memory = nn.Embedding(args.num_factor_types, args.plm_hidden_dim)
            # self.trigger_memory = nn.Embedding(args.num_trigger_types, args.plm_hidden_dim)
            # self.gate_fc = nn.Sequential(
            #     Linear(args.plm_hidden_dim * 2, 1),
            #     Sigmoid()
            # )
            if self.use_token_tag:
                self.tag_embeddings = nn.Embedding(args.num_tag_types, args.plm_hidden_dim)
                self.tag_transform = nn.Sequential(
                    Linear(args.plm_hidden_dim*2, args.plm_hidden_dim*2),
                    # GELU(),
                    # nn.Dropout(args.dropouti),
                )
                self.transform_node_back = nn.Sequential(
                    Linear(args.plm_hidden_dim*2, args.plm_hidden_dim),
                    GELU(),
                    # Tanh(),
                    nn.Dropout(args.dropouti),
                )
                self.transform_temporal_back = nn.Sequential(
                    Linear(args.plm_hidden_dim*2, args.plm_hidden_dim),
                    GELU(),
                    # Tanh(),
                    nn.Dropout(args.dropouti),
                )
                if self.cat_text_embed:
                    self.transform_cat_text = Linear(args.plm_hidden_dim*3, args.plm_hidden_dim)#self.internal_dim*2+
            if self.propagate_factor_embeddings !=0:
                self.propagator_1 = TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head, batch_first=True)
                self.propagator_2 = TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head, batch_first=True)

                self.trigger_transform = nn.Sequential(
                    nn.Dropout(args.dropouti),
                    Linear(args.plm_hidden_dim, args.plm_hidden_dim),
                    Tanh()
                )
                self.fc3_1 = Linear(args.plm_hidden_dim * 2, args.plm_hidden_dim)
                self.lstm_evt = nn.LSTM(input_size=args.plm_hidden_dim, hidden_size=args.plm_hidden_dim, batch_first=True, dropout=args.dropouti, bidirectional=True)  # //2


            # self.edge_transform = [Linear(args.plm_hidden_dim, args.plm_hidden_dim) for _ in range(args.num_edge_types)]

            if args.temporal_encoding == "transformer":
                self.lstm = TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head, batch_first=True)
                self.lstm_evt = TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head, batch_first=True)
                # 还有啥好的loss函数？
            if args.intra_event_encoding:
                self.graph_encoder = GNN(args=args, input_dim=self.internal_dim, g_dim=self.internal_dim, gnn_type=args.gnn_type, num_gnn_layers=2,
                                         encode_node_features=False, encode_edge_features=args.gnn_type in ["gine"], dropout_at_beginning=True, activation_at_the_end=True,
                                         pool_type="mean", global_pooling=False, pretrained_concept_emb=None)

            # self.lstm = TransformerEncoder(encoder_layer, 1)
            # encoder_layer =  TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=args.n_attention_head,
            #                                                  dim_feedforward=args.plm_hidden_dim, dropout=args.dropoutf)

            # self.lstm_dropout = nn.Dropout(self.dropout_e)

            # self.init_range = args.init_range
            # if args.init_range > 0:
            #     self.apply(self._init_weights2)

        # self.q_proj=Linear(args.plm_hidden_dim, args.plm_hidden_dim)
        # self.k_proj=Linear(args.plm_hidden_dim, args.plm_hidden_dim)
        # self.v_proj=Linear(args.plm_hidden_dim, args.plm_hidden_dim)

    def _init_weights2(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_embeddings=None, text_graphs=None, input_ids=None, entities=None, triggers=None, attn_mask=None):
        """
        :param text_embeddings: BxNxD
        :param g_data:
        :param g_data2:
        :param token2nodepos:
        :param token_lens:
        :param piece_idxs:
        :param nbs_msk: B
        :return:
        """
        if "est" in self.components:
            # lstm_in = R.pack_padded_sequence(graph_embeddings,torch.tensor([graph_embeddings.shape[1]], dtype=torch.long, device=graph_embeddings.device),
            #                                  batch_first=True, enforce_sorted=False)

            tmp = []
            # summ = torch.zeros_like(text_embeddings)
            # summ = text_embeddings
            # for key in ['idxs', 'token2nodepos']:  # 'max_token_num','max_token_len',
            #     text_graph[key] = text_graph[key].to(input_ids)
            # for key in ['masks']:  # orig mask is whether this verb at a timespoint  exists
            #     text_graph[key] = text_graph[key].to(text_embeddings)

            """=====================intra event encoding (to capture states)====================="""
            # text_graph_data, token2nodepos=text_graph["graph"].to(text_embeddings.device), text_graph["token2nodepos"].to(text_embeddings.device)
            text_graph_data, token2nodepos = text_graphs["graph"].to(text_embeddings.device), text_graphs["token2nodepos"].to(text_embeddings.device)

            tmp_text_embeddings=text_embeddings
            if self.use_token_tag:
                text_embeddings = torch.cat([text_embeddings, self.tag_embeddings(text_graphs["token_tags_batch"].to(input_ids))], dim=-1)
                text_embeddings=self.tag_transform(text_embeddings)


            factor_embeddings = encode_factors(text_embeddings, input_ids, text_graphs)

            # tmp_dict = text_graphs.copy()
            # for key in ['idxs']:  # 'max_token_num','max_token_len',
            #     tmp_dict[key] = text_graphs[key].to(input_ids)
            # for key in ['masks']:  # orig mask is whether this verb at a timespoint  exists
            #     tmp_dict[key] = text_graphs[key].to(text_embeddings)
            # factor_embeddings = self.encode_gather(text_embeddings, input_ids, tmp_dict['idxs'],
            #                                       tmp_dict['masks'], tmp_dict['max_token_num'], tmp_dict['max_token_len'])
            # tmp_dict = ent.copy()
            # for key in ['mention_idxs', 'verb_idxs', 'token2nodepos']:  # 'max_token_num','max_token_len',
            #     tmp_dict[key] = ent[key].to(input_ids)
            # for key in ['verb_masks', 'mention_masks', 'mention_existence_mask', 'verb_existence_mask']:  # orig mask is whether this verb at a timespoint  exists
            #     tmp_dict[key] = ent[key].to(text_embeddings)

            factor_embeddings = index_select_from_embeddings(factor_embeddings, lens=text_graphs['num_nodes'], batch_embeddings=True, merge=True)
            factor_embeddings = self.graph_encoder(factor_embeddings, text_graph_data)
            token_embeddings = extract_unmasked_embeddings(factor_embeddings, token2nodepos)  # each token from original text
            # summ += token_embeddings

            # propagate with transformer
            if self.propagate_factor_embeddings in [1, 3]:
                text_embeddings = self.propagator_1(token_embeddings + text_embeddings) #, src_key_padding_mask=torch.as_tensor(1 - attn_mask, dtype=torch.bool)
            elif self.propagate_factor_embeddings in [2]:
                text_embeddings = token_embeddings + text_embeddings

            """=====================inter entity encoding====================="""

            # each entity_list is a batch of entity ranges
            # each sample is a batch of entity ranges
            tmp_embed = torch.zeros_like(text_embeddings)
            for j, sample in enumerate(entities):  # good for memory since entity_embeddings replaced each turn, put all samples together make entity embeddings too large

                # way1 from token embeddings
                entity_embeddings = encode_factors(text_embeddings[j].unsqueeze(0), input_ids[j].unsqueeze(0), sample)  # num_entitiesxNxD , # single_batch_multi_target=True
                entity_embeddings = entity_embeddings.view(sample["num_entities"], sample["max_mention_num"], -1)

                ## factor_embeddings = encode_factors(token_embeddings, input_ids, sample)
                ## factor_embeddings = factor_embeddings * (sample['mention_existence_mask'])

                # # way2 select node from graph
                # g_indices = torch.tensor([i for i, bl in enumerate(text_graph_data.batch == j) if bl], dtype=torch.long, device=factor_embeddings.device)
                # entity_embeddings = torch.index_select(factor_embeddings, 0, g_indices)
                # entity_embeddings =index_select_from_embeddings(entity_embeddings, indices_list=sample['mention_node_idx'], batch_embeddings=False, merge=False)

                # has batch index in entity list
                entity_embeddings = self.factor_transform(entity_embeddings)

                # seq_lens=torch.as_tensor(sample['lens'], dtype=torch.long, device=entity_embeddings.device)
                lstm_in = R.pack_padded_sequence(entity_embeddings, sample['num_mentions'], batch_first=True, enforce_sorted=False)
                lstm_out = self.lstm(lstm_in)[0]
                lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
                lstm_out = self.dropout_e(lstm_out)
                lstm_out = self.fc2_1(lstm_out)

                # B, N, D = lstm_out.shape
                # lstm_out=lstm_out.view(B * N, D)
                # there is sample['token2nodeid'] for each entity
                # updated_embeddings = extract_unmasked_embeddings(lstm_out, sample['token2nodeid'], batch_embeddings=True)
                # tmp_embed[j] += updated_embeddings.sum(0)

                """each token2nodepos is for an entity"""
                token2nodepos = sample['token2nodepos'].to(text_embeddings.device)
                updated_embeddings = extract_unmasked_embeddings(lstm_out.view(sample["num_entities"] * sample["max_mention_num"], -1), token2nodepos.unsqueeze(0))
                tmp_embed[j] += updated_embeddings.squeeze(0)
                # for k, token2nodepos in enumerate(sample['token2nodepos']):
                #     # token2nodepos=token2nodepos.to(lstm_out.device)
                #     updated_embeddings = extract_unmasked_embeddings(lstm_out[k].unsqueeze(0), token2nodepos.unsqueeze(0), batch_embeddings=True)
                #     tmp_embed[j] += updated_embeddings.squeeze(0)

            """=====================inter event encoding====================="""
            # Currently in above
            #
            # trigger_embeddings = index_select_from_embeddings(factor_embeddings, indices_list=triggers['trigger_node_idx'], batch_embeddings=False)
            # trigger_embeddings = self.trigger_transform(trigger_embeddings)
            #
            # lstm_in = R.pack_padded_sequence(trigger_embeddings, torch.as_tensor(triggers['lens'], dtype=torch.long), batch_first=True, enforce_sorted=False)
            # lstm_out = self.lstm_evt(lstm_in)[0]
            # lstm_out, _ = R.pad_packed_sequence(lstm_out, batch_first=True)
            # lstm_out = self.dropout_e(lstm_out)
            # lstm_out = self.fc3_1(lstm_out)
            #
            # summ += extract_unmasked_embeddings(lstm_out, triggers['token2nodepos'], batch_embeddings=True)
            if self.use_token_tag:
                tmp_embed=self.transform_temporal_back(tmp_embed)
                token_embeddings=self.transform_node_back(token_embeddings)

            # propagate with transformer
            if self.propagate_factor_embeddings in [2, 3]:
                summ = text_embeddings + tmp_embed
                summ = self.propagator_2(summ, src_key_padding_mask=torch.as_tensor(1 - attn_mask, dtype=torch.bool))
            elif self.propagate_factor_embeddings == 1:
                summ = tmp_embed + text_embeddings
            elif self.propagate_factor_embeddings == 0:
                if self.cat_text_embed:
                    summ=self.transform_cat_text(torch.cat([tmp_text_embeddings, token_embeddings, tmp_embed], dim=-1))
                    summ+=tmp_text_embeddings

                else:
                    summ = tmp_text_embeddings # text_embeddings
                    if self.ablation!=1:
                        summ = summ + tmp_embed
                    if self.ablation!=2:
                        summ = summ + token_embeddings
            elif self.propagate_factor_embeddings in [4]:
                summ = tmp_embed + token_embeddings + text_embeddings




                summ = self.propagator_1(summ, src_key_padding_mask=torch.as_tensor(1 - attn_mask, dtype=torch.bool))

            return summ  # text_embeddings + tmp_embed

        # final_vec = [text_embeddings]
        #
        # """======Final======"""
        # # print("final")
        # if len(final_vec) > 1:
        #     logits = self.fc3(torch.cat(final_vec, dim=-1))  # concatenated B*seqlen*(plm_hidden_dim+2xg_dim), output B*seqlen*plm_hidden_dim
        # else:
        #     logits = final_vec[0]
        # # print("\nlogits", get_tensor_info(logits))
        #
        # return logits


class _EventReasoningModule(nn.Module):

    def __init__(self, args=None, pretrained_concept_emb=None):
        super().__init__()
        # print("args.g_dim", args.g_dim)
        self.components = args.components
        self.pool_type = args.pool_type
        self.batch_size = args.batch_size

        # self.mlp = Linear(args.plm_hidden_dim * 2, args.plm_hidden_dim)

        self.extra_emb = nn.Embedding(2, args.g_dim, 0)
        # self.rand_emb = nn.Embedding(1, args.plm_hidden_dim)
        # self.the_zero = nn.Embedding(1, args.plm_hidden_dim)
        # self.rand_emb.weight.data.fill_(0)
        # init.constant(self.rand_emb.weight.data, val=.0)

        self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device, requires_grad=False)
        self.the_one = torch.tensor(1, dtype=torch.long, device=args.device, requires_grad=False)
        self.the_two = torch.tensor(2, dtype=torch.long, device=args.device, requires_grad=False)
        self.zero_one_indicies = torch.tensor([0, 1], dtype=torch.long, device=args.device, requires_grad=False)
        # self.zero_vec = torch.tensor([0]*args.g_dim, dtype=torch.long, device=args.device, requires_grad=False)
        self.zero_vec = torch.tensor([0] * args.g_dim, dtype=torch.long, device=args.device, requires_grad=False).unsqueeze(0)

        # print("args.device", args.device)
        # print("self.the_one", self.the_one, get_tensor_info(self.the_one))
        # print("self.the_two", self.the_two, get_tensor_info(self.the_two))
        # print("self.the_zero", self.the_zero, get_tensor_info(self.the_zero))
        # print("self.zero_one_indicies", self.zero_one_indicies, get_tensor_info(self.zero_one_indicies))

        if "wm" in self.components:
            # self.wm = KnowMemSpace(args, pretrained_concept_emb=pretrained_concept_emb)
            self.wm = GNN(args, gnn_type=args.gnn_type, num_gnn_layers=args.num_gnn_layers,
                          encode_node_features=True, encode_edge_features=False,
                          pool_type=args.pool_type, global_pooling=False, pretrained_concept_emb=pretrained_concept_emb)

        if "fstm" in self.components:
            # print("=args.gnn_type2", args.gnn_type2)
            self.fstm = GNN(args, input_dim=args.g_dim, gnn_type=args.gnn_type2, num_gnn_layers=args.num_gnn_layers2,
                            # input_dim=args.plm_hidden_dim,
                            pool_type=args.pool_type, global_pooling=False, )
        self.activation = GELU()
        # self.pooler = MultiheadAttPoolLayer(args.n_attention_head, args.plm_hidden_dim, args.g_dim)

        # self.fc = MLP(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim, args.plm_hidden_dim, 1, dropout=args.dropoutf, layer_norm=True)
        # self.fc=Linear(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim)

        self.fc1 = Linear((args.g_dim + args.plm_hidden_dim) if "wm" in self.components else args.plm_hidden_dim, args.g_dim)  # args.plm_hidden_dim
        # self.fc2=Linear(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim)

        final_dim = args.plm_hidden_dim
        if "wm" in self.components:
            final_dim += args.g_dim
        if "fstm" in self.components:
            final_dim += args.g_dim
        self.fc3 = Linear(final_dim, args.plm_hidden_dim)

        self.dropout_e = nn.Dropout(args.dropouti)
        self.dropout_fc = nn.Dropout(args.dropoutf)
        self.plm_hidden_dim = args.plm_hidden_dim
        self.init_range = args.init_range
        if args.init_range > 0:
            self.apply(self._init_weights2)
        # if args.init_range > 0:
        #     self.apply(self._init_weights)
        # self.init_range = args.init_range
        # if args.init_range > 0:
        #     self.apply(self._init_weights)

    def _init_weights2(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, text_embeddings=None, g_data=None, g_data2=None, token2nodepos=None):
        """

        :param text_embeddings:
        :param g_data: working memory graph, num_nodes_in_a_batch
        :param g_data2: fstm graph, node: Bxseqlen *node_dim, edge_index: 2*num_edges
        :param token2nodepos: B x max_seq_len_in_batch * max_concepts_each_token
        :return:
        """

        """======WM======"""
        # print("\n\nsubmodule fwd")
        # print("g_data", g_data)
        # print("g_data x", g_data.x.shape)
        # print("g_data batch", g_data.batch)
        # print("g_data2", g_data2.x.shape)

        # print("self.the_one", self.the_one, get_tensor_info(self.the_one))
        # print("self.the_two", self.the_two, get_tensor_info(self.the_two))
        # print("self.the_zero", self.the_zero, get_tensor_info(self.the_zero))
        # print("self.zero_one_indicies", self.zero_one_indicies, get_tensor_info(self.zero_one_indicies))
        # print("self.extra_emb", self.extra_emb.weight)
        # print("the_two", get_tensor_info(self.the_two))
        # print("the_one", get_tensor_info(self.the_one))
        # print("the_zero", get_tensor_info(self.the_zero))
        # print("zero_one_indicies", get_tensor_info(self.zero_one_indicies))

        # self.zero_one_indicies=self.zero_one_indicies.cuda()

        # print("wm")
        # print("self.zero_vec", self.zero_vec)
        final_vec = [text_embeddings]
        # print("text_embeddings", get_tensor_info(text_embeddings))

        if "wm" in self.components:
            # prevent zero node
            if g_data.x is not None and g_data.x.shape[0] > 0:

                graph_embeddings = self.wm(g_data.x, g_data)  # num_nodes_in_a_batch * node_dim
                # print("graph_embeddings", get_tensor_info(graph_embeddings))
                # embed()
                # graph_embeddings_with_padding_embeddings = torch.cat([self.zero_vec , graph_embeddings], #self.extra_emb(self.zero_one_indicies)
                #                                                      dim=0)  # num_nodes_in_a_batch * node_dim

                graph_embeddings_with_padding_embeddings = torch.cat([self.extra_emb(self.zero_one_indicies), graph_embeddings],  # self.extra_emb(self.zero_one_indicies)
                                                                     dim=0)  # num_nodes_in_a_batch * node_dim
                # try:
                #     # print("in self.extra_emb( ")
                #     graph_embeddings_with_padding_embeddings = torch.cat([self.extra_emb(self.zero_one_indicies), graph_embeddings],
                #                                                          dim=0)  # num_nodes_in_a_batch * node_dim
                # except:
                #     print("in self.extra_emb( ")
                #     embed()
            else:
                # print("zero nodes")
                # graph_embeddings_with_padding_embeddings = torch.cat([self.zero_vec])#
                graph_embeddings_with_padding_embeddings = self.extra_emb(self.zero_one_indicies)  # self.extra_emb(self.zero_one_indicies)
            # print("here")
            # print("extra embed 0", self.extra_emb(0))
            # print("graph_embeddings_with_padding_embeddings", get_tensor_info(graph_embeddings_with_padding_embeddings))
            # self.the_zero(torch.tensor([])).unsqueeze(0), self.rand_emb(0).unsqueeze(0)

            tmp = torch.tensor(0, dtype=torch.long, device=text_embeddings.device)
            for i in range(token2nodepos.shape[-1]):
                indicies = token2nodepos[:, i] + self.the_two  # torch.tensor(2, dtype=torch.long, device=text_embeddings.device)

                # print("indicies", get_tensor_info(indicies))
                tmp = tmp + torch.index_select(graph_embeddings_with_padding_embeddings, 0, indicies)  # Bxseqlen * g_dim

                # try:
                #     tmp = tmp + torch.index_select(graph_embeddings_with_padding_embeddings, 0, indicies)  # Bxseqlen * g_dim
                # except:
                #     print("index sel")
                #     embed()
            # print("tmp0", get_tensor_info(tmp))

            graph_text_embed = self.dropout_fc(self.fc1(torch.cat(
                [tmp, text_embeddings.view(text_embeddings.shape[0] * text_embeddings.shape[1], -1)], dim=-1)))  # Bxseqlen * g_dim
            tmp = tmp.view(text_embeddings.shape[0], text_embeddings.shape[1], tmp.shape[-1])  # B * seqlen * g_dim

            final_vec.append(tmp)
        else:
            graph_text_embed = self.dropout_fc(self.fc1(text_embeddings))
        # print("tmp", get_tensor_info(tmp))

        """======FSTM======"""

        if "fstm" in self.components:
            # print("FSTM")
            # print("g_data2.x pre", get_tensor_info(g_data2.x))
            x = torch.index_select(graph_text_embed, 0, g_data2.x)  # Bxseqlen * g_dim

            temporal_graph_embeddings = self.fstm(x, g_data2)  # Bxseqlen * g_dim
            # print("temporal_graph_embeddings0", get_tensor_info(temporal_graph_embeddings))
            temporal_graph_embeddings = temporal_graph_embeddings.view(text_embeddings.shape[0], text_embeddings.shape[1],
                                                                       temporal_graph_embeddings.shape[
                                                                           -1])  # B * seqlen * g_dim
            final_vec.append(temporal_graph_embeddings)
        # print("temporal_graph_embeddings", get_tensor_info(temporal_graph_embeddings))
        # graph_text_embed2=self.dropout_fc(self.fc2(torch.cat([temporal_graph_embeddings, text_embeddings])))

        """======Final======"""
        # print("final")
        logits = self.fc3(torch.cat(final_vec, dim=-1))  # concatenated B*seqlen*(plm_hidden_dim+2xg_dim), output B*seqlen*plm_hidden_dim
        # print("\nlogits", get_tensor_info(logits))

        return logits

# class FET(torch.nn.Module):
#     def __init__(self, args):
#         super(FET, self).__init__()
#
#         self.components = args.components
#         self.num_labels = args.out_dim
#
#         # self.plm = AutoE.from_pretrained(args.plm)
#         self.plm = BartEncoder.from_pretrained(args.plm)
#
#         self.combiner = Linear(args.plm_hidden_dim, args.out_dim)
#
#         self.dropout = args.dropout
#
#         # self.loss = torch.nn.CrossEntropyLoss()
#         self.loss = torch.nn.MultiLabelSoftMarginLoss()
#         # self.loss = nn.BCEWithLogitsLoss()
#
#         self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
#         self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)
#
#     def add_modules(self, args, pretrained_concept_emb):
#         self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
#         ##* donwstream, so init differently
#     def forward(self, input_ids=None,
#                 attention_mask=None,
#                 token_type_ids=None,
#                 labels=None,
#                 ids=None,
#                 in_train=None,
#                 print_output=False,
#                 inputs=None,
#                 ):
#         final_vec = []
#
#         "=========Original Text Encoder=========="
#         # embed()
#         # print("texts", get_tensor_info(texts))
#         # print("texts_attn_mask", get_tensor_info(texts_attn_mask))
#         # hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask,
#         #                          return_dict=True).last_hidden_state[:,0,:]#.pooler_output
#         hidden_states = self.plm(input_ids=input_ids, attention_mask=attention_mask,
#                                  return_dict=True).last_hidden_state[:,0,:]#.pooler_output
#         # print("hidden_states", get_tensor_info(hidden_states))
#         hidden_states=self.submodule_1()
#         final_vec.append(hidden_states)
#
#         # if "co" in self.components:
#         #     hidden_states_context_only = self.plm(input_ids=masked_input_ids,
#         #                                           attention_mask=masked_texts_attention_mask,
#         #                                           return_dict=True).last_hidden_state
#         #     final_vec.append(hidden_states_context_only)
#
#         "=========Classification=========="
#         output = self.combiner(torch.cat(final_vec, dim=-1))
#         # print("output", get_tensor_info(output))
#
#
#         # print("output", output)
#         return {
#             "logits": output
#         }
#
#         if in_train:
#             # label smoothing
#             # return self.criterion(output, labels)
#             return self.loss(output, labels)
#             return torch.nn.functional.cross_entropy(output, labels)
#             return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
#         pred_out = (torch.sigmoid(output) > 0.5).float()
#         # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
#         # print("sigmoid output")
#         # pp(torch.sigmoid(output))
#         # print("pred_out")
#         # pp(pred_out)
#         # pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
#         # print('pred_out', get_tensor_info(pred_out))
#         return output
#         return pred_out

#
# class FET(torch.nn.Module):
#     def __init__(self, args, tokenizer):
#         super().__init__() #FET, self
#
#         # self.model_type = args.model_type
#         # self.plm = BertModel.from_pretrained(args.plm)
#         self.plm = BertModel.from_pretrained(args.plm)
#         print('len(tokenizer)',len(tokenizer))
#
#         # if self.plm.get_input_embeddings().weight.shape[0]!=len(tokenizer):
#         #     self.plm.resize_token_embeddings(len(tokenizer))
#
#
#
#         self.dropout = args.dropout
#
#         # self.loss = torch.nn.CrossEntropyLoss()
#         # self.loss = nn.MultiLabelSoftMarginLoss()
#         self.loss = nn.BCEWithLogitsLoss()
#         print("args.plm_hidden_dim", args.plm_hidden_dim)
#         final_dim=args.plm_hidden_dim
#         self.use_additional_module=False
#         if args.model_name=="see":
#             self.use_additional_module=True
#             self.tf=TransformerEncoderLayer(d_model=args.plm_hidden_dim, nhead=8, batch_first=True)
#             final_dim = args.plm_hidden_dim*2
#
#         self.combiner = Linear(final_dim, 1)#$args.out_dim
#         self.the_zero = torch.tensor(0, dtype=torch.long, device=args.device)
#         self.the_one = torch.tensor(1, dtype=torch.long, device=args.device)
#
#     def add_modules(self, args, pretrained_concept_emb):
#         self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
#         ##* donwstream, so init differently
#     def forward(self, input, args):
#         texts = input.texts
#         texts_attn_mask = input.texts_attn_mask
#
#         if self.use_additional_module:
#             g_data = input.g_data
#             g_data2 = input.g_data2
#             token2nodepos = input.token2nodepos  # .cuda()#.to(attention_mask)
#             event_position_ids = input.event_position_ids  # .cuda()#.to(attention_mask)
#
#         tgt_txt=input.tgt_txt
#         labels = input.labels
#         in_train = input.in_train
#
#         final_vec = []
#         "=========Original Text Encoder=========="
#         embed()
#         hidden_states = self.plm(input_ids=texts, attention_mask=texts_attn_mask, return_dict=True).last_hidden_state
#         # if "evttag" in self.submodule_1.components: hidden_states+=self.embed_positions.forward_event_tag(event_position_ids)
#         print("get_tensor_info(hidden_states) bef",get_tensor_info(hidden_states))
#         if self.use_additional_module:
#             hidden_states = self.submodule_1(hidden_states, g_data, g_data2, token2nodepos)  # (batch size, dim_mode)
#             print("get_tensor_info(hidden_states) mid",get_tensor_info(hidden_states))
#
#             hidden_states=F.gelu(hidden_states)
#             hidden_states=self.tf(hidden_states)
#             print("get_tensor_info(hidden_states) after",get_tensor_info(hidden_states))
#         final_vec.append(hidden_states[:, 0, :])
#
#         "=========Target Next Step=========="
#         hid_tgt_txt= self.plm(**tgt_txt, return_dict=True).last_hidden_state[:, 0, :]
#         print("hid_tgt_txt",get_tensor_info(hid_tgt_txt))
#         final_vec.append(hid_tgt_txt)
#
#         "=========Classification=========="
#         output = self.combiner(torch.cat(final_vec, dim=-1))
#         print("output",get_tensor_info(output))
#         output=output.squeeze(-1)
#         print("output aft sq",get_tensor_info(output))
#
#         if in_train:
#             # label smoothing
#             # return self.criterion(output, labels)
#             return self.loss(output, labels)
#             return torch.nn.functional.cross_entropy(output, labels)
#             return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
#         pred_out = (torch.sigmoid(output) > 0.5).float()
#         # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)
#         # print("sigmoid output")
#         # pp(torch.sigmoid(output))
#         # print("pred_out")
#         # pp(pred_out)
#         # pred_out = torch.argmax(torch.softmax(output, dim=-1), dim=-1)
#         # print('pred_out', get_tensor_info(pred_out))
#         return pred_out


# class EventReasoningEncoder(ModifiedBertModel):
#     # def __init__(self, configs):
#     #     super().__init__(configs)
#     #
#     #     # print("in init")
#     #     # print("configs", configs)
#     #     # bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
#     #
#     # def __init__(self, config, add_pooling_layer=True):
#     #     super().__init__(config)
#     #     self.config = config
#     #
#     #     self.embeddings = ModifiedBertEmbeddings(config)
#     #     self.encoder = BertEncoder(config)
#     #
#     #     self.pooler = BertPooler(config) if add_pooling_layer else None
#     #
#     #     self.init_weights()
#     #
#     # def get_input_embeddings(self):
#     #     return self.embeddings.word_embeddings
#     #
#     # def set_input_embeddings(self, value):
#     #     self.embeddings.word_embeddings = value
#     #
#     # def _prune_heads(self, heads_to_prune):
#     #     """
#     #     Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
#     #     class PreTrainedModel
#     #     """
#     #     for layer, heads in heads_to_prune.items():
#     #         self.encoder.layer[layer].attention.prune_heads(heads)
#     def add_modules(self, args, pretrained_concept_emb):
#
#         self.submodule_1 = EventReasoningModule(args, pretrained_concept_emb)
#         # self.embeddings=ModifiedBertEmbeddings()
#         if args.init_range > 0:
#             self.apply(self._init_weights2)
#
#     #     self.init_range = args.init_range
#     #     if args.init_range > 0:
#     #         self.apply(self._init_weights)
#     def _init_weights2(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=self.init_range)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#
#     def forward(self,
#                 input_ids=None,
#                 attention_mask=None,
#                 token_type_ids=None,
#                 position_ids=None,
#                 head_mask=None,
#                 inputs_embeds=None,
#                 encoder_hidden_states=None,
#                 encoder_attention_mask=None,
#                 past_key_values=None,
#                 use_cache=None,
#                 output_attentions=None,
#                 output_hidden_states=None,
#                 return_dict=None,
#                 extra_inputss=None):
#         print("in bert forward pre")
#
#         print("past_key_values", past_key_values)
#         print("position_ids", position_ids)
#         g_data = extra_inputss['g_data']
#         g_data2 = extra_inputss['g_data2']
#         token2nodepos = extra_inputss['token2nodepos']
#         event_position_ids = extra_inputss['event_position_ids']
#
#         g_data = g_data.to(attention_mask)
#         g_data2 = g_data2.to(attention_mask)
#         token2nodepos = token2nodepos.to(attention_mask)
#         event_position_ids = event_position_ids.to(attention_mask)
#         # print("g_data", g_data)
#         # gc.collect()
#         # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         # output_hidden_states = (
#         #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         # )
#         # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         #
#         # if self.config.is_decoder:
#         #     use_cache = use_cache if use_cache is not None else self.config.use_cache
#         # else:
#         #     use_cache = False
#         #
#         # if input_ids is not None and inputs_embeds is not None:
#         #     raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         # elif input_ids is not None:
#         #     input_shape = input_ids.size()
#         #     batch_size, seq_length = input_shape
#         # elif inputs_embeds is not None:
#         #     input_shape = inputs_embeds.size()[:-1]
#         #     batch_size, seq_length = input_shape
#         # else:
#         #     raise ValueError("You have to specify either input_ids or inputs_embeds")
#         #
#         # device = input_ids.device if input_ids is not None else inputs_embeds.device
#         #
#         # # past_key_values_length
#         # past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
#         #
#         # if attention_mask is None:
#         #     attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
#         # if token_type_ids is None:
#         #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
#         #
#         # # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # # ourselves in which case we just need to make it broadcastable to all heads.
#         # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
#         #
#         # # If a 2D or 3D attention mask is provided for the cross-attention
#         # # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         # if self.config.is_decoder and encoder_hidden_states is not None:
#         #     encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#         #     encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#         #     if encoder_attention_mask is None:
#         #         encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#         #     encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         # else:
#         #     encoder_extended_attention_mask = None
#         #
#         # # Prepare head mask if needed
#         # # 1.0 in head_mask indicate we keep the head
#         # # attention_probs has shape bsz x n_heads x N x N
#         # # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
#         #
#         # embedding_output = self.embeddings(
#         #     input_ids=input_ids,
#         #     position_ids=position_ids,
#         #     token_type_ids=token_type_ids,
#         #     inputs_embeds=inputs_embeds,
#         #     past_key_values_length=past_key_values_length,
#         #     event_position_ids=event_position_ids,
#         # )
#         # encoder_outputs = self.encoder(
#         #     embedding_output,
#         #     attention_mask=extended_attention_mask,
#         #     head_mask=head_mask,
#         #     encoder_hidden_states=encoder_hidden_states,
#         #     encoder_attention_mask=encoder_extended_attention_mask,
#         #     past_key_values=past_key_values,
#         #     use_cache=use_cache,
#         #     output_attentions=output_attentions,
#         #     output_hidden_states=output_hidden_states,
#         #     return_dict=return_dict,
#         # )
#         # sequence_output = encoder_outputs[0]
#         # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
#         #
#         # if not return_dict:
#         #     return (sequence_output, pooled_output) + encoder_outputs[1:]
#         #
#         # prior_result = BaseModelOutputWithPoolingAndCrossAttentions(
#         #     last_hidden_state=sequence_output,
#         #     pooler_output=pooled_output,
#         #     past_key_values=encoder_outputs.past_key_values,
#         #     hidden_states=encoder_outputs.hidden_states,
#         #     attentions=encoder_outputs.attentions,
#         #     cross_attentions=encoder_outputs.cross_attentions,
#         # )
#
#         prior_result = super(EventReasoningEncoder, self).forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             event_position_ids=event_position_ids)
#
#         "after"
#         """try change device for dataparaell"""
#         # g_data=g_data.cuda()
#
#         """prior_result"""
#
#         # todo:add sep positions
#
#         text_embeddings = prior_result.last_hidden_state
#         # print("\ntext_embeddings", get_tensor_info(text_embeddings))
#
#         prior_result.last_hidden_state = self.submodule_1(text_embeddings, g_data, g_data2,
#                                                           token2nodepos)  # (batch size, dim_mode)
#         # prior_result.last_hidden_state = text_embeddings #debugging
#
#         return prior_result
#
#         # graph_embeddings = graph_embeddings.unsqueeze(1).repeat(1, text_embeddings.shape[1], 1)
#         # print("\ngraph_embeddings new", get_tensor_info(graph_embeddings))
#
#         # graph_embeddings, pool_attn = self.pooler(text_embeddings, graph_embeddings)
#         # print("\ngraph_embeddings pooled", get_tensor_info(graph_embeddings))
#
#         # graph_embeddings = self.activation(graph_embeddings)
#
#         # concat = self.dropout_fc(torch.cat((graph_embeddings, text_embeddings), dim=-1))
#         # print("\nconcat", get_tensor_info(concat))
#
#         # logits = self.fc(concat)
#         # print("\nlogits", get_tensor_info(logits))
#
#         # out = self.mlp(torch.cat([text_embeddings, graph_embeddings], dim=-1))
#         # self.mlp = Linear(args.plm_hidden_dim*2, args.plm_hidden_dim)
#         # self.msg_passer = KnowMemSpace(args, pretrained_concept_emb=pretrained_concept_emb)
#         # self.activation = GELU()
#         # self.pooler = MultiheadAttPoolLayer(args.n_attention_head, args.plm_hidden_dim, args.concept_dim)
#         #
#         # self.fc = MLP(args.g_dim + args.plm_hidden_dim, args.plm_hidden_dim, 1, 1, args.dropoutf, layer_norm=True)
#         #
#         # self.dropout_e = nn.Dropout(args.dropouti)
#         # self.dropout_fc = nn.Dropout(args.dropoutf)
#         # print("position_ids", position_ids)
#         # print("head_mask", head_mask)
#         # print("inputs_embeds", inputs_embeds)
#         # print("encoder_hidden_states", encoder_hidden_states)
#         # print("encoder_attention_mask", encoder_attention_mask)
#         # print("use_cache", use_cache)
#
#         # print("prior_result.last_hidden_state", get_tensor_info(prior_result.last_hidden_state))
#         # print("pooler_output", prior_result.pooler_output)
#         # print("past_key_values", prior_result.past_key_values)
#         # print("hidden_states", prior_result.hidden_states)
#         # print("attentions", prior_result.attentions)
#         # print("cross_attentions", prior_result.cross_attentions)
#         # graph_embe
#         # concept_embeddings = super(EventReasoningEncoder, self).forward(
#         #     input_ids=concept_input_ids,
#         #     attention_mask=concept_attention_mask,
#         #     position_ids=position_ids,
#         #     head_mask=head_mask,
#         #     inputs_embeds=inputs_embeds,
#         #     encoder_hidden_states=encoder_hidden_states,
#         #     encoder_attention_mask=encoder_attention_mask,
#         #     past_key_values=past_key_values,
#         #     use_cache=use_cache,
#         #     output_attentions=output_attentions,
#         #     output_hidden_states=output_hidden_states,
#         #     return_dict=return_dict, )
#         # print("\nconcept_embeddings", get_tensor_info(concept_embeddings))
#         #
#         #
#         # g_data.x=concept_embeddings
#         # graph_embeddings = self.msg_passer(g_data)
#         # graph_embeddings = self.activation(graph_embeddings)
#         # print("\ngraph_embeddings", get_tensor_info(graph_embeddings))
#         #
#         # out = self.mlp(torch.cat([text_embeddings, graph_embeddings], dim=-1))
#         # print("\nout", get_tensor_info(out))
#         #
#         # return out
#
#         # = self.b(prior_result.last_hidden_state)
#
#         # embed()
#         # BaseModelOutputWithPastAndCrossAttentions(
#         #     last_hidden_state=sequence_output,
#         #     past_key_values=encoder_outputs.past_key_values,
#         #     hidden_states=encoder_outputs.hidden_states,
#         #     attentions=encoder_outputs.attentions,
#         #     cross_attentions=encoder_outputs.cross_attentions,
#         # )
#         # print(prior_result)
#         # embed()
#         #
#         # print("prior_result.last_hidden_state", get_tensor_info(prior_result.last_hidden_state))
#         # print("prior_result.hidden_states", get_tensor_info(prior_result.hidden_states))
#         # print("prior_result.past_key_values",prior_result.past_key_values)
#         # print("prior_result.last_hidden_state",prior_result.last_hidden_state)
#         # print("prior_result.hidden_states",prior_result.hidden_states)
#         # embed()
### Training
# bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
#
# a = BertGenerationEncoder.from_pretrained("bert-base-uncased", bos_token_id=101, eos_token_id=102)
#
# b = baseline.from_pretrained("bert-base-uncased", bos_token_id=101, eos_token_id=102)


# # batch_size = text_embeddings.shape[0]
# # idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
# # idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
# # masks = text_embeddings.new(masks).unsqueeze(-1)
# # text_embeddings = torch.gather(text_embeddings, 1, idxs) * masks
# # text_embeddings = text_embeddings.view(batch_size, token_num, token_len, self.bert_dim)
# # text_embeddings = text_embeddings.sum(2)  # max_seq_len in words
#
# # word_embeddings=self.encode(text_embeddings, piece_idxs, token_lens) # has 0s for paddings sents
# nb_word_embeddings = self.encode(nb_text_embeddings, piece_idxs, token_lens)
# # sim_score=torch.sigmoid(self.sim_fc(torch.cat([text_embeddings[:,0,:], nb_text_embeddings[:,0,:]], dim=-1)))
#
#
# """Token2Node"""
# token2node_idx = None
#
# """====Unwrap the concatenated original and neighbour embeddings===="""
#
# text_embeddings = text_embeddings.view(text_embeddings.shape[0] * text_embeddings.shape[1], -1)
#
# # node  gather
#
#
# # graph is Bxl
# graph_emb = self.gnn(word_embeddings, g_data)
# nb_graph_emb = self.gnn(nb_word_embeddings, nb_g_data)  # .transpose(1,0)
#
# attended_embs = []
# attended_emb_cls = []
# for batch_id in range(text_embeddings.shape[0]):
#     # for g1, g2 in zip(graph_emb, piece_to_word_mask)
#     g_indices = [i for i, bl in enumerate(g_data.batch == batch_id) if bl]
#     nb_g_indices = [i for i, bl in enumerate(nb_g_data.batch == batch_id) if bl]
#
#     if g_indices and len(nb_g_indices):
#         cur_graph_emb = torch.index_select(graph_emb, 0,
#                                            torch.tensor(g_indices, dtype=torch.long, device=text_embeddings.device))
#         cur_nb_graph_emb = torch.index_select(nb_graph_emb, 0, torch.tensor(nb_g_indices, dtype=torch.long,
#                                                                             device=text_embeddings.device))
#
#         attended_emb = self.multihead_attn(cur_graph_emb.unsqueeze(0).transpose(1, 0),
#                                            cur_nb_graph_emb.unsqueeze(0).transpose(1, 0),
#                                            cur_nb_graph_emb.unsqueeze(0).transpose(1, 0), need_weights=False)
#         attended_emb = attended_emb.transpose(1, 0).squeeze(0)
#         attended_embs.append(attended_emb)  # , attn_output_weights
#         attended_emb_cls.append(torch.mean(attended_emb, dim=0))
#
#     else:
#         assert False
# attended_embs = torch.cat(attended_embs, dim=0)
#
# # attended_emb =self.multihead_attn(graph_emb, nb_graph_emb, nb_graph_emb, need_weights=False)#, attn_output_weights
# # attended_emb=attended_emb.transpose(1,0)
# # print('attended_emb', attended_emb.shape)
#
# sim_score = torch.sigmoid(self.sim_fc(torch.cat([text_embeddings[:, 0, :], attended_emb_cls], dim=-1)))
# # sim_score=torch.sigmoid(self.sim_fc(torch.cat([text_embeddings[:,0,:], torch.mean(attended_emb, dim=1)], dim=-1)))
#
# # attended_emb cat with
# # zero_mask, Bxl, for token that doesn't belong to any node
# # token2node mask Bxl, which node the token belongs to
# token_zero_mask = None
# token_to_node_mask = None
# attended_text_embeddings = sim_score * torch.index_select(attended_embs, 0, token_to_node_mask)
# attended_text_embeddings * token_to_node_mask + (1 - token_to_node_mask) * self.the_zero  # 0 are no node reference
#
#
# # attended_emb=torch.stack([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(attended_emb, piece_to_word_mask)])
#
# # res=sim_score*torch.index_select(attended_emb, 0, torch.tensor(attended_emb, dtype=torch.long, device=text_embeddings.device))
#
#
# ## readd to encoder
#
#
# ##
#
