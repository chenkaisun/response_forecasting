import torch.nn
from torch.nn import Embedding, ModuleList
from torch.nn import Tanh
from utils.utils import module_exists
if module_exists("torch_geometric"):
    # from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, GINConv
    from torch_geometric.nn import GINEConv, FastRGCNConv, RGCNConv
    from torch_scatter import scatter
from model.model_utils import *
from torch.nn import Sequential, Linear, BatchNorm1d, Dropout

class NodeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_types=26, num_features_to_encode=1):
        super(NodeEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(num_features_to_encode):
            self.embeddings.append(Embedding(num_node_types, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        # print("xshape",x.shape)
        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out


class EdgeEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_edge_types=32, num_features_to_encode=1):
        super(EdgeEncoder, self).__init__()

        self.embeddings = torch.nn.ModuleList()

        for i in range(num_features_to_encode):
            self.embeddings.append(Embedding(num_edge_types, hidden_channels))

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()

    def forward(self, edge_attr):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        out = 0
        for i in range(edge_attr.size(1)):
            out += self.embeddings[i](edge_attr[:, i])
        return out


# class MoleGraphConv(torch.nn.Module):
#     def __init__(self, args, is_first_layer=False):
#         super().__init__()
#         args.g_dim = args.plm_hidden_dim
#         self.gnn_type = args.gnn_type
#         self.is_first_layer = is_first_layer
#
#         hidden_channels, num_layers, dropout = args.g_dim, args.num_gnn_layers, args.dropout
#
#         self.num_layers = args.num_gnn_layers
#         self.dropout = args.dropout
#
#         if is_first_layer:
#             self.atom_encoder = AtomEncoder(args.g_dim)
#
#         self.bond_encoders = ModuleList()
#         self.atom_convs = ModuleList()
#         # self.atom_batch_norms = ModuleList()
#
#         self.bond_encoders.append(BondEncoder(hidden_channels))
#
#         if args.gnn_type == "gat":
#             self.atom_convs.append(GATConv(hidden_channels, hidden_channels))
#         elif args.gnn_type == "gine":
#             nn = Sequential(
#                 Linear(hidden_channels, 2 * hidden_channels),
#                 # BatchNorm1d(2 * hidden_channels),
#                 # ReLU(),
#                 # Tanh(),
#                 Tanh(),
#
#                 Linear(2 * hidden_channels, hidden_channels),
#             )
#             self.atom_convs.append(GINEConv(nn, train_eps=True))
#
#         # self.atom_lin = Linear(hidden_channels, hidden_channels)
#
#     def forward(self, x, data, global_pooling=False):
#         if self.is_first_layer:
#             x = self.atom_encoder(data.x.squeeze())
#
#         edge_attr = self.bond_encoders[0](data.edge_attr)
#         x = self.atom_convs[0](x, data.edge_index, edge_attr) if self.gnn_type == "gine" else self.atom_convs[0](x,
#                                                                                                                  data.edge_index)
#
#         x = torch.tanh(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#
#         return x
#         # if not global_pooling:
#         # #     x = self.atom_lin(x)
#         # #     x = torch.tanh(x)
#         #     return x
#         #
#         # x = scatter(x, data.batch, dim=0, reduce='mean')
#         # x = F.dropout(x, self.dropout, training=self.training)
#         # x = self.atom_lin(x)
#         # x = F.gelu(x)
#         # return x
class KnowMemSpace(torch.nn.Module):
    def __init__(self, args, pretrained_concept_emb=None):
        super().__init__()

        self.gnn_type = args.gnn_type
        self.num_layers = args.num_gnn_layers
        self.dropout = args.dropoutg

        # print("args.concept_num", args.concept_num)
        self.concept_emb = CustomizedEmbedding(concept_num=args.concept_num, concept_out_dim=args.g_dim,
                                               use_contextualized=False, concept_in_dim=args.concept_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=args.freeze_ent_emb)
        # self.svec2nvec = Linear(args.plm_hidden_dim, args.concept_dim)

        # self.node_encoder = NodeEncoder(args.g_dim)
        self.edge_encoders = ModuleList()
        self.g_convs = ModuleList()
        self.batch_norms = ModuleList()
        self.activation = Tanh()
        for _ in range(args.num_gnn_layers):
            if args.gnn_type == "gat":
                self.g_convs.append(GATConv(args.g_dim, args.g_dim))
            elif args.gnn_type == "gine":
                self.edge_encoders.append(EdgeEncoder(args.g_dim))
                nn = Sequential(
                    Linear(args.g_dim, 2 * args.g_dim),
                    # BatchNorm1d(2 * hidden_channels),
                    # ReLU(),
                    # Tanh(),
                    Tanh(),
                    Linear(2 * args.g_dim, args.g_dim),
                )
                self.g_convs.append(GINEConv(nn, train_eps=True))
            elif args.gnn_type == "rgcn":
                self.g_convs.append(FastRGCNConv(args.g_dim, args.g_dim, args.num_relation))
            elif args.gnn_type == "compgcn":
                self.g_convs.append(FastRGCNConv(args.g_dim, args.g_dim, args.num_relation))
            # self.atom_convs.append(GATConv(hidden_channels, hidden_channels))
            # self.atom_batch_norms.append(BatchNorm1d(hidden_channels))
        self.lin = Linear(args.g_dim, args.g_dim)

    def forward(self, data, data2, global_pooling=True):
        # x = data.x.squeeze()
        # print(data.x.shape)
        x=data.x.squeeze()
        # x = self.node_encoder(data.x.squeeze())
        x = self.concept_emb(x, contextualized_emb=None) #(batch_size, n_node-1, dim_node) concept_ids[:, 1:]-1
        # gnn_input1 = gnn_input1.to(x.device)
        # print("x after concept emb", get_tensor_info(x))
        for i in range(self.num_layers):
            if self.gnn_type in ["gine"]:
                data.edge_attr = self.edge_encoders[i](data.edge_attr)
            x = self.g_convs[i](x, data.edge_index, data.edge_attr) if self.gnn_type in ["gine", "rgcn"] \
                                                                    else self.g_convs[i](x, data.edge_index)
            # print("x after conv", get_tensor_info(x))

            # x = self.atom_convs[i](x, data.edge_index)
            # x = self.atom_batch_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if not global_pooling:
            # print("global_pooling", global_pooling)
            x = self.lin(x)
            x = self.activation(x)
            return x

        x = scatter(x, data.batch, dim=0, reduce='mean')
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        x = self.activation(x)
        return x

# General GNN
class GNN(torch.nn.Module):
    def __init__(self, args=None, input_dim=None, g_dim=None, gnn_type=None, num_gnn_layers=2,
                 encode_node_features=False, encode_edge_features=False, dropout_at_beginning=True, activation_at_the_end=True,
                 pool_type="mean", global_pooling=False,
                 pretrained_concept_emb=None):
        super().__init__()

        self.num_layers = num_gnn_layers
        self.dropout = args.dropoutg
        self.activation_at_the_end=activation_at_the_end
        self.pool_type=pool_type
        self.global_pooling=global_pooling
        self.input_dim = input_dim
        self.dropout_at_beginning = dropout_at_beginning

        self.gnn_type = gnn_type
        assert gnn_type in ["fast_rgcn","rgcn", "gat", "gine", "gin","compgcn"]
        self.encode_node_features = encode_node_features
        self.encode_edge_features = encode_edge_features
        self.use_edge_features = gnn_type in ["gine"]

        print("self.use_edge_features", self.use_edge_features)
        self.is_multi_relation = gnn_type in ["fast_rgcn","rgcn", "compgcn"]
        if self.is_multi_relation: self.encode_edge_features=False

        # print("self.is_multi_relation", self.is_multi_relation)
        # print("self.use_edge_features", self.use_edge_features)
        # print("self.encode_node_features", self.encode_node_features)
        # print("self.encode_edge_features", self.encode_edge_features)

        self.use_concept_emb = False
        # if using concept emb
        if encode_node_features:
            # self.node_encoder = Linear(input_dim, g_dim)
            if pretrained_concept_emb is not None:
                # print("args.concept_num", args.concept_num)
                self.node_encoder = CustomizedEmbedding(concept_num=args.concept_num, concept_out_dim=g_dim,
                                                       use_contextualized=False, concept_in_dim=args.concept_dim,
                                                       pretrained_concept_emb=pretrained_concept_emb,
                                                       freeze_ent_emb=args.freeze_ent_emb)
                self.input_dim=g_dim#args.concept_dim
                self.use_concept_emb=True
            else:
                self.node_encoder = NodeEncoder(g_dim,num_node_types=args.num_node_types)
                self.input_dim = g_dim

        print("self.input_dim", self.input_dim)
        # self.node_encoder = NodeEncoder(args.g_dim)\
        if self.encode_edge_features:
            self.edge_encoders = ModuleList()
        self.g_convs = ModuleList()
        self.batch_norms = ModuleList()
        self.activation = Tanh()
        for i in range(num_gnn_layers):
            if self.use_edge_features:
                self.edge_encoders.append(EdgeEncoder(g_dim, num_edge_types=args.num_edge_types))

            in_dim=self.input_dim if i == 0 else g_dim
            if gnn_type == "gat":
                self.g_convs.append(GATConv(in_dim, g_dim))
            elif gnn_type == "gin":

                nn = Sequential(
                    Linear(in_dim, 2 * g_dim),
                    BatchNorm1d(2 * g_dim),
                    # ReLU(),
                    Tanh(),
                    Linear(2 * g_dim, g_dim),
                )
                self.g_convs.append(GINConv(nn, train_eps=True))
            elif gnn_type == "gine":

                nn = Sequential(
                    Linear(in_dim, 2 * g_dim),
                    # BatchNorm1d(2 * g_dim),
                    # ReLU(),
                    Tanh(),
                    Linear(2 * g_dim, g_dim),
                )
                self.g_convs.append(GINEConv(nn, train_eps=True))
            elif gnn_type == "fast_rgcn":
                self.g_convs.append(FastRGCNConv(in_dim, g_dim, args.num_relation))
            elif gnn_type == "rgcn":
                self.g_convs.append(RGCNConv(in_dim, g_dim, args.num_relation))
            elif gnn_type == "compgcn":
                self.g_convs.append(FastRGCNConv(in_dim, g_dim, args.num_relation))
            self.batch_norms.append(BatchNorm1d(g_dim))

        self.lin = Linear(g_dim, g_dim)
        self.dropout_e = Dropout(args.dropoute)

    def forward(self, x, data): #, global_pooling=True, pooling_method="mean"
        # x = data.x.squeeze()
        # print(data.x.shape)
        # x = data.x
        edge_attr=data.edge_attr
        edge_index=data.edge_index
        # print("x beginning", get_tensor_info(x))
        # print("data.edge_index", get_tensor_info(data.edge_index))
        # if edge_attr is not None: print("data.edge_attr", get_tensor_info(edge_attr))
        # print("x", x.shape)
        # print("edge_index", edge_index)
        # print("edge_attr", edge_attr)
        # print("====in gnn forward====")
        # print("x", get_tensor_info(x))
        # print("edge_attr", get_tensor_info(edge_attr))
        # print("edge_index", get_tensor_info(edge_index))

        # breakpoint()
        if self.encode_node_features: # x is list of node types
            x = x.squeeze()
            if self.use_concept_emb:
                # print("use_concept_emb")
                x = self.node_encoder(x, contextualized_emb=None) #(batch_size, n_node-1, dim_node) concept_ids[:, 1:]-1
            else:
                x = self.node_encoder(x)

        # print("x encoded", get_tensor_info(x))
        if self.dropout_at_beginning:
            x=self.dropout_e(x)

        for i in range(self.num_layers):
            if self.encode_edge_features:  # list of edge types, multi-relation doesn't need to encode
                cur_edge_attr = self.edge_encoders[i](edge_attr)
                # print("cur_edge_attr", get_tensor_info(cur_edge_attr))

            if self.is_multi_relation or self.use_edge_features:
                x = self.g_convs[i](x, edge_index, cur_edge_attr)
            else:
                x = self.g_convs[i](x, edge_index)
            # print("gnn e0 x", get_tensor_info(x))
            # x = self.atom_convs[i](x, data.edge_index)
            # x = self.batch_norms[i](x)
            x = torch.tanh(x)
            x = F.dropout(x, self.dropout, training=self.training)
        # print("x conved", get_tensor_info(x))

        # if not pooling
        if not self.global_pooling:
            x = self.lin(x)
            if self.activation_at_the_end:
                x = self.activation(x)
            return x

        # if pooling
        x = scatter(x, data.batch, dim=0, reduce = self.pool_type)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        if self.activation_at_the_end: x = self.activation(x)

        return x
    def reset_parameters(self):
        if self.encode_node_features:
            self.node_encoder.reset_parameters()
        if self.encode_edge_features:
            for emb in self.edge_encoders:
                emb.reset_parameters()
        for conv, batch_norm in zip(self.g_convs, self.batch_norms):
            conv.reset_parameters()
            batch_norm.reset_parameters()
        self.lin.reset_parameters()
