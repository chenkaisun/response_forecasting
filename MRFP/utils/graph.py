import torch
import networkx as nx
import itertools
import json
from tqdm import tqdm
from .conceptnet import merged_relations
import numpy as np
from scipy import sparse
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from multiprocessing import Pool
from collections import OrderedDict
from utils.utils import dump_file, load_file
from IPython import embed
from .maths import *
import collections

__all__ = ['generate_graph']

concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}


def load_cpnet(cpnet_graph_path):
    global cpnet, cpnet_simple
    cpnet = nx.read_gpickle(cpnet_graph_path)
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def relational_graph_generation(qcs, acs, paths, rels):
    raise NotImplementedError()  # TODO


# plain graph generation
def plain_graph_generation(qcs, acs, paths, rels):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple

    graph = nx.Graph()
    for p in paths:
        for c_index in range(len(p) - 1):
            h = p[c_index]
            t = p[c_index + 1]
            # TODO: the weight can computed by concept embeddings and relation embeddings of TransE
            graph.add_edge(h, t, weight=1.0)

    for qc1, qc2 in list(itertools.combinations(qcs, 2)):
        if cpnet_simple.has_edge(qc1, qc2):
            graph.add_edge(qc1, qc2, weight=1.0)

    for ac1, ac2 in list(itertools.combinations(acs, 2)):
        if cpnet_simple.has_edge(ac1, ac2):
            graph.add_edge(ac1, ac2, weight=1.0)

    if len(qcs) == 0:
        qcs.append(-1)

    if len(acs) == 0:
        acs.append(-1)

    if len(paths) == 0:
        for qc in qcs:
            for ac in acs:
                graph.add_edge(qc, ac, rel=-1, weight=0.1)

    g = nx.convert_node_labels_to_integers(graph, label_attribute='cid')  # re-index
    return nx.node_link_data(g)


def generate_adj_matrix_per_inst(nxg_str):
    global id2relation
    n_rel = len(id2relation)

    nxg = nx.node_link_graph(json.loads(nxg_str))
    n_node = len(nxg.nodes)
    cids = np.zeros(n_node, dtype=np.int32)
    for node_id, node_attr in nxg.nodes(data=True):
        cids[node_id] = node_attr['cid']

    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet_all.has_edge(s_c, t_c):
                for e_attr in cpnet_all[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    cids += 1
    adj = coo_matrix(adj.reshape(-1, n_node))
    return (adj, cids)


def concepts2adj(node_ids):
    global id2relation
    cids = np.array(node_ids, dtype=np.int32)
    n_rel = len(id2relation)
    n_node = cids.shape[0]
    adj = np.zeros((n_rel, n_node, n_node), dtype=np.uint8)
    for s in range(n_node):
        for t in range(n_node):
            s_c, t_c = cids[s], cids[t]
            if cpnet.has_edge(s_c, t_c):
                for e_attr in cpnet[s_c][t_c].values():
                    if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel:
                        adj[e_attr['rel']][s][t] = 1
    # cids += 1  # note!!! index 0 is reserved for padding
    adj = coo_matrix(adj.reshape(-1, n_node))
    return adj, cids


def concepts_to_adj_matrices_1hop_neighbours(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            extra_nodes |= set(cpnet[u])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_1hop_neighbours_without_relatedto(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for u in set(qc_ids) | set(ac_ids):
        if u in cpnet.nodes:
            for v in cpnet[u]:
                for data in cpnet[u][v].values():
                    if data['rel'] not in (15, 32):
                        extra_nodes.add(v)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_2hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask




class OrderedSet(collections.OrderedDict, collections.MutableSet):

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return 'OrderedSet([%s])' % (', '.join(map(repr, self.keys())))

    def __str__(self):
        return '{%s}' % (', '.join(map(repr, self.keys())))

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)

def concepts_to_adj_matrices_2hop_all_pair(data, only_last_first_and_last=True, last_emit=True, num_hops=2, max_node=200):

    global id2relation
    n_rel = len(id2relation)
    qc_ids, ac_ids, doc_id, global_doc_id = data

    # ac_ids=set()

    # if qc_ids=={575,
    #      1848,
    #      2071,
    #      3198,
    #      4445,
    #      5700,
    #      7634,
    #      13439,
    #      26657,
    #      34983,
    #      214884,
    #      234633,
    #      648721}:
    #     breakpoint()

    # print("qc_ids, ac_ids", qc_ids, ac_ids)
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    connected_pairs=set()
    qa_nodes=sorted(qa_nodes)

    checked_pairs=set()

    middle_nodes = OrderedDict()  # we make the the first sliced remaining ones are definetly connecting first and ;last node
    if num_hops==3:

        # 3 hop
        for qid in qc_ids:
            for aid in ac_ids:
                if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:

                    if (min(qid, aid), max(qid, aid)) in checked_pairs: continue
                    checked_pairs.add((min(qid, aid), max(qid, aid)))
                    # print("checked_pairs", checked_pairs)

                    cur_middle_nodes=set()
                    has_middle=False
                    for u in cpnet_simple[qid]:
                        for v in cpnet_simple[aid]:
                            if cpnet_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                                has_middle=True
                                middle_nodes[u]=None
                                middle_nodes[v]=None
                            if u == v:  # ac is a 2-hop neighbour of qc
                                has_middle = True
                                middle_nodes[u]=None
                    if has_middle:
                        connected_pairs.add((min(qid, aid), max(qid, aid)))
                # middle_nodes.update(cur_middle_nodes)
    elif num_hops==2:

        tmp_qc_ids=qc_ids-ac_ids

        # connect middle
        for qid in tmp_qc_ids:
            for aid in ac_ids:
                if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                    cur_middle_nodes = set(cpnet_simple[qid]) & set(cpnet_simple[aid])
                    if cur_middle_nodes:
                        connected_pairs.add((min(qid, aid), max(qid, aid)))
                    for nd in cur_middle_nodes:
                        middle_nodes[nd]=None


        # connect middle
        # for i, qid in enumerate(qa_nodes):
        #     for j in range(i+1, len(qa_nodes)):
        #         aid=qa_nodes[j]
        #
        #         if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
        #             cur_middle_nodes = set(cpnet_simple[qid]) & set(cpnet_simple[aid])
        #             if cur_middle_nodes:
        #                 connected_pairs.add((min(qid, aid), max(qid, aid)))
        #
        #             for nd in cur_middle_nodes:
        #                 middle_nodes[nd]=None
        #             # middle_nodes.update(cur_middle_nodes)


        for cur_set in [sorted(tmp_qc_ids), sorted(ac_ids)]:
            for i, qid in enumerate(cur_set):
                for j in range(i+1, len(cur_set)):
                    aid=cur_set[j]
                    if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                        cur_middle_nodes = set(cpnet_simple[qid]) & set(cpnet_simple[aid])
                        # extra_nodes |= set(cpnet_simple[qid]) | set(cpnet_simple[aid])# try having all extra nodes
                        if cur_middle_nodes:
                            connected_pairs.add((min(qid, aid), max(qid, aid)))
                        for nd in cur_middle_nodes:
                            middle_nodes[nd]=None
    if last_emit:
        if len(set(qa_nodes)|set(middle_nodes.keys()))< max_node:
            useful_rels= set([relation2id[rel_name] for rel_name in ['capableof', 'causes', 'createdby', 'desires', 'hassubevent', 'madeof', 'notdesires',
                                                                     'hascontext', 'hasproperty','madeof', 'notcapableof', 'receivesaction', 'usedfor']])
            #expand last node
            for i, aid in enumerate(ac_ids):
                if aid not in cpnet_simple:
                    # print(aid)
                    # embed()
                    pass
                else:
                    for neib in set(cpnet_simple[aid]):
                        found = False
                        if cpnet.has_edge(aid, neib):
                            for e_attr in cpnet[aid][neib].values():
                                if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel and e_attr['rel'] in useful_rels:
                                    found = True
                                    extra_nodes.add(neib)
                                    break
                        if not found and cpnet.has_edge(neib,aid): # check reverse edge
                            for e_attr in cpnet[neib][aid].values():
                                if e_attr['rel'] >= 0 and e_attr['rel'] < n_rel and e_attr['rel'] in useful_rels:
                                    extra_nodes.add(neib)
                                    break


    # print("here")
    # tmp=set()
    # for iid in ac_ids:
    #     if iid in cpnet_simple:
    #         tmp.update(set(cpnet_simple[iid]))
    # print("extra node length", len(tmp))

    qa_nodes=set(qa_nodes)


    # extra_nodes = extra_nodes - qa_nodes #nieghbors
    # extra_nodes = extra_nodes - middle_nodes #nieghbors


    # middle_nodes = middle_nodes - qa_nodes #nieghbors
    for node in qa_nodes:
         if node in middle_nodes:
             middle_nodes.pop(node)

    extra_nodes = extra_nodes - qa_nodes #nieghbors
    extra_nodes = extra_nodes - set(middle_nodes.keys()) #nieghbors
    # print("len(qa_nodes)", len(qa_nodes))
    if len(middle_nodes)>max_node:
        print("len(middle_nodes)>max_node", len(middle_nodes))
    # print("len(extra_nodes)", len(extra_nodes)) # TOo large

    # print("len(middle_nodes)", len(middle_nodes))
    # print("len(extra_nodes)", len(extra_nodes))
    # print("len(qa_nodes)", len(qa_nodes))
    # schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(middle_nodes)#+sorted(extra_nodes) # priority order

    # middle node maintains order
    schema_graph = sorted(qa_nodes) +list(middle_nodes)+ sorted(extra_nodes)#+sorted(extra_nodes) # priority order
    schema_graph=schema_graph[:max_node]

    # print("len(schema_graph)", len(schema_graph))
    # arange = np.arange(len(schema_graph))
    # qmask = arange < len(qc_ids)
    # amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    qmask, amask=None, None

    if schema_graph:
        adj, concepts = concepts2adj(schema_graph)
    else:
        adj, concepts = None, None
    if 7634 in qc_ids and 7634 not in concepts:
        breakpoint()

    return adj, concepts, qmask, amask, connected_pairs, doc_id, global_doc_id, qc_ids, ac_ids


def concepts_to_adj_matrices_2step_relax_all_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    intermediate_ids = extra_nodes - qa_nodes
    for qid in intermediate_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    for qid in qc_ids:
        for aid in intermediate_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask


def concepts_to_adj_matrices_3hop_qa_pair(data):
    qc_ids, ac_ids = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qc_ids:
        for aid in ac_ids:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                for u in cpnet_simple[qid]:
                    for v in cpnet_simple[aid]:
                        if cpnet_simple.has_edge(u, v):  # ac is a 3-hop neighbour of qc
                            extra_nodes.add(u)
                            extra_nodes.add(v)
                        if u == v:  # ac is a 2-hop neighbour of qc
                            extra_nodes.add(u)
    extra_nodes = extra_nodes - qa_nodes
    schema_graph = sorted(qc_ids) + sorted(ac_ids) + sorted(extra_nodes)
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return adj, concepts, qmask, amask



######################################################################
from transformers import RobertaTokenizer, RobertaForMaskedLM

class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)
    #
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, masked_lm_labels=None):
        #
        assert attention_mask is not None
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0] #hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs

# print ('loading pre-trained LM...')
# TOKENIZER = RobertaTokenizer.from_pretrained('roberta-large')
# LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained('roberta-large')
# LM_MODEL.cuda(); LM_MODEL.eval()
# print ('loading done')

def get_LM_score(cids, question):
    cids = cids[:]
    cids.insert(0, -1) #QAcontext node
    sents, scores = [], []
    for cid in cids:
        if cid==-1:
            sent = question.lower()
        else:
            sent = '{} {}.'.format(question.lower(), ' '.join(id2concept[cid].split('_')))
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)
    n_cids = len(cids)
    cur_idx = 0
    batch_size = 50
    while cur_idx < n_cids:
        #Prepare batch
        input_ids = sents[cur_idx: cur_idx+batch_size]
        max_len = max([len(seq) for seq in input_ids])
        for j, seq in enumerate(input_ids):
            seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
            input_ids[j] = seq
        input_ids = torch.tensor(input_ids).cuda() #[B, seqlen]
        mask = (input_ids!=1).long() #[B, seq_len]
        #Get LM score
        with torch.no_grad():
            outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
            loss = outputs[0] #[B, ]
            _scores = list(-loss.detach().cpu().numpy()) #list of float
        scores += _scores
        cur_idx += batch_size
    assert len(sents) == len(scores) == len(cids)
    cid2score = OrderedDict(sorted(list(zip(cids, scores)), key=lambda x: -x[1])) #score: from high to low
    return cid2score

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1(data):
    qc_ids, ac_ids, question = data
    qa_nodes = set(qc_ids) | set(ac_ids)
    extra_nodes = set()
    for qid in qa_nodes:
        for aid in qa_nodes:
            if qid != aid and qid in cpnet_simple.nodes and aid in cpnet_simple.nodes:
                extra_nodes |= set(cpnet_simple[qid]) & set(cpnet_simple[aid])
    extra_nodes = extra_nodes - qa_nodes
    return (sorted(qc_ids), sorted(ac_ids), question, sorted(extra_nodes))

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(data):
    qc_ids, ac_ids, question, extra_nodes = data
    cid2score = get_LM_score(qc_ids+ac_ids+extra_nodes, question)
    return (qc_ids, ac_ids, question, extra_nodes, cid2score)

def concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3(data):
    qc_ids, ac_ids, question, extra_nodes, cid2score = data
    schema_graph = qc_ids + ac_ids + sorted(extra_nodes, key=lambda x: -cid2score[x]) #score: from high to low
    arange = np.arange(len(schema_graph))
    qmask = arange < len(qc_ids)
    amask = (arange >= len(qc_ids)) & (arange < (len(qc_ids) + len(ac_ids)))
    adj, concepts = concepts2adj(schema_graph)
    return {'adj': adj, 'concepts': concepts, 'qmask': qmask, 'amask': amask, 'cid2score': cid2score}

################################################################################



#####################################################################################################
#                     functions below this line will be called by preprocess.py                     #
#####################################################################################################


def generate_graph(grounded_path, pruned_paths_path, cpnet_vocab_path, cpnet_graph_path, output_path):
    print(f'generating schema graphs for {grounded_path} and {pruned_paths_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet, cpnet_simple
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    nrow = sum(1 for _ in open(grounded_path, 'r'))
    with open(grounded_path, 'r') as fin_gr, \
            open(pruned_paths_path, 'r') as fin_pf, \
            open(output_path, 'w') as fout:
        for line_gr, line_pf in tqdm(zip(fin_gr, fin_pf), total=nrow):
            mcp = json.loads(line_gr)
            qa_pairs = json.loads(line_pf)

            statement_paths = []
            statement_rel_list = []
            for qas in qa_pairs:
                if qas["pf_res"] is None:
                    cur_paths = []
                    cur_rels = []
                else:
                    cur_paths = [item["path"] for item in qas["pf_res"]]
                    cur_rels = [item["rel"] for item in qas["pf_res"]]
                statement_paths.extend(cur_paths)
                statement_rel_list.extend(cur_rels)

            qcs = [concept2id[c] for c in mcp["qc"]]
            acs = [concept2id[c] for c in mcp["ac"]]

            gobj = plain_graph_generation(qcs=qcs, acs=acs,
                                          paths=statement_paths,
                                          rels=statement_rel_list)
            fout.write(json.dumps(gobj) + '\n')

    print(f'schema graphs saved to {output_path}')
    print()


def generate_adj_matrices(ori_schema_graph_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes, num_rels=34, debug=False):
    print(f'generating adjacency matrices for {ori_schema_graph_path} and {cpnet_graph_path}...')

    global concept2id, id2concept, relation2id, id2relation
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)

    global cpnet_all
    if cpnet_all is None:
        cpnet_all = nx.read_gpickle(cpnet_graph_path)

    with open(ori_schema_graph_path, 'r') as fin:
        nxg_strs = [line for line in fin]

    if debug:
        nxgs = nxgs[:1]

    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(generate_adj_matrix_per_inst, nxg_strs), total=len(nxg_strs)))

    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adjacency matrices saved to {output_path}')
    print()


def generate_adj_data_from_grounded_concepts(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        # print("load_resources")
        load_resources(cpnet_vocab_path)

    if cpnet is None or cpnet_simple is None:
        # print("load_cpnet")
        load_cpnet(cpnet_graph_path)

    qa_data = []

    for dic in load_file(grounded_path):
        q_ids = set(concept2id[c] for c in dic['first_qc']) # qc
        a_ids = set(concept2id[c] for c in dic['last_qc'])
        doc_id = dic['doc_id']
        global_doc_id = dic['global_doc_id']
        # first_qc = dic['first_qc']
        # last_qc= dic['last_qc']

        # if dic["doc_list"]==[['How', 'to', 'Tape', 'Off', 'a', 'Room', 'for', 'Painting'], ['Taping', 'off', 'the', 'Room'], ['Choose', 'the', 'right', 'tape'], ['Wipe', 'down', 'the', 'areas', 'you', 'wish', 'to', 'tape'], ['Getting', 'the', 'Room', 'Ready']]:
        #     breakpoint()

        # print("q_ids before", q_ids, "a_ids",a_ids)
        # q_ids = q_ids - a_ids
        # first_qc = first_qc - last_qc
        # last_qc = last_qc - a_ids
        qa_data.append((q_ids, a_ids, doc_id, global_doc_id, ))
    # with open(grounded_path, 'r', encoding='utf-8') as fin:
    #     for line in fin:
    #         dic = json.loads(line)
    #         q_ids = set(concept2id[c] for c in dic['qc'])
    #         a_ids = set(concept2id[c] for c in dic['ac'])
    #
    #         # print("q_ids before", q_ids, "a_ids",a_ids)
    #         q_ids = q_ids - a_ids
    #         qa_data.append((q_ids, a_ids))

    if num_processes<=1:
        res=[concepts_to_adj_matrices_2hop_all_pair(item) for item in tqdm(qa_data)]
    else:
        with Pool(num_processes) as p:
            res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))

    # embed()
    # print("adj res", res)
    # res is a list of tuples, each tuple consists of four elements (adj, concepts, qmask, amask)
    with open(output_path, 'wb') as fout:
        pickle.dump(res, fout)

    print(f'adj data saved to {output_path}')
    print()

def generate_adj_data_from_grounded_concepts_direct(qcs, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    # print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        # print("load_resources")
        load_resources(cpnet_vocab_path)

    if cpnet is None or cpnet_simple is None:
        # print("load_cpnet")
        load_cpnet(cpnet_graph_path)

    qa_data = []

    for qc in qcs:
        q_ids = set(concept2id[c] for c in qc)
        a_ids = set()
        q_ids = q_ids - a_ids
        qa_data.append((q_ids, a_ids))

    if num_processes<=1:
        res=[concepts_to_adj_matrices_2hop_all_pair(item) for item in tqdm(qa_data)]
    else:
        with Pool(num_processes) as p:
            res = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair, qa_data), total=len(qa_data)))
    return res

def generate_adj_data_from_grounded_concepts__use_LM(grounded_path, cpnet_graph_path, cpnet_vocab_path, output_path, num_processes):
    """
    This function will save
        (1) adjacency matrics (each in the form of a (R*N, N) coo sparse matrix)
        (2) concepts ids
        (3) qmask that specifices whether a node is a question concept
        (4) amask that specifices whether a node is a answer concept
        (5) cid2score that maps a concept id to its relevance score given the QA context
    to the output path in python pickle format

    grounded_path: str
    cpnet_graph_path: str
    cpnet_vocab_path: str
    output_path: str
    num_processes: int
    """
    print(f'generating adj data for {grounded_path}...')

    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(cpnet_vocab_path)
    if cpnet is None or cpnet_simple is None:
        load_cpnet(cpnet_graph_path)

    qa_data = []
    statement_path = grounded_path.replace('grounded', 'statement')
    with open(grounded_path, 'r', encoding='utf-8') as fin_ground, open(statement_path, 'r', encoding='utf-8') as fin_state:
        lines_ground = fin_ground.readlines()
        lines_state  = fin_state.readlines()
        assert len(lines_ground) % len(lines_state) == 0
        n_choices = len(lines_ground) // len(lines_state)
        for j, line in enumerate(lines_ground):
            dic = json.loads(line)
            q_ids = set(concept2id[c] for c in dic['qc'])
            a_ids = set(concept2id[c] for c in dic['ac'])
            q_ids = q_ids - a_ids
            statement_obj = json.loads(lines_state[j//n_choices])
            QAcontext = "{} {}.".format(statement_obj['question']['stem'], dic['ans'])
            qa_data.append((q_ids, a_ids, QAcontext))

    with Pool(num_processes) as p:
        res1 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part1, qa_data), total=len(qa_data)))

    res2 = []
    for j, _data in enumerate(res1):
        if j % 100 == 0: print (j)
        res2.append(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part2(_data))

    with Pool(num_processes) as p:
        res3 = list(tqdm(p.imap(concepts_to_adj_matrices_2hop_all_pair__use_LM__Part3, res2), total=len(res2)))

    # res is a list of responses
    with open(output_path, 'wb') as fout:
        pickle.dump(res3, fout)

    print(f'adj data saved to {output_path}')
    print()



#################### adj to sparse ####################

def coo_to_normalized_per_inst(data):
    adj, concepts, qm, am, max_node_num = data
    ori_adj_len = len(concepts)
    concepts = torch.tensor(concepts[:min(len(concepts), max_node_num)])
    adj_len = len(concepts)
    qm = torch.tensor(qm[:adj_len], dtype=torch.uint8)
    am = torch.tensor(am[:adj_len], dtype=torch.uint8)
    ij = adj.row
    k = adj.col
    n_node = adj.shape[1]
    n_rel = 2 * adj.shape[0] // n_node
    i, j = ij // n_node, ij % n_node
    mask = (j < max_node_num) & (k < max_node_num)
    i, j, k = i[mask], j[mask], k[mask]
    i, j, k = np.concatenate((i, i + n_rel // 2), 0), np.concatenate((j, k), 0), np.concatenate((k, j), 0)  # add inverse relations
    adj_list = []
    for r in range(n_rel):
        mask = i == r
        ones = np.ones(mask.sum(), dtype=np.float32)
        A = sparse.csr_matrix((ones, (k[mask], j[mask])), shape=(max_node_num, max_node_num))  # A is transposed by exchanging the order of j and k
        adj_list.append(normalize_sparse_adj(A, 'coo'))
    adj_list.append(sparse.identity(max_node_num, dtype=np.float32, format='coo'))
    return ori_adj_len, adj_len, concepts, adj_list, qm, am


def coo_to_normalized(adj_path, output_path, max_node_num, num_processes):
    print(f'converting {adj_path} to normalized adj')

    with open(adj_path, 'rb') as fin:
        adj_data = pickle.load(fin)
    data = [(adj, concepts, qmask, amask, max_node_num) for adj, concepts, qmask, amask in adj_data]

    ori_adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    adj_lengths = torch.zeros((len(data),), dtype=torch.int64)
    concepts_ids = torch.zeros((len(data), max_node_num), dtype=torch.int64)
    qmask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)
    amask = torch.zeros((len(data), max_node_num), dtype=torch.uint8)

    adj_data = []
    with Pool(num_processes) as p:
        for i, (ori_adj_len, adj_len, concepts, adj_list, qm, am) in tqdm(enumerate(p.imap(coo_to_normalized_per_inst, data)), total=len(data)):
            ori_adj_lengths[i] = ori_adj_len
            adj_lengths[i] = adj_len
            concepts_ids[i][:adj_len] = concepts
            qmask[i][:adj_len] = qm
            amask[i][:adj_len] = am
            adj_list = [(torch.LongTensor(np.stack((adj.row, adj.col), 0)),
                         torch.FloatTensor(adj.data)) for adj in adj_list]
            adj_data.append(adj_list)

    torch.save((ori_adj_lengths, adj_lengths, concepts_ids, adj_data), output_path)

    print(f'normalized adj saved to {output_path}')
    print()

# if __name__ == '__main__':
#     generate_adj_matrices_from_grounded_concepts('./data/csqa/grounded/train.grounded.jsonl',
#                                                  './data/cpnet/conceptnet.en.pruned.graph',
#                                                  './data/cpnet/concept.txt',
#                                                  '/tmp/asdf', 40)
