import argparse
# import configparser
# from utils.utils import load_file
# from IPython import embed
# from pprint import pprint as pp
import os
import torch
import random
import numpy as np
from multiprocessing import cpu_count
import yaml
from constants import TASK_SETTINGS

# parser = argparse.ArgumentParser()
# parser.add_argument("--gpu_id", default="", type=str, help="gpu_id", )
# args, _ = parser.parse_known_args()
# if len(args.gpu_id):
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#     print("gpuids", os.environ["CUDA_VISIBLE_DEVICES"])

# from multiprocessing import cpu_count
# from multiprocessing import Pool
EMB_PATHS = {
    'transe': 'data/transe/glove.transe.sgd.ent.npy',
    'lm': 'data/transe/glove.transe.sgd.ent.npy',
    'numberbatch': 'data/transe/concept.nb.npy',
    'tzw': 'data/cpnet/tzw.ent.npy',
}
PLM_DICT = {
    "bert-mini": "prajjwal1/bert-mini",
    "bert-tiny": "prajjwal1/bert-tiny",
    "bert-base-cased": "bert-base-cased",
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-cased": "bert-large-cased",
    "bert-large-uncased": "bert-large-uncased",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "sap": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
    "sci": "allenai/scibert_scivocab_uncased",
    "bart-tiny": "sshleifer/bart-tiny-random",
    "bart-base": "facebook/bart-base",
    "bart-large": "facebook/bart-large",
    "mt5-tiny": "stas/mt5-tiny-random",
    "t5-tiny": "patrickvonplaten/t5-tiny-random",
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "roberta-sentiment": "siebert/sentiment-roberta-large-english",
    "bertweet-sentiment": "finiteautomata/bertweet-base-sentiment-analysis",
    "roberta-emotion": "cardiffnlp/twitter-roberta-base-emotion",
    "cardiffnlp-sentiment-latest": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "t5-emotion": "mrm8488/t5-base-finetuned-emotion",
    "deberta-xlarge": "microsoft/deberta-v2-xlarge",
    "deberta-large": "microsoft/deberta-v3-large",
    "deberta-base": "microsoft/deberta-v3-base",
    "deberta-xsmall": "microsoft/deberta-v3-xsmall",
    "gpt2-tiny": "sshleifer/tiny-gpt2",
    "gpt2": "gpt2",
    "gpt-tiny": "hf-internal-testing/tiny-random-OpenAIGPTModel",
    "gpt": "openai-gpt",

}
PLM_DIM_DICT = {
    "bert-mini": 256,
    "bert-tiny": 128,
    "bert-small": 512,
    "bert-medium": 512,
    "bert-base-cased": 768,
    "bert-base-uncased": 768,
    "bert-large-cased": 1024,
    "bert-large-uncased": 1024,
    "roberta-base": 1024,
    "roberta-large": 1024,
    "sap": 768,
    "sci": 768,
    "bart-tiny": 24,
    "bart-base": 768,
    "bart-large": 1024,
    "t5-tiny": 64,
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024,
    "roberta-sentiment": 1024,
    "bertweet-sentiment": 768,
    "roberta-emotion": 768,
    "cardiffnlp-sentiment-latest": 768,
    "t5-emotion": 768,
    "deberta-xlarge": 1536,
    "deberta-large": 1024,
    "deberta-base": 768,
    "deberta-xsmall": 384,
    "gpt2": 768,
    "gpt2-tiny": 768,
    "gpt-tiny": 32,
    "gpt": 768,

}


# PLM_LEN_DICT = {
#     "bert-mini": 512,
#     "bert-tiny": 512,
#     "bert-small": 512,
#     "bert-medium": 512,
#     "bert-base-cased": 512,
#     "bert-base-uncased": 512,
#     "roberta-base": 1024,
#     "roberta-large": 1024,
#     "sap": 512,
#     "sci": 512,
#     "bart-tiny": 1024,
#     "bart-base": 1024,
#     "bart-large": 1024,
#     "t5-tiny": 1024,
#     "t5-base": 1024,
#     "t5-large": 1024
# }

def add_generation_arguments(parser):
    parser.add_argument("--length_penalty", default=1.2, type=float)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--generated_max_length", default=90, type=int)
    parser.add_argument("--generated_min_length", default=2, type=int)
    args, _ = parser.parse_known_args()


def add_experiment_arguments(parser):
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--config_file", default="", type=str)
    # parser.add_argument("--experiment", default="exp", type=str)
    parser.add_argument("--experiment_path", default="../experiment/", type=str)
    parser.add_argument("--exp", default="exp", type=str)
    parser.add_argument("--analysis_dir", default="analysis/", type=str)
    parser.add_argument("--group_output", default=0, type=int)
    parser.add_argument("--use_anno", default=1, type=int)
    parser.add_argument("--t_anno", default=1, type=int)
    parser.add_argument("--use_gpu", default=1, type=int, help="Using gpu or cpu", )
    args, _ = parser.parse_known_args()
    # parser.add_argument("--exp_id", default="0", type=str)
    parser.set_defaults(exp_id=os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else "0")
    num_devices = 1
    if args.use_gpu:
        num_devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) if 'CUDA_VISIBLE_DEVICES' in os.environ else torch.cuda.device_count()
    # parser.set_defaults(num_devices=num_devices)
    parser.set_defaults(my_num_devices=num_devices)

    print("numdevice", num_devices)

    parser.add_argument("--static_graph", default=0, type=int)
    parser.add_argument("--use_event_tag", default=0, type=int)
    parser.add_argument("--task_mode", default="clf", type=str)  # clf
    parser.add_argument("--script_completion", default=0, type=int)  # clf
    # parser.add_argument("--script_completion", default=0,  type=int)
    parser.add_argument("--intra_event_encoding", default=1, type=int)  # clf
    parser.add_argument("--no_dl_score", default=1, type=int)  # clf
    parser.add_argument("--visualize_scatterplot", action="store_true")  # clf
    parser.add_argument("--save_output", default=0, type=int)  # clf
    parser.add_argument('--scatterplot_metrics', default=["rouge", "bleu", "meteor"], nargs='+')

    parser.add_argument("--exp_msg", default="", type=str)  # clf
    parser.add_argument("--label_mode", default="multiclass", choices=["multiclass", "multilabel", "binary"], type=str)  # multilabel or not # todo

    tunable_params = ['model_name', 'plm', 'subset', 'components', 'use_special_tag', 'metric_for_best_model', 'greater_is_better',
                      'length_penalty', 'num_beams', 'generated_max_length', 'generated_min_length', 'has_concepts', 'patience',
                      'num_epochs', 'batch_size', 'true_batch_size', 'eval_batch_size', 'plm_lr', 'lr', 'scheduler',
                      'warmup_ratio', 'eval_steps', 'eval_strategy', 'label_smoothing_factor', 'num_evals_per_epoch',
                      'g_dim', 'pool_type', 'num_gnn_layers', 'num_gnn_layers2', 'gnn_type', 'gnn_type2', "optim", 'use_cache', 'max_num_ents',
                      'init_range', 'decattn_layer_idx', 'freeze_ent_emb', 'max_node_num', 'gpu_id', 'use_gpu', 'exp_msg', 'script_completion', "no_test", 'ret_components',
                      'nb_threshold', 'data_mode', 'seed', 'use_token_tag', 'cat_text_embed',
                      'intra_event_encoding', 'temporal_encoding', 'propagate_factor_embeddings']
    parser.set_defaults(
        # rare_params=['data_dir', 'use_cache', 'subset', 'ent_emb', 'max_node_num', 'train_file', 'dev_file', 'test_file',
        #              'train_adj', 'dev_adj', 'test_adj', 'weight_decay', 'adam_epsilon', 'max_grad_norm', 'use_gpu', 'use_amp', 'grad_accumulation_steps',
        #              'optim', 'model_path', 'downstream_layers', 'dropoute', 'dropouti', 'gnn_type', 'gnn_type2', 'freeze_ent_emb', 'debug', 'plm_hidden_dim'],
        tunable_params=sorted(set(tunable_params))
    )


def add_data_arguments(parser):
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--lower_case_tgt", default=0, type=int)
    parser.add_argument("--has_tgt_type", default=0, type=int)
    parser.add_argument("--database_io", default=0, type=int)
    parser.add_argument("--skip_empty_profile", default=0, type=int)
    parser.add_argument("--custom_dev_file", default="", type=str)
    parser.add_argument("--custom_test_file", default="", type=str)
    parser.add_argument("--use_intensity_for_sentiment", default=0, type=int)
    parser.add_argument("--sent_clf", default=0, type=int)
    parser.add_argument("--pred_only", default=0, type=int)
    parser.add_argument("--user_attributes", default="12", type=str)  # 1 profile 2 history 3 id
    parser.add_argument("--text_only", default=0, type=int)  # 1 profile 2 history 3 id

    args, _ = parser.parse_known_args()
    # global glob_seed
    # glob_seed = args.seed

    parser.add_argument("--num_workers", default=0, type=int)
    # parser.add_argument("--num_processes", default=cpu_count(), type=int)
    parser.add_argument("--out_dim", default=1, type=int)

    parser.add_argument("--data_dir", default="data/response_pred/labeler_data/", type=str)  # ../twitter_crawl/data_new2/
    parser.add_argument("--dataset", default="semeval18", type=str)  # CNN
    # parser.add_argument("--train_file", default="", type=str)
    # parser.add_argument("--dev_file", default="", type=str)
    # parser.add_argument("--test_file", default="", type=str)
    parser.add_argument("--cache_filename", default="", type=str)
    parser.add_argument("--index_filename", default="", type=str)
    parser.add_argument("--step_db_filename", default="", type=str)

    parser.add_argument("--use_cache", default=0, type=int)
    parser.add_argument("--use_mat_cache", default=0, type=int)

    parser.add_argument("--check_data", default=0, type=int)
    parser.add_argument("--subset", default="", type=str)
    parser.add_argument('--ent_emb', default=['tzw'], choices=['tzw', 'transe'], nargs='+', help='sources for entity embeddings')
    parser.add_argument('--num_relation', default=38, type=int, help='number of relations')
    parser.add_argument('--max_node_num', default=200, type=int)
    parser.add_argument('--max_concepts_num_for_each_token', default=2, type=int)
    parser.add_argument('--has_concepts', default=0, type=int)
    parser.add_argument('--use_special_tag', default=2, type=int)
    parser.add_argument('--num_nbs', type=int, default=2)
    parser.add_argument('--num_edge_types', default=12, type=int)
    parser.add_argument('--num_node_types', default=-1, type=int)
    parser.add_argument('--max_num_ents', default=10, type=int)
    parser.add_argument('--history_length', default="", type=str)
    parser.add_argument('--completion_length', default="", type=str)
    parser.add_argument('--hierachy', default=0, type=int)
    parser.add_argument('--num_tgt_steps', default=-1, type=int)
    parser.add_argument('--factor_expander', default=0, type=int)
    parser.add_argument('--augment_data', default=0, type=int)
    parser.add_argument('--pretrain_concept_generator', default=0, type=int)
    parser.add_argument('--use_generated_factors', default=0, type=int)
    parser.add_argument('--data_file', default=0, type=int)
    parser.add_argument('--label_category', default="emotion", type=str)
    parser.add_argument('--comment_generation', default=0, type=int)
    parser.add_argument('--label_generation', default=0, type=int)
    parser.add_argument('--personal_response_pred', default=0, type=int)
    parser.add_argument('--response_pred_labeling', default=0, type=int)
    # parser.add_argument('--target_splits', default="", type=str)
    parser.add_argument('--target_splits', default=[2], nargs='+')  # layer_norm.bias layer_norm.weight

    # parser.add_argument('--', default=0, type=int)
    parser.add_argument('--task_setting', default=-1, type=int)
    parser.add_argument('--min_num_likes', default=-1, type=int)

    parser.add_argument('--data_mode', default="full", type=str)
    # choices=['fs0.1', 'fs0.3', 'fs0.5', 'fs0.5',"crossdomain"],

    args, _ = parser.parse_known_args()

    # parser.set_defaults(ent_emb_paths=[EMB_PATHS.get(s) for s in args.ent_emb])
    # print("args.ent_emb", args.ent_emb)

    prefix_dict = {
        "full": "all_month_data_no_politics2",
        "comment_generation_prt": "all_month_data_no_politics",
        "full_cgen": "data_",
        "debug": "data_",
        # "": "all_month_data_no_politics2_",
        "": "",
    }


def add_training_arguments(parser):
    parser.add_argument("--num_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--true_batch_size", default=-1, type=int, help="True Batch size for training.")  # todo
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for training.")
    parser.add_argument("--weighted_loss", default=0, type=int, help="Batch size for training.")

    args, _ = parser.parse_known_args()
    # parser.set_defaults(total_batch_size=args.batch_size * args.num_devices) #true_batch_size=args.batch_size,

    parser.add_argument("--eval_batch_size", default=12, type=int, help="Batch size for training.")
    parser.add_argument("--plm_lr", default=3e-5, type=float, help="The initial learning rate for PLM.")
    parser.add_argument("--lr", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--metric_for_best_model", default='', type=str, )  # todobertscore
    parser.add_argument("--greater_is_better", default=1, type=int, )  # todobertscore

    parser.add_argument("--scheduler", default="linear", type=str, )  # constant_with_warmup  constant
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")

    parser.add_argument("--eval_steps", default=1000, type=int, help="Number of steps between each evaluation.")
    parser.add_argument("--num_evals", default=5, type=int, help="Number of eavls")
    parser.add_argument("--num_evals_per_epoch", default=2, type=int, help="Number of eavls")
    parser.add_argument("--eval_strategy", default="steps", type=str)  # epoch
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--burn_in', type=int, default=0)

    parser.add_argument("--label_smoothing_factor", default=0.0, type=float, )
    # parser.add_argument('-l', '--list', nFargs='+', help='<Required> Set flag', required=True)

    parser.add_argument("--gpu_id", default="", type=str, help="gpu_id", )
    args, _ = parser.parse_known_args()
    if len(args.gpu_id):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        print("gpuids", os.environ["CUDA_VISIBLE_DEVICES"])

    parser.add_argument("--n_gpu", default=1, type=int, help="Number of gpu", )
    parser.add_argument("--use_amp", default=0, type=int, help="Using mixed precision")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument('--optim', default='radam', choices=['sgd', 'adam', 'adamw', 'radam'])


def add_model_arguments(parser):
    parser.add_argument("--model_name", default="", type=str)  # , choices=['bsl_bert', 'bsl_bart', 'bsl_t5', 'bsl_gpt2', 'see', 'retriever']
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--wandb_run_id", default="", type=str)
    parser.add_argument("--load_model_path", default="", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--temporal_encoding", default="lstm", type=str)
    parser.add_argument("--ablation", default=0, type=int)  # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--num_tag_types", default=2, type=int)  # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--use_token_tag", default=0, type=int)  # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--cat_text_embed", default=0, type=int)  # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument("--generator_predictor", default=0, type=int)  # 0: all 1: no evtent track 2: no amr graph
    parser.add_argument('--confidence_threshold', type=float, default=0.8)  # ca is cross attention b/w memory, cm is causal memory

    parser.add_argument("--plm", default="bart-base", type=str, metavar='N')  # default="base-uncased",

    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument('--downstream_layers', default=["submodule_1", "gnn", "combiner", ], nargs='+')  # "pooler", "classifier"
    parser.add_argument('--no_decay_layers', default=['bias', 'LayerNorm.bias', 'LayerNorm.weight'], nargs='+')  # layer_norm.bias layer_norm.weight

    module_choices = ["event_tag", "wm", "fstm", "cm", "ca", "none"]
    module_choices = ["match", "graph", "cbr", "cat_nbs", "none"]
    parser.add_argument('--components', type=str, default="")  # ca is cross attention b/w memory, cm is causal memory
    # parser.add_argument('--components', type=str, default=module_choices, choices=module_choices, nargs='+')  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--remove_modules', type=str, default=[], choices=module_choices, nargs='+')  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--ret_components', type=str, default="seq")  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--nb_threshold', type=float, default=0.99986)  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--decattn_layer_idx', type=str, default='all')  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--propagate_factor_embeddings', type=int, default=0)  # ca is cross attention b/w memory, cm is causal memory
    parser.add_argument('--freeze_plm', default=0, type=int, help='freeze plm embedding layer')

    # parser.add_argument('--ret_components', default=["title", "cat", "seq"], choices=module_choices, nargs='+')  # ca is cross attention b/w memory, cm is causal memory

    # parser.add_argument('--concept_dim', type=int, default=256, help='Number of final hidden units for graph.')
    # parser.add_argument('--plm_hidden_dim', type=int, default=768, help='Number of hidden units for plm.')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Number of embedding units.')
    parser.add_argument('--g_dim', type=int, default=-1, help='Number of final hidden units for graph.')
    # parser.add_argument('--g_dim2', type=int, default=256, help='Number of final hidden units for graph.')

    parser.add_argument("--dropout", default=0.2, type=float, help="Dropout")
    parser.add_argument('--dropoute', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropouti', type=float, default=0.2, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.2, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    parser.add_argument("--activation", default="gelu", type=str)  # gelu
    parser.add_argument('--batch_norm', default=False, help="Please give a value for batch_norm")
    parser.add_argument('--pool_type', default="mean", type=str)  # for cm, 0 mean max, 1 max mean, 2 mean, 3 max

    parser.add_argument('--gnn_type', default="gine")  # "gat", "gine", "gin", , choices=["rgcn", "compgcn", "fast_rgcn"]
    parser.add_argument('--gnn_type2', default="gine")  # , choices=["gat", "gin"]
    # parser.add_argument('--g_global_pooling', default=0, type=int)
    parser.add_argument('--num_gnn_layers', type=int, default=2, help='Number of final hidden units for graph.')
    parser.add_argument('--num_gnn_layers2', type=int, default=2, help='Number of final hidden units for graph.')

    parser.add_argument('--freeze_ent_emb', default=1, type=int, help='freeze entity embedding layer')
    parser.add_argument('--init_range', default=0.02, type=float,
                        help='stddev when initializing with normal distribution')
    parser.add_argument('--n_attention_head', type=int, default=8)


def add_extra_arguments(parser):
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_r", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--only_cache_data", action="store_true")
    parser.add_argument("--eval_only", type=int, default=0)
    parser.add_argument("--run_dev", action="store_true")
    parser.add_argument("--update_input", type=int, default=0)
    parser.add_argument("--verbose_eval", type=int, default=1)
    parser.add_argument("--is_labeling", type=int, default=0)

    parser.add_argument("--no_test", type=int, default=0)
    parser.add_argument("--top_few", type=int, default=-1)

    # parser.add_argument("--debug", default=1, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--split_data", action="store_true")
    parser.add_argument("--analyze", default=0, type=int)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--slow_connection", action="store_true")


def add_preprocess_arguments(parser):
    parser.add_argument('--run', default=['wikihow_subset3'],  # 'wikihow_debug', 'common',‘csqa’
                        choices=['wikihow', 'wikihow_full', 'wikihow_debug', 'wikihow_subset1', 'wikihow_subset2',
                                 'wikihow_subset3', 'wikihow_subset4', 'wikihow_subset5', 'wikihow_subset6',
                                 'wikihow_subset7', 'wikihow_subset8', 'wikihow_subset9', 'wikihow_subset10',
                                 'wikihow_gosc', 'wikihow_clf', 'common', 'wikihow_union', 'csqa',
                                 'hswag', 'anli', 'exp', 'scitail', 'phys', 'socialiqa',
                                 'obqa', 'obqa-fact', 'make_word_vocab'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    # parser.add_argument('--max_node_num', type=int, default=500, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(),
                        help='number of processes to use')  # cpu_count() 10
    # parser.add_argument('--seed', type=int, default=0, help='random seed')
    # parser.add_argument('--debug', action='store_true', help='enable debug mode')
    # parser.add_argument('--data_dir', default="../twitter_crawl/data_new2/", )
    # parser.add_argument('--dataset', default="CNN", )
    parser.add_argument('--splits', default=['train', "dev", "test"], nargs='+')
    parser.add_argument('--steps', default=["0", "1", "2"], nargs='+')
    parser.add_argument('--roberta_batch_size', default=31, type=int)
    parser.add_argument('--num_files_after_split', default=7, type=int)
    parser.add_argument('--sub_file_id', default="", type=str)  # for distributed file processing


def read_args():
    parser = argparse.ArgumentParser()
    add_experiment_arguments(parser)
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_training_arguments(parser)
    add_generation_arguments(parser)
    add_extra_arguments(parser)
    add_preprocess_arguments(parser)

    "======READ FROM CONFIG======"
    args = parser.parse_args()
    if args.config:
        with open("configs/" + args.config + ".yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for key, value in config.items():
                setattr(args, key, value)

    "======PROCESS ARGUMENTS======"

    # Data
    if not args.subset:
        subset_dir = args.data_dir + args.dataset + "/"
    else:
        subset_dir = os.path.join(args.data_dir + args.dataset, args.subset) + "/"

    user_file, post_file, history_file, graph_file = [""] * 4
    splits = ["train", "dev", "test"]
    if args.config in ['response_pred', 'response_pred_sent',
                       'response_pred_by_gen', 'response_pred_by_gen_eval']:  # args.task_setting in [TASK_SETTINGS['response_pred'], TASK_SETTINGS['response_pred_sent']]: #args.personal_response_pred or
        # train_file = f'{subset_dir}response_data_train.json'
        # dev_file = f'{subset_dir}response_data_dev.json'
        # test_file = f'{subset_dir}response_data_test.json'

        train_file = f'{subset_dir}response_data_balanced_train.json'
        if args.t_anno:
            train_file = f'{subset_dir}response_data_balanced_train_.json'

        if args.use_anno:
            dev_file = f'{subset_dir}response_data_balanced_dev_.json'
            test_file = f'{subset_dir}response_data_balanced_test.json'
        else:
            dev_file = f'{subset_dir}response_data_balanced_dev.json'
            test_file = f'{subset_dir}response_data_balanced_test.json'
        if args.eval_only:
            if 0 not in args.target_splits:
                train_file= ''
            if 1 not in args.target_splits:
                dev_file = ''
            if 2 not in args.target_splits:
                test_file = ''
        user_file = f'{subset_dir}users_info_dict.json'
        history_file = f'{subset_dir}users_history_dict.json'
        post_file = f'{subset_dir}post_data.json'
        graph_file = f'{subset_dir}graph_data.pkl'
        # args.target_splits = [2]
        # if args.is_labeling:
        #     if args.target_split == "train":
        #         args.test_file = args.train_file
        #     elif args.target_split == "dev":
        #         args.test_file = args.dev_file
    elif args.config in ['response_pred_by_gen_custom'] or '_custom' in args.config:

        train_file = dev_file = ''
        test_file = f'{subset_dir}response_data_balanced_custom.json'
        user_file = f'{subset_dir}users_info_dict_custom.json'
        history_file = f'{subset_dir}users_history_dict_custom.json'
        post_file = f'{subset_dir}post_data_custom.json'
        graph_file = f'{subset_dir}graph_data.pkl'
        # args.target_splits = [2]

    elif args.config in ['label_sent'] or args.task_setting in [
        1]:  # args.task_setting in [TASK_SETTINGS['response_pred'], TASK_SETTINGS['response_pred_sent']]: #args.personal_response_pred or
        train_file = f'{subset_dir}response_data_test.json'
        dev_file = f'{subset_dir}{args.custom_dev_file}' if args.custom_dev_file else f'{subset_dir}response_data_test.json'
        test_file = f'{subset_dir}{args.custom_test_file}'
        args.target_splits = []
        if args.custom_dev_file:
            args.target_splits.append(1)
        if args.custom_test_file:
            args.target_splits.append(2)
        # args.target_splits = [1,2]
    elif args.config in ['response_pred_labeling']:
        train_file, dev_file, test_file = [f'{subset_dir}V-oc/{src}.json' for src in splits]
    else:
        train_file, dev_file, test_file = [f'{subset_dir}{args.label_category}_{src}.json' for src in splits]

    args.subset_dir = subset_dir
    args.train_file, args.dev_file, args.test_file = train_file, dev_file, test_file
    args.user_file, args.post_file, args.history_file, args.graph_file = user_file, post_file, history_file, graph_file
    args.save_dev_location, args.save_test_location = f"{subset_dir}dev_concepts.json", f"{subset_dir}test_concepts.json"

    # Components
    args.components = [item.strip() for item in args.components.split(",") if item.strip()]
    print("args.components", args.components)

    # Training
    if args.is_labeling: args.weighted_loss = 0

    # Metrics
    if not args.metric_for_best_model:
        if args.task_mode == "clf":
            metric_for_best_model = "accuracy"
            greater_is_better = 1
        elif args.task_mode == "gen":
            metric_for_best_model = "bleu"
            greater_is_better = 1
            if args.metric_for_best_model in ["loss"]:
                greater_is_better = 0
        elif args.task_mode == "prt":
            metric_for_best_model = "loss"
            greater_is_better = 0
        else:
            assert False
        args.metric_for_best_model = metric_for_best_model
    if args.metric_for_best_model in ["loss"]:
        args.greater_is_better = 0
    # Plm Class
    # args, _ = parser.parse_known_args()
    # parser.set_defaults(load_model_path=f"sunchenk/Projects/{args.wandb_run_id}")  #

    # PLM
    plm_class = args.plm.split("-")[0]  # .split("/")[1]
    if "cased" in args.plm:
        plm_class += "_cased"
    elif "uncased" in args.plm:
        plm_class += "_uncased"
    args.plm_class = plm_class.lower()
    if not args.model_name:
        if "bart-" in args.plm:
            args.model_name = "bart"
        elif "t5-" in args.plm:
            args.model_name = "t5"
        elif "gpt2" in args.plm:
            args.model_name = "gpt2"
        elif "gpt" in args.plm:
            args.model_name = "gpt"
        elif "roberta" in args.plm:
            args.model_name = "roberta"

    "======DEBUG======"
    if args.debug:
        print("Debug Mode ON")
        # args.test_file = args.dev_file
        # args.train_file = args.dev_file
        if "gpt2" in args.plm:
            args.plm = "gpt2-tiny"
        elif "gpt" in args.plm:
            args.plm = "gpt-tiny"
        elif "t5-" in args.plm:
            args.plm = "t5-tiny"
        elif args.model_name != "predictor" and 'label_' not in args.config:
            args.plm = "bart-tiny" if args.task_mode == "gen" else "deberta-xsmall"
        args.batch_size = 2
        args.eval_batch_size = 2
        args.true_batch_size = args.batch_size
        args.num_epochs = 2
        args.patience = 5
        args.eval_steps = 6

    "======PLM Model and Dim======"
    args.plm_hidden_dim = PLM_DIM_DICT[args.plm]
    if args.g_dim == -1:
        args.g_dim = args.plm_hidden_dim
    args.plm = PLM_DICT[args.plm]
    print("PLM Model is", args.plm)
    print("plm_hidden_dim  is", args.plm_hidden_dim)

    "======SETTING======"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # pp(args)tunable_params

    return args
