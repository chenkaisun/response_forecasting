import logging
import random

import numpy
import numpy as np
import torch
from datasets import load_metric
from pynvml import *
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import wandb
from model.load_model import get_model, load_model_from_path
from transformers import AutoTokenizer, GPT2Tokenizer, OpenAIGPTTokenizer
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils.optimization_utils import OPTIMIZER_CLASSES
# from options import read_args
from utils.utils import mkdir, dump_file, load_file
from IPython import embed
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ScoreRecorder:
    def __init__(self, path):
        pass

    def get_highest(self):
        pass


def pretraining(files, data, model, args):
    pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_logger(args):
    # if os.path.exists(args.experiment_path + args.experiment + ".txt"):
    #
    #     with open(args.experiment_path + "count.txt", mode="r") as f:
    #         pos = int(f.readlines()[-1].strip())
    #     with open(args.experiment_path + "count.txt", mode="w") as f:
    #         f.write(str(pos + 1))
    #     os.rename(args.experiment_path + args.experiment + ".txt",
    #               args.experiment_path + args.experiment + "_" + str(pos) + ".txt")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    mkdir(args.experiment_path)
    output_file_handler = logging.FileHandler(args.experiment_path + args.exp + "_" + args.exp_id + ".txt")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger


def get_plm_fullname(abbr):
    plm_dict = {
        "mini": "prajjwal1/bert-mini",
        "base-cased": "bert-base-cased",
        "base-uncased": "bert-base-uncased",
        "sap": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
        "sci": "allenai/scibert_scivocab_uncased",
        "tiny": "prajjwal1/bert-tiny"
    }
    return plm_dict[abbr]


def setup_common(args, tokenizer, concept_index_mapping=None):
    # debug dataset first so don't uncomment below
    # args.max_seq_len = tokenizer.model_max_length if tokenizer.model_max_length <= 100000 else 512  # TODO

    "=========Directories and Init========="""
    mkdir("model")
    mkdir("model/states")

    # todo
    subdir = f'{args.dataset}_{args.subset}'

    mkdir(f"{args.analysis_dir}")
    mkdir(f"model/states/{subdir}/")
    mkdir(f"model/states/{subdir}/{args.model_name}/")
    mkdir(f"model/states/{subdir}/{args.model_name}/{args.exp_id}/")
    args.general_model_state_dir = f"model/states/{subdir}/{args.model_name}/"  # /"#"model/states/best_dev_" + args.exp_id + ".pt"
    args.output_dir = f"model/states/{subdir}/{args.model_name}/{args.exp_id}/"  # /"#"model/states/best_dev_" + args.exp_id + ".pt"
    mkdir(args.output_dir)
    mkdir("model_states/")

    if not len(args.model_path):
        if not args.debug and args.task_mode!="ret" and wandb.run.id is not None:
            args.model_path = args.output_dir + wandb.run.id + ".pt"  # "model/states/best_dev_" + args.exp_id + ".pt"
        else:
            args.model_path = args.output_dir + "best_dev_" + str(args.exp_id) + ".pt"

    # set_seeds(args)

    "=========Device========="""
    args.device = gpu_setup(use_gpu=args.use_gpu, gpu_id=args.gpu_id)
    if "cpu" in str(args.device):
        args.use_amp = 0

    """=========Model========="""
    # if not "bsl" in args.model_name:
    #
    #     # args.concept_dim, args.concept_num = 1, 799273
    #     # cp_emb = torch.rand(args.concept_num, 1)
    #     cp_emb=None
    #     if args.debug:
    #         args.concept_dim, args.concept_num=1, 799273
    #         cp_emb=torch.rand(args.concept_num, 1)
    #
    #     elif args.has_concepts:
    #         # print("args.ent_emb_paths",args.ent_emb_paths)
    #         cp_emb = [np.load(path) for path in args.ent_emb_paths]
    #         cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)
    #         # print("cp_emb", cp_emb.shape)
    #         # embed()
    #         # breakpoint()
    #
    #         ## not useful right now
    #         if concept_index_mapping is not None:
    #             cp_emb=torch.index_select(cp_emb, 0, torch.tensor(sorted(list(concept_index_mapping.keys())), dtype=torch.long))
    #         concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    #         print("cp_emb", cp_emb.shape)
    #         print('| num_concepts: {} |'.format(concept_num))
    #         args.concept_dim=concept_dim
    #         args.concept_num=concept_num
    #
    #     model = get_model(args, tokenizer, pretrained_concept_emb=cp_emb)



    model = get_model(args, tokenizer)
    view_model_param(args, model)
    print("Model", model)

    """=========Optimizer========="""
    optimizer = get_optimizer(args, model, args.downstream_layers, args.no_decay_layers)

    model, optimizer, args.start_epoch, args.best_dev_score = load_model_from_path(model, optimizer, args.load_model_path,
                                                                                   args.use_gpu, args.device)
    if len(args.load_model_path.strip()):
        # wandb.restore(23kb8cce, run_path="vanpelt/my-project/23kb8cce")
        # wandb.restore("1mznai1q.pt", run_path="run-20220616_150029-1mznai1q")
        #
        if "/" not in args.load_model_path and ".pt" not in args.load_model_path:
            cache_path=f"model_states/{args.load_model_path}.pt"
            if not os.path.exists(cache_path):

                ## fill in run path
                model_data=wandb.restore(f"{args.load_model_path}.pt", run_path=f"MRFP/{args.load_model_path}")
                model_data = torch.load(model_data.name)
                torch.save(model_data, cache_path)
                print("saving model to", cache_path)
            model_data = torch.load(cache_path)
        else:
            model_data=torch.load(args.load_model_path)
            #save model data to pt

        model.load_state_dict(model_data)
        print("Saved model loaded from ",args.load_model_path)

    """=========Logger========="""
    args.logger = get_logger(args)
    args.writer = SummaryWriter(log_dir=args.experiment_path + args.exp + "/")

    print("\n\n=====begin tunable args=====")
    arg_dict = vars(args)
    for key in args.tunable_params:
        print(f"{key}: {arg_dict[key]}")

    print("\n\n======rare args=======")
    for key in sorted(set(arg_dict.keys()) - set(args.tunable_params) - {"tunable_params"}):
        args.logger.debug(f"{key}: {arg_dict[key]}")  # print(f"{key}: {arg_dict[key]}")

    if not args.debug and args.task_mode!="ret":
        # wandb.config.update({p: arg_dict[p] for p in args.tunable_params})
        wandb.config.update({p: arg_dict[p] for p in sorted(arg_dict.keys())})

    print("=====end of args=====")

    return args, model, optimizer


def gpu_setup(use_gpu=True, gpu_id="0"):  # , use_random_available=True
    print("\nSetting up GPU")
    # if len(gpu_id):
    #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # print("visibes",os.environ["CUDA_VISIBLE_DEVICES"])
    num_gpus = 1
    if torch.cuda.is_available() and use_gpu:
        print(f"{torch.cuda.device_count()} GPU available")
        # print('cuda available with GPU:', torch.cuda.get_device_name(0))

        # use all
        device = torch.device("cuda")

    else:
        if use_gpu and not torch.cuda.is_available():
            print('cuda not available')
        device = torch.device("cpu")

    print("Device is set to", device)
    return device


def view_model_param(args, model):
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', args.model_name, total_param)
    return total_param
def view_model_param_state(model, rq=True):
    for name, param in model.named_parameters():
        if not rq or rq and param.requires_grad:
            print(name, " | ",param.requires_grad, " | ", param.data.size())

def match_target(target, arr, type="any"):
    return any(nd in target for nd in arr)


def get_optimizer(args, model, downstream_layers, no_decay):
    # no_decay and downstream_layers, not no_decay and no_decaydownstream_layers
    # no_decay and not downstream_layers, not no_decay and not downstream_layers

    # todo
    # grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in downstream_layers)], },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in downstream_layers)], "lr": args.lr},  # 1e-4,  'weight_decay': 1e-4
    # ]
    # print("model.named_parameters()",model.named_parameters())
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if not match_target(n, downstream_layers) and not match_target(n, no_decay)], },
    #     {"params": [p for n, p in model.named_parameters() if match_target(n, downstream_layers) and match_target(n, no_decay)], 'weight_decay': 0.0, "lr": args.lr},
    #     {"params": [p for n, p in model.named_parameters() if match_target(n, downstream_layers) and not match_target(n, no_decay)], "lr": args.lr},
    #     {"params": [p for n, p in model.named_parameters() if not match_target(n, downstream_layers) and match_target(n, no_decay)], 'weight_decay': 0.0},
    #     # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #     #   'weight_decay': 0.0, 'lr': args.plm_lr}
    # ]
    # grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and not match_target(n, no_decay)], 'weight_decay': args.weight_decay,
    #      "lr": args.plm_lr},
    #     {"params": [p for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers) and match_target(n, no_decay)], 'weight_decay': 0.0,
    #      "lr": args.lr},
    #     {"params": [p for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers) and not match_target(n, no_decay)], "lr": args.lr, 'weight_decay': args.weight_decay},
    #     {"params": [p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and match_target(n, no_decay)], 'weight_decay': 0.0,  "lr": args.plm_lr},
    #     # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #     #   'weight_decay': 0.0, 'lr': args.plm_lr}
    # ]
    # grouped_parameters = [
    #     {"params": [p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and not match_target(n, no_decay) and p.requires_grad], 'weight_decay': args.weight_decay,
    #      "lr": args.plm_lr},
    #     {"params": [p for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers) and p.requires_grad], 'weight_decay': args.weight_decay,
    #      "lr": args.lr},
    #     {"params": [p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and match_target(n, no_decay) and p.requires_grad], 'weight_decay': 0.0,  "lr": args.plm_lr},
    #     # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #     #   'weight_decay': 0.0, 'lr': args.plm_lr}
    # ]

    grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if
                    not match_target(n, downstream_layers) and not match_target(n, no_decay) and p.requires_grad], 'weight_decay': args.weight_decay,
         "lr": args.plm_lr},
        {"params": [p for n, p in model.named_parameters() if
                    match_target(n, downstream_layers) and not match_target(n, no_decay) and p.requires_grad], 'weight_decay': args.weight_decay,
         "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if
                    not match_target(n, downstream_layers) and match_target(n, no_decay) and p.requires_grad], 'weight_decay': 0.0,  "lr": args.plm_lr},
        {"params": [p for n, p in model.named_parameters() if
                    match_target(n, downstream_layers) and match_target(n, no_decay) and p.requires_grad], 'weight_decay': 0.0, "lr": args.lr},
        # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        #   'weight_decay': 0.0, 'lr': args.plm_lr}
    ]
    # a=[p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and not match_target(n, no_decay)]
    # aa=[p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and not match_target(n, no_decay) and p.requires_grad]
    # b=[p for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers)]
    # bb=[p for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers) and p.requires_grad]
    # c=[p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and match_target(n, no_decay)]
    # cc=[p for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and match_target(n, no_decay) and p.requires_grad]
    # d=[n for n, p in model.named_parameters()]
    # grouped_parameters=[gp for gp in grouped_parameters if len(gp['params'])]

    # for tmp in [a,b,c,d]:
    #     print(len(set(tmp))==len(tmp))
    # a=set(a)
    # b=set(b)
    # c=set(c)
    # d=set(d)
    #
    # for j, tmp1 in enumerate([a,b,c]):
    #     for i, tmp2 in enumerate([a,b,c]):
    #         if i>j:
    #             print(tmp1&tmp2)
    # print(d==a.union(b).union(c))

    # grouped_parameterss = [
    #     {"params": [n for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and not match_target(n, no_decay)],
    #      'weight_decay': args.weight_decay,
    #      "lr": args.plm_lr},
    #     {"params": [n for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers) and match_target(n, no_decay)], 'weight_decay': 0.0,
    #      "lr": args.lr},
    #     {"params": [n for n, p in model.named_parameters() if
    #                 match_target(n, downstream_layers) and not match_target(n, no_decay)], "lr": args.lr,
    #      'weight_decay': args.weight_decay},
    #     {"params": [n for n, p in model.named_parameters() if
    #                 not match_target(n, downstream_layers) and match_target(n, no_decay)], 'weight_decay': 0.0,
    #      "lr": args.plm_lr},
    #     # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #     #   'weight_decay': 0.0, 'lr': args.plm_lr}
    # ]

    if args.optim == "radam":
        return OPTIMIZER_CLASSES[args.optim](grouped_parameters, lr=args.plm_lr, weight_decay=args.weight_decay)
    else:
        return optim.AdamW(grouped_parameters,
                           lr=args.plm_lr,
                           weight_decay=args.weight_decay,
                           # betas=(0.9, 0.999),
                           eps=args.adam_epsilon)

def to_tensor_float(data):
    return torch.as_tensor(data, dtype=torch.float)


def to_tensor_long(data):
    return torch.as_tensor(data, dtype=torch.long)


def get_tensor_info(tensor):
    return f"Shape: {tensor.shape} | Type: {tensor.type()} | Device: {tensor.device}"
def get_tensor_long(data):
    return torch.tensor(data, dtype=torch.long)
# def get_tensor_float(data):
#     return torch.tensor(data, dtype=torch.float)


def get_tensor_float(data):
    return torch.tensor(data, dtype=torch.float)


def get_scheduler(optimizer, scheduler_type, num_warmup_steps=None, num_training_steps=None, eta_min=0, T_max=None):
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
    elif scheduler_type == 'cosine':
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
        #                                             num_training_steps=num_training_steps, eta_min=eta_min, T_max=T_max)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps, num_cycles = 3)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
    else:
        print("Unknown schedular")
        assert False, print("scheduler_type", scheduler_type)
    return scheduler


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_tokenizer(plm, save_dir="tokenizer/", slow_connection=False):

    if "gpt2" in plm.lower():
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', cls_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|endoftext|>')#<|pad|>
        tokenizer.padding_side = "left"
    elif "gpt" in plm.lower():
        tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt", bos_token='<|startoftext|>', cls_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|endoftext|>')#<|pad|>
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(plm)
    # if "t5-" in plm in ["patrickvonplaten/t5-tiny-random","t5-small","t5-base","t5-large"]:
    #     tokenizer = T5Tokenizer.from_pretrained(plm)
    return tokenizer

    mkdir(save_dir)
    tk_name = plm.split("/")[-1].replace("-", "_") + "_tokenizer.pkl"
    tk_name = os.path.join(save_dir, tk_name)
    if not os.path.exists(tk_name) or not slow_connection:
        tokenizer = AutoTokenizer.from_pretrained(plm)
        dump_file(tokenizer, tk_name)
    return load_file(tk_name)


def get_metric_program(metric_name="sacrebleu", save_dir="metrics/", slow_connection=False):
    # tokenizer = AutoTokenizer.from_pretrained(plm)
    # return tokenizer

    mkdir(save_dir)
    tk_name = metric_name + "_metric.pkl"
    tk_name = os.path.join(save_dir, tk_name)
    if not os.path.exists(tk_name) or not slow_connection:
        metric_program = load_metric(metric_name)
        dump_file(metric_program, tk_name)
    return load_file(tk_name)

def get_consine_sim_matrix(a,b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    # print(res)
    return res
def get_gpu_mem_info():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
