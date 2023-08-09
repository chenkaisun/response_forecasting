from options import *
from train import train, train_clf_trainer
from train_utils import *
from utils.utils import check_error2, get_best_ckpt, load_file_batch, load_file
from evaluate import evaluate
from data import PrimitiveGenerationDataset, PrimitivePredictionDataset
import wandb
import os
import shutil
import gc
from constants import *


from sklearn.utils.class_weight import compute_class_weight
def main_func():
    # global args
    """main"""
    """=========INIT========="""

    args = read_args()
    gc.collect()
    # print("args.nb_threshold", args.nb_threshold)

    if not args.debug and not args.task_mode == "ret":
        wandb.init()
        print("WANDB")

    """=========Set Tokenizer========="""
    tokenizer = get_tokenizer(args.plm, slow_connection=args.slow_connection)
    if args.use_special_tag == 1: tokenizer.add_special_tokens({'additional_special_tokens': ['[label]', '[PROFILE]', '[HISTORY]', '[POST]']})  # '[GOAL]','[SUBGOAL]' , '[LABEL_SEP]'
    elif args.use_special_tag == 2:
        tokenizer.add_tokens(['[label]', '[PROFILE]', '[HISTORY]', '[POST]'])  # '[UID]','[SN]',


    args.max_seq_len = tokenizer.model_max_length if tokenizer.model_max_length <= 100000 else 512  # TODO
    if args.plm == "google/rembert":
        args.max_seq_len = 512
    if "gpt" in args.plm.lower(): args.generated_max_length = args.max_seq_len - 2
    print("max_seq_len", args.max_seq_len)

    """=========Set MetaData & Parameter========="""
    labels = LABEL_SPACE[args.task_setting]
    args.label_name = LABEL_NAME[args.task_setting]

    args.out_dim = len(labels)
    args.multi_label = False

    """=========General Setup========="""

    """SPecial"""
    train_data, dev_data, test_data = None, None, None
    user_data, post_data, graph_data = None, None, None
    if args.task_mode == "gen":
        extra_data=None
        if args.user_file:

            if args.debug:
                # extra_data[2] = {k: v[:3] for k, v in extra_data[2].items()}
                tmp=load_file_batch(filenames=[args.user_file, args.post_file, args.graph_file])
                extra_data=(tmp[0],tmp[1],{},tmp[2])

            else:
                extra_data = load_file_batch(filenames=[args.user_file, args.post_file, args.history_file, args.graph_file])

        # extra_data = load_file_batch(filenames=[args.user_file, args.post_file, args.history_file, args.graph_file])
        train_data = PrimitiveGenerationDataset(args, args.train_file, tokenizer, in_train=True, extra_data=extra_data)
        dev_data = PrimitiveGenerationDataset(args, args.dev_file, tokenizer, extra_data=extra_data)
        test_data = PrimitiveGenerationDataset(args, args.test_file, tokenizer, extra_data=extra_data)
    elif args.task_mode == "clf":
        extra_data=None
        if args.user_file: extra_data = load_file_batch(filenames=[args.user_file, args.post_file, args.history_file, args.graph_file])
        train_data = PrimitivePredictionDataset(args, args.train_file, tokenizer, labels=labels, in_train=True, extra_data=extra_data)
        dev_data = PrimitivePredictionDataset(args, args.dev_file, tokenizer, labels=labels, extra_data=extra_data)
        test_data = PrimitivePredictionDataset(args, args.test_file, tokenizer, labels=labels, extra_data=extra_data)

        print("train_data.get_class_weights()", train_data.get_class_weights())
    if args.top_few != -1:
        test_data.instances = test_data.instances[: args.top_few]
    if args.data_mode.startswith("fs"):
        train_data.instances = train_data.instances[: int(args.data_mode[2:])]
    # for dt in [train_data, dev_data, test_data]:
    #     dt.instances=[sample for sample in dt.instances if len(sample["ent_list"])]
    # dt.instances=dt.instances[:int(len(dt.instances)//2)]

    # embed()
    if args.only_cache_data:  # no need to run program
        return

    args, model, optimizer = setup_common(args, tokenizer)
    print("setup done")
    # if args.check_data:
    #     for ds in [train_data, dev_data, test_data]:
    #         check_error2(ds.instances)
    if args.debug:
        # print("args.debug", args.debug)
        train_data.instances, dev_data.instances, test_data.instances = train_data.instances[:8], dev_data.instances[:4], test_data.instances[:4]
    if args.debug or args.detect_anomaly:
        print("set detect_anomaly")
        torch.autograd.set_detect_anomaly(True)
    if args.task_mode in ["gen", "prt"]:
        train(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))
    elif args.task_mode == "clf":
        # train_clf(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))
        train_clf_trainer(args, model, optimizer, tokenizer, (train_data, dev_data, test_data))

    # remove all train states due to space

    # with os.scandir(args.output_dir) as entries:
    #     for entry in entries:
    #         if entry.is_dir() and not entry.is_symlink():
    #             shutil.rmtree(entry.path)
    #         else:
    #             os.remove(entry.path)

    if not args.debug:
        print("wandb.config2", dict(wandb.config))
    # arg_dict=vars(args)
    # for key in wandb.config.keys():
    #     if key not in args.tunable_params:
    #         wandb.config._items.pop(key, None)

    # embed() #wandb.config._items.pop("_n_gpu",None)


if __name__ == '__main__':
    # with launch_ipdb_on_exception():
    #     main_func()
    main_func()
