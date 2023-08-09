import json
import os
import time
import argparse
# import torch
# import torch
import os
import pickle as pkl
import json
# import time
# import os
import numpy as np
# from bs4 import BeautifulSoup
# import argparse
# import gc
# from copy import deepcopy
# import torch.nn as nn
# import torch.nn.functional as F
# from pynvml import *
# import requests
# from numba import jit
from matplotlib.pyplot import plot
import logging
# from pynvml import *
import codecs
from IPython import embed
from glob import glob

logging.getLogger('matplotlib.font_manager').disabled = True
from skmultilearn.model_selection import iterative_train_test_split
import matplotlib.pyplot as plt
import requests
from collections import Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
import wandb
import math
from sklearn.model_selection import train_test_split
import pandas as pd

class Printer:
    def __init__(self):
        self.cur_string = ""

    def print(self, *strs):
        print(*strs)
        self.cur_string += " ".join(strs)


def check_data_unique(samples, keys):
    for key in keys:
        tmp = set()
        for sample in samples:
            if sample[key] in tmp:
                print(f"same {key} in data")
                embed()
                breakpoint()
            tmp.add(sample[key])
    # assert set([sample["title"] for sample in samples])==len(samples), breakpoint()


def construct_cache_name(name, *args, **kwargs):
    return name + "_" + "_".join(map(str, args)) + "_" + "_".join(map(str, kwargs.items()))


def get_samples_by_indices(samples, indices):
    return [samples[i] for i in indices]


def get_class_to_indices(samples, key='categories', filter_out_multiclass=False):
    categories_by_size = Counter([subitem for item in samples for subitem in item[key]])
    print("categories_by_size", categories_by_size)
    categories_by_size = sort_key_by_value(categories_by_size, reverse=True)  # largest left
    category2indices = {}
    multi_categories_indicies = set([i for i, sample in enumerate(samples) if len(sample[key]) != 1])

    recorded_indicies = set()
    for k in categories_by_size:

        vals = [i for i, sample in enumerate(samples) if (i not in recorded_indicies) and
                ((i not in multi_categories_indicies) or (not filter_out_multiclass))]
        if vals:
            recorded_indicies.update(vals)
            category2indices[k] = vals
    return category2indices


def convert_strlabels_to_idxlabels(labels, dic, exclude='inconfident', concatenated_input=False):
    res = []

    if concatenated_input:
        labels = [[cs.strip() for cs in item] for item in labels]
    for item in labels:
        tmp = [0] * (len(dic) - 1)
        for cs in item:
            if cs in dic and cs != exclude:
                tmp[dic[cs]] = 1
        res.append(tmp)
    return res


def convert_strlabels_to_rankinglabels(labels, dic, exclude='inconfident', concatenated_input=False, rel_scores=None):
    res = []

    if concatenated_input:
        labels = [[cs.strip() for cs in item] for item in labels]

    rel_scores = []
    for item in labels:
        rel_score_dic = {}
        tmp = [0] * (len(dic) - 1)
        top_score = len(item)
        for j, cs in enumerate(item):
            rel_score_dic[cs] = top_score
            tmp[j] = top_score
            top_score -= 1
        rel_scores.append(rel_score_dic)
        res.append(tmp)
    return res


def convert_dic_set_to_list(dic, do_sort=False):
    return {key: list(val) for key, val in dic.items()}



def convert_list_to_dict(dic, key="id"):
    return {item[key]: item for item in dic}

def stratified_split(categories, ratio=(.8, .1, .1), multilabel=False, X=None):
    category_list = sorted(set([subitem for sample in categories for subitem in sample]))
    category2id = {cat: i for i, cat in enumerate(category_list)}

    labels = []
    for sample in categories:
        labels.append([0] * len(category_list))
        for cat in sample:
            labels[-1][category2id[cat]] = 1

    if X is None: X = np.array(range(len(labels)))
    y = np.array(labels)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - ratio[0], train_size=ratio[0], random_state=42)
    train_index, test_index = list(msss.split(X, y))[0]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print("X split")
    # print(X_train, )
    # print(X_test, )
    assert not (set(X_train.tolist()) & set(X_test.tolist()))

    if ratio[2] <= 0:
        return sorted(X_train.tolist()), sorted(X_test.tolist())

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=ratio[2] / (ratio[1] + ratio[2]), train_size=1 - (ratio[2] / (ratio[1] + ratio[2])), random_state=42)
    dev_index, test_index = list(msss.split(X_test, y_test))[0]
    X_dev, X_test_tmp = X_test[dev_index], X_test[test_index]
    X_test = X_test_tmp

    # print("dev_index, test_index")
    # print(dev_index, test_index)
    return sorted(X_train.tolist()), sorted(X_dev.tolist()), sorted(X_test.tolist())


# category2indices no overlap
def _stratified_split(category2indices, ratio=(.8, .1, .1), id2cat=None, samples=None, multiclass=False):
    # ratio = (.8, .1, .1)

    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, ])
    y = np.array([[0, 0], [0, 0], [0, 1], [0, 0], [1, 1], [0, 0], [1, 0], [1, 0]])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, train_size=.5, random_state=0)

    for train_index, test_index in msss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # accumulate
    min_ratio = min(ratio)
    ratio[1] += ratio[0]
    # ratio[2]=1

    assigned_dict = {}

    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=0.5)

    train_set, dev_set, test_set = set(), set(), set()
    for k in category2indices:
        cur_indices = category2indices[k]
        l = len(cur_indices)

        if int(len(cur_indices) * min_ratio) >= 1:
            np.random.shuffle(cur_indices)
            print("cur_indices", cur_indices)

            new_l = l + assigned_dict[k]["train"] + assigned_dict[k]["dev"] + assigned_dict[k]["test"]
            num_train, num_dev, num_test = int(new_l * ratio[0]), int(new_l * (ratio[1] - ratio[0])), int(new_l * ratio[2])
            num_train_additional, num_dev_additional, num_test_additional = num_train - assigned_dict[k]["train"], \
                                                                            num_dev - assigned_dict[k]["dev"], \
                                                                            num_test - assigned_dict[k]["test"]

            train_indices = cur_indices[:int(l * ratio[0])]
            dev_indices = cur_indices[int(l * ratio[0]):int(l * ratio[1])]
            test_indices = cur_indices[int(l * ratio[1]):]
            print("cur_indices", cur_indices)
            print("train_indices", train_indices)
            print("dev_indices", dev_indices)
            print("test_indices", test_indices)

            train_set.add(train_indices)
            dev_set.add(dev_indices)
            test_set.add(test_indices)

            for j in cur_indices:
                for cat in id2cat[j]:
                    if cat not in assigned_dict:
                        assigned_dict[cat] = {"train": 0, "dev": 0, "test": 0}
                    if j in train_set:
                        assigned_dict[cat]["train"] += 1
                    elif j in dev_set:
                        assigned_dict[cat]["dev"] += 1
                    elif j in test_set:
                        assigned_dict[cat]["test"] += 1
    return list(train_set), list(dev_set), list(test_set)


def sort_key_by_value(arr, reverse=True):
    return [k for k, v in sorted(arr.items(), key=lambda x: x[1], reverse=reverse)]


def request_get(url, headers=None, params=None):
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/39.0.2171.95 Safari/537.36'}
    try:
        # print("START REQUEST")
        time.sleep(0.2)
        if params is not None:
            r = requests.get(url, headers=headers, params=params)
        else:
            r = requests.get(url, headers=headers)

        # print("GOT")
        if r.ok:
            # print(r)
            return r
        else:

            print(r)
            return None
    except Exception as e:
        print(e)
        return None


def visualize_plot(x=None, y=None, label_names=None, path="", x_name="", y_name="", x_int=True, title=""):
    for i, sub_y in enumerate(y):
        plt.plot(range(len(sub_y)) if not x else x[i], sub_y, 'o-', label=label_names[i])
    if x_int:
        new_list = range(math.floor(min(x[0])), math.ceil(max(x[0])) + 1)
        plt.xticks(new_list)
    plt.legend()
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if path: plt.savefig(path)
    plt.show()
    plt.clf()


def modify_dict_keys(d, prefix=""):
    return {f"{prefix}{k}": v for k, v in d.items()}


def get_attribute_from_data(data, attribute, return_set=False, indices=None):
    if indices is not None:
        res = [data[i][attribute] for i in indices]
    else:
        res = [it[attribute] for it in data]

    if return_set: res = set(res)
    return res


# def filter_data(data, attribute, return_set=False, indices=None):
#     if indices is not None:
#         res = [data[i][attribute] for i in indices]
#     else: res=[it[attribute] for it in data]
#
#     if return_set: res=set(res)
#     return res

def is_symmetric(g):
    return np.sum(np.abs(g.T - g)) == 0


def join(str1, str2):
    return os.path.join(str1, str2)


def get_ext(filename):
    return os.path.splitext(filename)[1]


def get_path_name(filename):
    return os.path.splitext(filename)[0]


def get_path_info(path):
    path_name, ext = os.path.splitext(path)
    directory = os.path.join(*(path.replace('\\', '/').split('/')[:-1]))
    filename, _ = os.path.splitext(os.path.split(path)[-1])

    return directory, path_name, filename, ext


def get_directory(path):
    return os.path.join(*(path.replace('\\', '/').split('/')[:-1]))


def get_filename_with_ext(filename):
    return os.path.split(filename)[-1]


def dump_file(obj, filename):
    if get_ext(filename) == ".json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif get_ext(filename) == ".pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    else:
        print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)


def dump_file_batch(objs, filenames):
    for obj, filename in zip(objs, filenames):
        dump_file(obj, filename)


def find_all_pos(my_list, target):
    return [i for i, x in enumerate(my_list) if x == target]


def path_exists(path):
    return os.path.exists(path)


def load_file(filename, init=None):
    if not path_exists(filename):
        if init is not None:
            print("file doesn't  exist, initializing")
            return init
    res=None
    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)
    elif get_ext(filename) == ".html":
        res = codecs.open(filename, 'r', encoding="utf-8")

    elif get_ext(filename) in [".pkl", ".pk"]:
        with open(filename, "rb") as r:
            res = pkl.load(r)
    elif get_ext(filename) in [".txt"]:
        with open(filename, "r", encoding="utf-8") as r:
            # print(r)
            res = r.readlines()
    elif get_ext(filename) in [".csv"]:
        res=pd.read_csv(filename)
    else:
        print("not available file type")

    return res

def load_file_batch(filenames, init=None):
    return [load_file(filename, init) for filename in filenames]


def browse_folder(path):
    return os.listdir(path)


def load_file_default(filename, init=None):
    if not path_exists(filename):
        if init == "{}": return {}
        if init == "[]": return []
        if init == 0: return 0
        if init == "": return ""
        return None
    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)
            # try:
            #     res = json.load(r)
            # except:
            #     print("here")
            #     res = [json.loads(line.strip()) for i, line in enumerate(r)]
            #     return res
            #     print(r)
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def load_file_lines(filename):
    if get_ext(filename) == ".json":
        with open(filename, mode="r", encoding="utf-8") as fin:
            res = [json.loads(line.strip()) for i, line in enumerate(fin)]
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def split_files(file_path, num_files=8):
    """
    Splits a file into multiple files of size split_size
    :param file_path:
    :param split_size:
    :return:
    """
    data = load_file(file_path)
    path_name, ext = os.path.splitext(file_path)
    split_size = math.ceil(len(data) / num_files)
    data_split = [data[(i * split_size):min((i + 1) * split_size, len(data))] for i in range(0, num_files)]
    filenames = []
    for i, split in enumerate(data_split):
        filename = path_name + "_" + str(i) + ext
        dump_file(split, filename)
        filenames.append(filename)
    return filenames


def split_to_tr_val_test_data(data, ratio="811"):
    # f = load_file_lines(path)
    f = data
    tmp = list(ratio)
    assert len(tmp) == 3
    tr_ratio, dev_ratio, te_ratio = map(lambda d: int(d) / 10, tmp)
    assert tr_ratio + dev_ratio + te_ratio == 1
    # print("tr_ratio, dev_ratio, te_ratio ", tr_ratio, dev_ratio, te_ratio)

    tr, others = train_test_split(f, test_size=1 - tr_ratio)
    dev, te = train_test_split(others, test_size=te_ratio / (dev_ratio + te_ratio))

    # np.random.shuffle(f)
    #
    # tmp = list(ratio)
    # assert len(tmp) == 3
    # tr_ratio, dev_ratio, te_ratio = map(lambda d: int(d) / 10, tmp)
    #
    # print("tr_ratio, dev_ratio, te_ratio ",tr_ratio, dev_ratio, te_ratio)
    #
    # mid1, mid2 = int(tr_ratio * len(f)), int((tr_ratio+dev_ratio) * len(f))

    return tr, dev, te


def split_to_tr_val_test(path, ratio="811", target_dir=""):
    # f = load_file_lines(path)
    f = load_file(path)
    directory, path_name, filename, ext = get_path_info(path)
    # get_filename_with_ext(path)

    # tmp = list(ratio)
    # assert len(tmp) == 3
    # tr_ratio, dev_ratio, te_ratio = map(lambda d: int(d) / 10, tmp)
    # assert tr_ratio + dev_ratio + te_ratio == 1
    # # print("tr_ratio, dev_ratio, te_ratio ", tr_ratio, dev_ratio, te_ratio)
    #
    # tr, others = train_test_split(f, test_size=1 - tr_ratio)
    # dev, te = train_test_split(others, test_size=te_ratio / (dev_ratio + te_ratio))
    tr, dev, te = split_to_tr_val_test_data(f, ratio=ratio)
    # np.random.shuffle(f)
    #
    # tmp = list(ratio)
    # assert len(tmp) == 3
    # tr_ratio, dev_ratio, te_ratio = map(lambda d: int(d) / 10, tmp)
    #
    # print("tr_ratio, dev_ratio, te_ratio ",tr_ratio, dev_ratio, te_ratio)
    #
    # mid1, mid2 = int(tr_ratio * len(f)), int((tr_ratio+dev_ratio) * len(f))
    output_dir = os.path.join(directory, target_dir)
    mkdir(output_dir)

    dump_file_batch([tr, dev, te], [f"{output_dir}/{filename}_{split}{ext}" for split in ["train", "dev", "test"]])

    return tr, dev, te
    # dump_file(tr, path_name + "_train" + ext)
    # dump_file(dev, path_name + "_dev" + ext)
    # dump_file(te, path_name + "_test" + ext)


def merge_files(base_file_path, num_files=8):
    """
    Merges multiple files into one file
    :param file_paths:
    :return:
    """
    path_name, ext = os.path.splitext(base_file_path)
    file_paths = [path_name + "_" + str(i) + ext for i in range(num_files)]
    data = []
    for file_path in file_paths:
        data += load_file(file_path)
    dump_file(data, base_file_path)
    return data


def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def module_exists(module_name):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        return True


# def get_gpu_mem_info():
#     nvmlInit()
#     h = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(h)
#     print(f'total    : {info.total}')
#     print(f'free     : {info.free}')
#     print(f'used     : {info.used}')

def check_error(input_list):
    """
    check data quality
    :param input_list: list of instances
    :return: None
    """
    for item in input_list:
        sections = item['sections']
        if not len(sections):
            print(item)
            # breakpoint()
            embed()
        for section in sections:
            if not section:
                print(item)
                embed()
            for step in section:
                if not step:
                    print(item)
                    embed()


def check_error2(input_list):
    """
    check data quality
    :param input_list: list of instances
    :return: None
    """

    for item in input_list:
        src, tgt = item["src_text"], item['tgt_text']
        if not src.replace("[SEP]", "").strip() or not tgt.replace("[SEP]", "").strip():
            print(src, "\n", tgt)
            embed()
    print("ok")


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def get_best_ckpt():
    all_ckpts = list(glob("model/states/checkpoint-*"))
    all_ckpts_ids = np.array([int(item.split("checkpoint-")[-1]) for item in all_ckpts])
    best_ckpt = all_ckpts[all_ckpts_ids.argsort()[-1]]
    print("best_ckpt", best_ckpt)
    return best_ckpt
    # best_ckpt = sorted(list(glob("model/states/checkpoint-*")), reverse=True)[0]
    # print("all ckpts", sorted(list(glob("model/states/checkpoint-*")), reverse=True))
    # best_ckpt = "model/states/checkpoint-80000"


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w') as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch
