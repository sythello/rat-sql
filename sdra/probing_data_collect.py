import sys
import os
import time
import torch
# import datasets
# from transformers import (
#     HfArgumentParser,
#     set_seed,
#     AutoTokenizer
# )
# from utils.configue import Configure
# from utils.training_arguments import WrappedSeq2SeqTrainingArguments
# from models.unified.prefixtuning import Model
from argparse import ArgumentParser

import nltk

import json
from copy import deepcopy
from collections import Counter, defaultdict
import importlib
import pickle
import random

# from seq2seq_construction import spider
# from third_party.spider.preprocess.get_tables import dump_db_json_schema

import numpy as np
from tqdm import tqdm
# import editdistance
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re
from copy import deepcopy
from typing import List, Dict

# from datasets.dataset_dict import DatasetDict
# from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co

# from third_party.miscs.bridge_content_encoder import get_database_matches

from language.xsp.data_preprocessing import spider_preprocessing, wikisql_preprocessing, michigan_preprocessing

from sdr_analysis.helpers.general_helpers import _random_select_indices

from sdra import probing_data_utils as pb_utils


def main(args):
    if args.dataset == 'spider':
        probe_data_collector_cls = pb_utils.LinkPredictionDataCollector_ratsql_spider
    elif args.dataset == 'wikisql':
        probe_data_collector_cls = pb_utils.LinkPredictionDataCollector_ratsql_wikisql
    else:
        raise ValueError(args.dataset)
    
    probe_data_collector  = probe_data_collector_cls(
        orig_dataset_dir=args.orig_dataset_dir,
        graph_dataset_dir=args.graph_dataset_dir,
        probing_data_in_dir=args.probing_data_in_dir,
        probing_data_out_dir=args.probing_data_out_dir,
        max_rel_occ=args.max_rel_occ,
        ds_size=args.ds_size,
    )

    # probe_data_collector._start_idx = 495

    probe_data_collector.load_model(args)

    probe_data_collector.collect_all_probing_datasets()



if __name__ == '__main__':
    parser = ArgumentParser()

    # ratsql specific args
    parser.add_argument('-ratsql_root', '--ratsql_root_dir', type=str, required=True)
    parser.add_argument('-ratsql_config', '--ratsql_exp_config_path', type=str, required=True)
    parser.add_argument('-ratsql_model', '--ratsql_model_dir', type=str, required=True)

    # general args
    parser.add_argument('-ds', '--dataset', type=str, required=True,
        help="Which dataset; now support 'spider' and 'wikisql'.")
    # parser.add_argument('-tables_path', '--tables_path', type=str, required=True,
    #     help="Input spider tables file (tables.json)")
    # parser.add_argument('-db_path', '--db_path', type=str, required=True,
    #     help="Path to databases. spider: db dir; wikisql: db file")
    parser.add_argument('-orig_dataset_dir', '--orig_dataset_dir', type=str, required=True,
        help="Dir with original input dataset files (e.g. .../xsp/data/{dataset})")
    parser.add_argument('-graph_dataset_dir', '--graph_dataset_dir', type=str, required=True,
        help="Dir with graph-preprocessed input dataset files (e.g. .../SDR-analysis/data/{dataset})")
    parser.add_argument('-pb_in_dir', '--probing_data_in_dir', type=str, required=False,
        help="The directory with input probing data files (to load pos file from)")
    parser.add_argument('-sz', '--ds_size', type=int, required=False, default=500,
        help="Only used when no 'pb_in_dir' given. Use X samples from original dataset to collect probing samples.")
    parser.add_argument('-mo', '--max_rel_occ', type=int, required=False, default=1,
        help="Only used when no 'pb_in_dir' given. For each spider sample, include at most X probing samples per relation type.")
    parser.add_argument('-pb_out_dir', '--probing_data_out_dir', type=str, required=True,
        help="The directory to have output probing data files (for uskg)")

    args = parser.parse_args()

    main(args)





