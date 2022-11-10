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


def collect_probing_dataset(args,
        model,
        probe_data_collector,
        db_schemas_dict,
        orig_dataset,
        orig_ds,
        prob_ds,
        **extra_kwargs):

    if args.probing_data_in_dir is not None:
        pos_file_path = os.path.join(args.probing_data_in_dir, f'{orig_ds}.{prob_ds}.pos.txt')
    else:
        pos_file_path = None

    if pos_file_path is not None:
        with open(pos_file_path, 'r') as f:
            lines = f.read().strip().split('\n')
            all_pos_triplets = [tuple([int(s) for s in l.split('\t')]) for l in lines]
        # len(all_pos_triplets), all_pos_triplets[0]

        sample_ds_indices = []               # [ds_idx], based on occurring order 
        pos_per_sample = defaultdict(list)   # key = ds_idx, value = pos_list: List[(i, j)]

        for ds_idx, i, j in all_pos_triplets:
            if not sample_ds_indices or sample_ds_indices[-1] != ds_idx:
                sample_ds_indices.append(ds_idx)
            pos_per_sample[ds_idx].append((i, j))
        # len(sample_ds_indices), len(pos_per_sample)
        print(f'Loaded pos file from {pos_file_path}: {len(all_pos_triplets)} triplets, {len(sample_ds_indices)} orig samples')
    else:
        pos_per_sample = defaultdict(lambda: None)
        sample_ds_indices = _random_select_indices(orig_len=len(orig_dataset), k=args.ds_size, ds=prob_ds, seed=42)
        print(f'Generated pos: {len(sample_ds_indices)} orig samples')


    all_X = []
    all_y = []
    all_pos = []

    for sample_ds_idx in tqdm(sample_ds_indices, ascii=True):
        dataset_sample = orig_dataset[sample_ds_idx]
        pos_list = pos_per_sample[sample_ds_idx]

        X, y, pos = probe_data_collector.extract_probing_samples_link_prediction(
            dataset_sample=dataset_sample,
            db_schemas_dict=db_schemas_dict,
            model=model,
            pos=pos_list,
            max_rel_occ=args.max_occ,  # when given pos, this is not needed 
            debug=False)
        
        all_X.extend(X)
        all_y.extend(y)
        pos = [(sample_ds_idx, i, j) for i, j in pos]   # add sample idx 
        all_pos.extend(pos)

        # time.sleep(0.2)
    # len(all_X), len(all_y), len(all_pos)

    # probing_data_out_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg"
    probing_data_out_dir = args.probing_data_out_dir
    os.makedirs(probing_data_out_dir, exist_ok=True)

    output_path_test_X = os.path.join(probing_data_out_dir, f'{orig_ds}.{prob_ds}.X.pkl')
    output_path_test_y = os.path.join(probing_data_out_dir, f'{orig_ds}.{prob_ds}.y.pkl')
    output_path_test_pos = os.path.join(probing_data_out_dir, f'{orig_ds}.{prob_ds}.pos.txt')

    with open(output_path_test_X, 'wb') as f:
        pickle.dump(all_X, f)
    with open(output_path_test_y, 'wb') as f:
        pickle.dump(all_y, f)
    with open(output_path_test_pos, 'w') as f:
        for idx, i, j in all_pos:
            f.write(f'{idx}\t{i}\t{j}\n')



def main(args):
    rat_sql_model_dict = pb_utils.Load_Rat_sql(root_dir=args.ratsql_root_dir,
                                      exp_config_path=args.ratsql_exp_config_path,
                                      model_dir=args.ratsql_model_dir,
                                      checkpoint_step=40000)
    model = rat_sql_model_dict['model']

    # probing_data_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/probing/text2sql/link_prediction/spider/ratsql"
    # probing_data_dir = args.probing_data_dir

    # xsp_data_dir = "/Users/mac/Desktop/syt/Deep-Learning/Repos/Google-Research-Language/language/language/xsp/data"
    # spider_tables_path = os.path.join(xsp_data_dir, 'spider', 'tables.json')

    if args.dataset == 'spider':
        probe_data_collector = pb_utils.LinkPredictionDataCollector_ratsql_spider()
    elif args.dataset == 'wikisql':
        probe_data_collector = pb_utils.LinkPredictionDataCollector_ratsql_wikisql()
    else:
        raise ValueError(args.dataset)

    db_schemas_dict = probe_data_collector.precompute_schemas_dict(
        orig_tables_path=args.tables_path,
        db_path=args.db_path)

    kwargs = {
        'model': model,
        'probe_data_collector': probe_data_collector,
        'db_schemas_dict': db_schemas_dict,
    }

    orig_ds_list = ['train', 'dev']
    prob_ds_list = ['train', 'test']
    for orig_ds in orig_ds_list:
        # dataset_path = f"/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/spider/{orig_ds}+ratsql_graph.json"
        dataset_path = os.path.join(args.dataset_dir, f'{orig_ds}+ratsql_graph.json')
        with open(dataset_path, 'r') as f:
            orig_dataset = json.load(f)
        for d in orig_dataset:
            d['rat_sql_graph']['relations'] = json.loads(d['rat_sql_graph']['relations'])

        kwargs['orig_ds'] = orig_ds
        kwargs['orig_dataset'] = orig_dataset

        for prob_ds in prob_ds_list:
            kwargs['prob_ds'] = prob_ds

            collect_probing_dataset(args, **kwargs)



if __name__ == '__main__':
    parser = ArgumentParser()

    # ratsql specific args
    parser.add_argument('-ratsql_root', '--ratsql_root_dir', type=str, required=True)
    parser.add_argument('-ratsql_config', '--ratsql_exp_config_path', type=str, required=True)
    parser.add_argument('-ratsql_model', '--ratsql_model_dir', type=str, required=True)

    # general args
    parser.add_argument('-ds', '--dataset', type=str, required=True,
        help="Which dataset; now support 'spider' and 'wikisql'.")

    parser.add_argument('-dataset_dir', '--dataset_dir', type=str, required=True,
        help="Dir with input spider dataset files (xxx+rat_sql_graph.json)")
    parser.add_argument('-tables_path', '--tables_path', type=str, required=True,
        help="Input spider tables file (tables.json)")
    parser.add_argument('-db_path', '--db_path', type=str, required=True,
        help="Path to databases. spider: db dir; wikisql: db file")
    parser.add_argument('-pb_in_dir', '--probing_data_in_dir', type=str, required=False,
        help="The directory with input probing data files (to load pos file from)")
    parser.add_argument('-sz', '--ds_size', type=int, required=False, default=500,
        help="Only used when no 'pb_in_dir' given. Use X samples from original dataset to collect probing samples.")
    parser.add_argument('-mo', '--max_occ', type=int, required=False, default=1,
        help="Only used when no 'pb_in_dir' given. For each spider sample, include at most X probing samples per relation type.")

    parser.add_argument('-pb_out_dir', '--probing_data_out_dir', type=str, required=True,
        help="The directory to have output probing data files (for uskg)")

    args = parser.parse_args()

    main(args)





