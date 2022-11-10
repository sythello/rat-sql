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
    os.makedirs(args.output_dir, exist_ok=True)

    rat_sql_model_dict = pb_utils.Load_Rat_sql(root_dir=args.ratsql_root_dir,
                                    exp_config_path=args.ratsql_exp_config_path,
                                    model_dir=args.ratsql_model_dir,
                                    checkpoint_step=40000)
    model = rat_sql_model_dict['model']

    if args.dataset == 'spider':
        data_collector = pb_utils.LinkPredictionDataCollector_ratsql_spider()
        paths_dict = {
            orig_ds : {
                'orig_dataset_path': os.path.join(args.input_dir, f'{orig_ds}.json'),
                'orig_tables_path': os.path.join(args.input_dir, 'tables.json'),
                'db_path': os.path.join(args.input_dir, 'database'),    # For spider, db_path is database root dir
            }
            for orig_ds in ['train', 'dev']
        }
    elif args.dataset == 'wikisql':
        data_collector = pb_utils.LinkPredictionDataCollector_ratsql_wikisql()
        paths_dict = {
            orig_ds : {
                'orig_dataset_path': os.path.join(args.input_dir, f'{orig_ds}.jsonl'),
                'orig_tables_path': os.path.join(args.input_dir, f'{orig_ds}.tables.jsonl'),
                'db_path': os.path.join(args.input_dir, f'{orig_ds}.db'),    # For wikisql, db_path is the db file path
            }
            for orig_ds in ['train', 'dev', 'test']
        }
    else:
        raise ValueError(args.dataset)

    for orig_ds, path_dict in paths_dict.items():
        orig_dataset_path = path_dict['orig_dataset_path']
        orig_tables_path = path_dict['orig_tables_path']
        db_path = path_dict['db_path']

        output_dataset_path = os.path.join(args.output_dir, f"{orig_ds}+ratsql_graph.json")

        db_schemas_dict = data_collector.precompute_schemas_dict(
            orig_tables_path=orig_tables_path,
            db_path=db_path)

        with open(orig_dataset_path, 'r') as f:
            ## TODO: refactor data_collector so we don't need to do so many if-else here 
            if orig_dataset_path.endswith('.jsonl'):
                orig_dataset = [json.loads(l) for l in f]
            else:
                orig_dataset = json.load(f)

        model = rat_sql_model_dict['model']

        for d in tqdm(orig_dataset, ascii=True):
            db_id = d['db_id'] if args.dataset == 'spider' else d['table_id']
            db_schema = db_schemas_dict[db_id]
            question = d['question']

            # get relation matrix
            graph_dict = pb_utils.get_rat_sql_graph(question=question, db_schema=db_schema, model=model)
            nodes = graph_dict['nodes']
            q_nodes_orig = graph_dict['q_nodes_orig']
            relations = json.dumps(graph_dict['relations'].tolist(), indent=None)  # dump to a line to save space in json
            
            d['rat_sql_graph'] = {
                'nodes': nodes,
                'q_nodes_orig': q_nodes_orig,
                'relations': relations
            }
            time.sleep(0.2)

        with open(output_dataset_path, 'w') as f:
            json.dump(orig_dataset, f, indent=2)




if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-ratsql_root', '--ratsql_root_dir', type=str, required=True)
    parser.add_argument('-ratsql_config', '--ratsql_exp_config_path', type=str, required=True)
    parser.add_argument('-ratsql_model', '--ratsql_model_dir', type=str, required=True)

    parser.add_argument('-ds', '--dataset', type=str, required=True,
        help="Which dataset; now support 'spider' and 'wikisql'.")

    parser.add_argument('-input_dir', '--input_dir', type=str, required=True,
        help="Root dir of input dataset (e.g. .../language/xsp/data/wikisql")
    
    parser.add_argument('-output_dir', '--output_dir', type=str, required=True,
        help="Dir to put the output files (e.g. .../SDR-analysis/data/wikisql)")

    # dataset = 'wikisql'
    # orig_ds = 'dev'
    # orig_dataset_file = f"/Users/mac/Desktop/syt/Deep-Learning/Repos/Google-Research-Language/language/language/xsp/data/wikisql/{orig_ds}.jsonl"
    # orig_tables_file = f"/Users/mac/Desktop/syt/Deep-Learning/Repos/Google-Research-Language/language/language/xsp/data/wikisql/{orig_ds}.tables.jsonl"
    # db_file = f"/Users/mac/Desktop/syt/Deep-Learning/Repos/Google-Research-Language/language/language/xsp/data/wikisql/{orig_ds}.db"

    # output_dir = "/Users/mac/Desktop/syt/Deep-Learning/Projects-M/SDR-analysis/data/wikisql"

    args = parser.parse_args()

    main(args)





