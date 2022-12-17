import sys
import os
import time
import torch

import nltk

import json
from copy import deepcopy
from collections import Counter, defaultdict
import importlib
import pickle
import random

from ratsql.utils import registry, batched_sequence
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderDataset, SpiderItem, Column, Table, Schema

import numpy as np
from tqdm import tqdm
import editdistance
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re
from copy import deepcopy
from typing import List, Dict

# from datasets.dataset_dict import DatasetDict
# from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co

from language.xsp.data_preprocessing import spider_preprocessing, wikisql_preprocessing, michigan_preprocessing

from sdr_analysis.helpers import general_helpers
from sdr_analysis.helpers.general_helpers import SDRASampleError
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider, BaseGraphDataCollector_wikisql
from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector, collect_link_prediction_samples
from sdr_analysis.helpers.single_node_reconstruction_collector import SingleNodeReconstructionDataCollector, collect_single_node_reconstruction_samples
from sdra.probing_data_utils import get_rat_sql_graph
from sdra.probing_data_collectors import BaseGraphDataCollector_ratsql, BaseGraphDataCollector_ratsql_spider, BaseGraphDataCollector_ratsql_wikisql


class SingleNodeReconstructionDataCollector_ratsql(SingleNodeReconstructionDataCollector, BaseGraphDataCollector_ratsql):
    def extract_probing_samples_single_node_reconstruction(self, dataset_samples, pos_list=None):
        """
        Args:
            dataset_sample (Dict): a sample dict from spider dataset
            pos (List[int|None]): the positions (node-ids) to use. If none, will randomly generate        
        Return:
            X (List[np.array]): input features, "shape" = (n, (toks, dim))
            y (List[str]): output labels, "shape" = (n, words)
            pos (List[int]): actual positions (node-id)
        """

        # db_id = d['db_id']
        # db_schema = db_schemas_dict[db_id]
        # question = d['question']

        # get relation matrix (relation_id2name not available as it needs rat-sql model)
        # graph_dict = d['rat_sql_graph']
        # graph_dict['relation_id2name'] = {v : k for k, v in model.encoder.encs_update.relation_ids.items()}

        # Get encodings
        # all_enc_repr: "shape" = (bsz, nodes, (toks, dim)); keep each sub-token encoding, no pooling; bsz = num of valid samples
        # valid_in_batch_ids: List[int], which samples in batch are valid, len = num of valid samples
        all_enc_repr, valid_in_batch_ids = self.get_node_encodings(samples=dataset_samples) 

        all_X = []
        all_y = []
        all_pos = []

        # for in_batch_idx, d in enumerate(dataset_samples):
        for in_batch_idx, enc_repr in zip(valid_in_batch_ids, all_enc_repr):
            d = dataset_samples[in_batch_idx]

            graph_dict = d['rat_sql_graph']
            # enc_repr = all_enc_repr[in_batch_idx]
            sample_pos = None if pos_list is None else pos_list[in_batch_idx]

            X, y, pos = collect_single_node_reconstruction_samples(
                graph_dict,
                enc_repr,
                pos=sample_pos,
                max_node_type_occ=self.max_label_occ,
                debug=self.debug)
            
            all_X.extend(X)
            all_y.extend(y)
            all_pos.extend([(in_batch_idx, node_idx) for node_idx in pos])

        return all_X, all_y, all_pos
    

class SingleNodeReconstructionDataCollector_ratsql_spider(SingleNodeReconstructionDataCollector_ratsql, BaseGraphDataCollector_ratsql_spider):
    pass


class SingleNodeReconstructionDataCollector_ratsql_wikisql(SingleNodeReconstructionDataCollector_ratsql, BaseGraphDataCollector_ratsql_wikisql):
    pass

