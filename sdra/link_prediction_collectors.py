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


class LinkPredictionDataCollector_ratsql(LinkPredictionDataCollector, BaseGraphDataCollector_ratsql):
    def extract_probing_samples_link_prediction(self, dataset_sample, pos=None):
        """
        Args:
            dataset_sample (Dict): a sample dict from spider dataset
            pos (List[Tuple]): the position pairs to use. If none, will randomly generate

        Return:
            X (List[np.array]): input features, "shape" = (n, dim)
            y (List): output labels, "shape" = (n,)
            pos (List[Tuple]): actual position (node-id) pairs for X and y
        """

        # TODO (later): add a batched version

        d = dataset_sample

        # db_id = d['db_id']
        db_id = self.get_sample_db_id(d)
        db_schema = self.db_schemas_dict[db_id]
        question = d['question']

        # get relation matrix
        if 'rat_sql_graph' not in dataset_sample:
            graph_dict = get_rat_sql_graph(
                question=question, db_schema=db_schema, model=self.model)
        else:
            graph_dict = dataset_sample['rat_sql_graph']
            graph_dict['relation_id2name'] = {
                v: k for k, v in self.model.encoder.encs_update.relation_ids.items()}

        # get encodings
        # enc_repr = self.get_node_encodings(sample=d)
        enc_repr = self.get_node_encodings([d])[0][0]

        X, y, pos = collect_link_prediction_samples(
            graph_dict,
            enc_repr,
            pos=pos,
            max_rel_occ=self.max_label_occ,
            debug=self.debug)

        return X, y, pos


class LinkPredictionDataCollector_ratsql_spider(LinkPredictionDataCollector_ratsql, BaseGraphDataCollector_ratsql_spider):
    # no extra functions needed for now
    pass


class LinkPredictionDataCollector_ratsql_wikisql(LinkPredictionDataCollector_ratsql, BaseGraphDataCollector_ratsql_wikisql):
    # no extra functions needed for now
    pass
