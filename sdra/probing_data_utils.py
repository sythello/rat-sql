import json
import os, sys
from sys import modules
import re
import _jsonnet
from tqdm.notebook import tqdm
import spacy
import networkx as nx
import numpy as np
import random
import importlib
from copy import deepcopy
import editdistance
import datetime
from collections import Counter, defaultdict
import sqlite3
import time
import pickle

from sklearn.linear_model import LogisticRegression

from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize.treebank import TreebankWordDetokenizer

import torch

from ratsql.utils import registry, batched_sequence
from ratsql.commands.infer import Inferer
from ratsql.datasets.spider import SpiderDataset, SpiderItem, Column, Table, Schema

from language.xsp.data_preprocessing import spider_preprocessing, wikisql_preprocessing, michigan_preprocessing

from sdr_analysis.helpers import general_helpers
from sdr_analysis.helpers.general_helpers import SDRASampleError


def Load_Rat_sql(root_dir,
                 exp_config_path,
                 model_dir,
                 checkpoint_step=40000):

    exp_config = json.loads(_jsonnet.evaluate_file(exp_config_path))
    
    model_config_path = os.path.join(root_dir, exp_config["model_config"])
    model_config_args = exp_config.get("model_config_args")
    
    infer_config = json.loads(_jsonnet.evaluate_file(model_config_path, tla_codes={'args': json.dumps(model_config_args)}))

    inferer = Inferer(infer_config)
    inferer.device = torch.device("cpu")
    model = inferer.load_model(model_dir, checkpoint_step)
    
#     dataset = registry.construct('dataset', inferer.config['data']['val'])
#     for _, schema in dataset.schemas.items():
#         model.preproc.enc_preproc._preprocess_schema(schema)
    
    _ret_dict = {
        'model': model,
        'inferer': inferer,
    }
    
    return _ret_dict
    

def Question(q, db_schema, model_dict):
    model = model_dict['model']
    inferer = model_dict['inferer']
    
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=db_schema,
        orig_schema=db_schema.orig,
        orig={"question": q}
    )
    
    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)
    preproc_data = enc_input, None
    
    with torch.no_grad():
        return inferer._infer_one(model, data_item, preproc_data, beam_size=1, use_heuristic=True)



def get_rat_sql_graph(question, db_schema, model):
    """
    Args:
        question (str)
        db_schema (ratsql.datasets.spider.Schema): output from db_dict_to_ratsql_schema()
        model (ratsql.models.EncDec)
    
    Return:
        rat_sql_graph_dict: Dict[
            "nodes" (List[str]): the name of nodes (question toks, columns, tables)
            "relations" (np.array): the integer relation matrix, shape = (N, N) where N = #nodes
            "relation_id2name" (Dict[int, ?]): translates the integer relation to readable name (str or tuple)
        ]
    """
    
    data_item = SpiderItem(
        text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
        code=None,
        schema=db_schema,
        orig_schema=db_schema.orig,
        orig={"question": question}
    )

    model.preproc.clear_items()
    enc_input = model.preproc.enc_preproc.preprocess_item(data_item, None)

    desc = enc_input
    
    ## Adapted from SpiderEncoderV2.forward
    q_enc, _ = model.encoder.question_encoder([[desc['question']]])
    c_enc, c_boundaries = model.encoder.column_encoder([desc['columns']])
    t_enc, t_boundaries = model.encoder.table_encoder([desc['tables']])
    
    ## Adapted from RelationalTransformerUpdate.forward
    enc = batched_sequence.PackedSequencePlus.cat_seqs((q_enc, c_enc, t_enc))

    q_enc_lengths = list(q_enc.orig_lengths())
    c_enc_lengths = list(c_enc.orig_lengths())
    t_enc_lengths = list(t_enc.orig_lengths())
    enc_lengths = list(enc.orig_lengths())
    max_enc_length = max(enc_lengths)

    enc_length = enc_lengths[0]
    relations = model.encoder.encs_update.compute_relations(
        desc,
        enc_length,
        q_enc_lengths[0],
        c_enc_lengths[0],
        c_boundaries[0],
        t_boundaries[0])
    
    ## Collect nodes 
    nodes = []

    nodes.extend(enc_input['question'])

    for c_id, c_toks in enumerate(enc_input['columns']):
        c_name = '_'.join(c_toks[1:])
        t_id = enc_input['column_to_table'][str(c_id)]
        if t_id is None:
            t_name = 'NONE'
        else:
            t_toks = enc_input['tables'][t_id]
            t_name = '_'.join(t_toks)
        c_save_name = f'<C>{t_name}::{c_name}'
        nodes.append(c_save_name)

    for t_toks in enc_input['tables']:
        nodes.append('<T>' + '_'.join(t_toks))
    
    ## Get relation_id2name (a constant, just passing for convenience)
    relation_id2name = {v : k for k, v in model.encoder.encs_update.relation_ids.items()}
    
    return {
        'nodes': nodes,
        'relations': relations,
        'relation_id2name': relation_id2name,
        'q_nodes_orig': enc_input['question_for_copying'],
    }
    
    

