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
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider_Mixin, BaseGraphDataCollector_wikisql_Mixin
from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector


"""
Legacy: before splitting LinkPredictionDataCollector defs into new files
"""

class LinkPredictionDataCollector_ratsql(LinkPredictionDataCollector):
    def load_model(self, main_args):
        model_dict = Load_Rat_sql(root_dir=main_args.ratsql_root_dir,
            exp_config_path=main_args.ratsql_exp_config_path,
            model_dir=main_args.ratsql_model_dir,
            checkpoint_step=40000)
        
        model = model_dict['model']

        self.model = model
        self.model_dict = model_dict

        return model

    def general_fmt_dict_to_schema(self, general_fmt_dict):
        """
        Args:
            general_fmt_dict (Dict): {
                "db_id": str
                "table_names_original": List[str], original table name (concert_singer)
                "table_names_clean": List[str], clean table names (concert_singer)
                "column_names_original": List[str], original column name (singer_id)
                "column_names_clean": List[str], clean columns names (singer id)
                "column_db_full_names": List[str], name of table::column in DB (may differ from column_names) (singer::singer_id)
                "column_table_ids": List[int], for each column, the corresponding table index
                "column_types": List[str], column types
                "primary_keys": List[int], the columns indices that are primary key
                "foreign_keys": List[[int, int]], the f-p column index pairs (fk_id, pk_id)
                "sqlite_path": str
                "sqlite_conn": sqlite3.Connection
            }
        
        Return:
            db_schema (ratsql.datasets.spider.Schema)
        """
        
        db_id = general_fmt_dict["db_id"]
        db_table_orig_names = general_fmt_dict["table_names_original"]
        db_table_clean_names = general_fmt_dict["table_names_clean"]
        db_column_orig_names = general_fmt_dict["column_names_original"]
        db_column_clean_names = general_fmt_dict["column_names_clean"]
        col_db_full_names = general_fmt_dict["column_db_full_names"]
        db_column_table_ids = general_fmt_dict["column_table_ids"]
        db_column_types = general_fmt_dict["column_types"]
        db_primary_keys = general_fmt_dict["primary_keys"]
        db_foreign_keys = general_fmt_dict["foreign_keys"]
        sqlite_path = general_fmt_dict["sqlite_path"]
        sqlite_conn = general_fmt_dict["sqlite_conn"]
        
        schema_dict = {
            "column_names": list(zip(db_column_table_ids, db_column_clean_names)),
            "column_names_original": list(zip(db_column_table_ids, db_column_orig_names)),
            "column_types": db_column_types,
            "db_id": db_id,
            "foreign_keys": db_foreign_keys,
            "primary_keys": db_primary_keys,
            "table_names": db_table_clean_names,
            "table_names_original": db_table_orig_names,
        }
        
        ## Rat-sql specific 
        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                unsplit_name=name,
                orig_name=orig_name,
            )
            for i, (name, orig_name) in enumerate(zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
        )
        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                unsplit_name=col_name,
                orig_name=orig_col_name,
                type=col_type,
            )
            for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                schema_dict['column_names'],
                schema_dict['column_names_original'],
                schema_dict['column_types']))
        )

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for column_id in schema_dict['primary_keys']:
            # Register primary keys
            column = columns[column_id]
            column.table.primary_keys.append(column)

        foreign_key_graph = nx.DiGraph()
        for source_column_id, dest_column_id in schema_dict['foreign_keys']:
            # Register foreign keys
            source_column = columns[source_column_id]
            dest_column = columns[dest_column_id]
            source_column.foreign_key_for = dest_column
            foreign_key_graph.add_edge(
                source_column.table.id,
                dest_column.table.id,
                columns=(source_column_id, dest_column_id))
            foreign_key_graph.add_edge(
                dest_column.table.id,
                source_column.table.id,
                columns=(dest_column_id, source_column_id))

        db_id = schema_dict['db_id']
        schema = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
    #     eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)
        
        schema.connection = sqlite_conn  ## sqlite_conn determined above 

        return schema

    # def precompute_schemas_dict(self, orig_tables_path, db_path):
    #     raise NotImplementedError

    def get_node_encodings(self, sample, debug=False):
        db_id = self.get_sample_db_id(sample)
        db_schema = self.db_schemas_dict[db_id]
        question = sample['question']
        
        data_item = SpiderItem(
            text=None,  # intentionally None -- should be ignored when the tokenizer is set correctly
            code=None,
            schema=db_schema,
            orig_schema=db_schema.orig,
            orig={"question": question}
        )

        self.model.preproc.clear_items()
        enc_input = self.model.preproc.enc_preproc.preprocess_item(data_item, None)
        
        ## Adapted from EncDec.begin_inference
        with torch.no_grad():
            if getattr(self.model.encoder, 'batched'):
                enc_state, = self.model.encoder([enc_input])
            else:
                enc_state = self.model.encoder(enc_input)
        
        encodings = enc_state.memory.squeeze(0).detach().cpu().numpy()
        return encodings

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
        
        d = dataset_sample
        
        # db_id = d['db_id']
        db_id = self.get_sample_db_id(d)
        db_schema = self.db_schemas_dict[db_id]
        question = d['question']

        # get relation matrix
        if 'rat_sql_graph' not in dataset_sample:
            graph_dict = get_rat_sql_graph(question=question, db_schema=db_schema, model=self.model)
        else:
            graph_dict = dataset_sample['rat_sql_graph']
            graph_dict['relation_id2name'] = {v : k for k, v in self.model.encoder.encs_update.relation_ids.items()}
        
        # get encodings
        enc_repr = self.get_node_encodings(sample=d)
        
        X, y, pos = general_helpers.collect_link_prediction_samples(
            graph_dict,
            enc_repr,
            pos=pos,
            max_rel_occ=self.max_label_occ,
            debug=self.debug)
        
        return X, y, pos



class LinkPredictionDataCollector_ratsql_spider(LinkPredictionDataCollector_ratsql, BaseGraphDataCollector_spider_Mixin):
    # no extra functions needed for now
    pass


class LinkPredictionDataCollector_ratsql_wikisql(LinkPredictionDataCollector_ratsql, BaseGraphDataCollector_wikisql_Mixin):
    # no extra functions needed for now
    pass


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
    
    

