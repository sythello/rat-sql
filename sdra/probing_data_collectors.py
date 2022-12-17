import json
import os
import sys
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
from sdr_analysis.helpers.general_helpers import SDRASampleError, _wikisql_db_id_to_table_name
from sdr_analysis.helpers.base_graph_data_collector import BaseGraphDataCollector, BaseGraphDataCollector_spider, BaseGraphDataCollector_wikisql
# from sdr_analysis.helpers.link_prediction_collector import LinkPredictionDataCollector, collect_link_prediction_samples
from sdra.probing_data_utils import Load_Rat_sql, Question, get_rat_sql_graph


class BaseGraphDataCollector_ratsql(BaseGraphDataCollector):
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

        # Rat-sql specific
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

        schema.connection = sqlite_conn  # sqlite_conn determined above

        return schema

    def get_node_encodings(self, samples, debug=False):
        """
        Args:
            samples (List[Dict]): each sample must have 'question' for user input and 'rat_sql_graph' for graph info
            pooling_func (Callable): np.array(n_pieces, dim) ==> np.array(dim,); default is np.mean
        """

        if isinstance(samples, dict):
            # back compatible: passing a single dict
            samples = [samples]

        self.model.preproc.clear_items()

        enc_inputs = []
        for sample in samples:
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

            enc_input = self.model.preproc.enc_preproc.preprocess_item(data_item, None)

            enc_inputs.append(enc_input)

        # Adapted from EncDec.begin_inference
        with torch.no_grad():
            if getattr(self.model.encoder, 'batched'):
                enc_state, = self.model.encoder(enc_inputs)
            else:
                # enc_state = self.model.encoder(enc_input)
                raise NotImplementedError

        # encodings = enc_state.memory.squeeze(0).detach().cpu().numpy()
        encodings = enc_state.memory.detach().cpu().numpy()
        valid_in_batch_ids = list(range(len(samples)))
        
        return encodings, valid_in_batch_ids


class BaseGraphDataCollector_ratsql_spider(BaseGraphDataCollector_ratsql, BaseGraphDataCollector_spider):
    # no extra functions needed for now
    pass


class BaseGraphDataCollector_ratsql_wikisql(BaseGraphDataCollector_ratsql, BaseGraphDataCollector_wikisql):
    # no extra functions needed for now
    pass
