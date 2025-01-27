PROJ_DIR='/home/yshao/Projects'
DATASET='spider'
PROBE_TASK='single_node_reconstruction'
# PROBE_TASK='link_prediction'

python -m sdra.probing_data_collect \
-ratsql_root ${PROJ_DIR}/rat-sql  \
-ratsql_config ${PROJ_DIR}/rat-sql/experiments/spider-glove-run.jsonnet  \
-ratsql_model ${PROJ_DIR}/rat-sql/logdir/glove_run/bs=20,lr=7.4e-04,end_lr=0e0,att=0  \
-ds ${DATASET} \
-probe_task ${PROBE_TASK} \
-orig_dataset_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
-graph_dataset_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET} \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/ratsql \
-enc_bsz 1

# -pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/${PROBE_TASK}/${DATASET}/uskg

