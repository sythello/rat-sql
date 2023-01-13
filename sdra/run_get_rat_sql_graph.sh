PROJ_DIR='/home/yshao/Projects'
DATASET='spider'

python -m sdra.get_rat_sql_graph \
-ds ${DATASET} \
-ratsql_root ${PROJ_DIR}/rat-sql  \
-ratsql_config ${PROJ_DIR}/rat-sql/experiments/spider-glove-run.jsonnet  \
-ratsql_model ${PROJ_DIR}/rat-sql/logdir/glove_run/bs=20,lr=7.4e-04,end_lr=0e0,att=0  \
-input_dir ${PROJ_DIR}/language/language/xsp/data/${DATASET} \
-output_dir ${PROJ_DIR}/SDR-analysis/data/${DATASET}
