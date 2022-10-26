PROJ_DIR='/home/yshao/Projects'

python -m sdra.probing_data_collect \
-ratsql_root ${PROJ_DIR}/rat-sql  \
-ratsql_config ${PROJ_DIR}/rat-sql/experiments/spider-glove-run.jsonnet  \
-ratsql_model ${PROJ_DIR}/rat-sql/logdir/glove_run/bs=20,lr=7.4e-04,end_lr=0e0,att=0  \
-in_spider_dir ${PROJ_DIR}/SDR-analysis/data/spider \
-in_tables ${PROJ_DIR}/language/language/xsp/data/spider/tables.json \
-in_dbs_dir ${PROJ_DIR}/language/language/xsp/data/spider/database \
-pb_out_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/spider/ratsql \
-pb_in_dir ${PROJ_DIR}/SDR-analysis/data/probing/text2sql/link_prediction/spider/uskg
