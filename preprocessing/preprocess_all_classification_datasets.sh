# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

HERE=${PWD}
DATA_DIR=${HERE}/data
RAW_DATA_DIR=${DATA_DIR}/raw_data
BIOBERT_DIR=${RAW_DATA_DIR}/biobert_data
BLUE_DIR=${RAW_DATA_DIR}/blue_raw_data
I2B2_DIR=${RAW_DATA_DIR}/i2b2_raw_data
MEDNLI_DIR=${RAW_DATA_DIR}/mednli_raw_data
TASKS_DIR=${DATA_DIR}/tasks
mkdir -p ${TASKS_DIR}


# Process BioBERT datasets:
echo "#######################################"
echo "##### Processing euadr #####"
echo "#######################################"
cp -r ${BIOBERT_DIR}/euadr ${TASKS_DIR}
echo "#######################################"
echo "##### Processing GAD #####"
echo "#######################################"
cp -r ${BIOBERT_DIR}/GAD ${TASKS_DIR}

# Process BLUEBERT datasets
echo "#######################################"
echo "##### Processing ChemProt #####"
echo "#######################################"
TASK=ChemProt
mkdir -p ${TASKS_DIR}/${TASK}
cp ${BLUE_DIR}/bert_data/ChemProt/train.tsv ${TASKS_DIR}/${TASK}/train.tsv
cp ${BLUE_DIR}/bert_data/ChemProt/dev.tsv ${TASKS_DIR}/${TASK}/dev.tsv
cp ${BLUE_DIR}/bert_data/ChemProt/test.tsv ${TASKS_DIR}/${TASK}/test.tsv

echo "#######################################"
echo "##### Processing HOC #####"
echo "#######################################"
TASK=HOC
mkdir -p ${TASKS_DIR}/${TASK}
cp ${BLUE_DIR}/bert_data/hoc/train.tsv ${TASKS_DIR}/${TASK}/train.tsv
cp ${BLUE_DIR}/bert_data/hoc/dev.tsv ${TASKS_DIR}/${TASK}/dev.tsv
cp ${BLUE_DIR}/bert_data/hoc/test.tsv ${TASKS_DIR}/${TASK}/test.tsv

echo "#######################################"
echo "##### Processing DDI #####"
echo "#######################################"
TASK=DDI
mkdir -p ${TASKS_DIR}/${TASK}
cp ${BLUE_DIR}/bert_data/ddi2013-type/train.tsv ${TASKS_DIR}/${TASK}/train.tsv
cp ${BLUE_DIR}/bert_data/ddi2013-type/dev.tsv ${TASKS_DIR}/${TASK}/dev.tsv
cp ${BLUE_DIR}/bert_data/ddi2013-type/test.tsv ${TASKS_DIR}/${TASK}/test.tsv


# Process I2B2-RE
TASK="I2B21010RE"
if test -d "${I2B2_DIR}/i2b2-2010/reference_standard_for_test_data"; then
    echo "#######################################"
    echo "##### Processing I2B21010RE #####"
    echo "#######################################"
    mkdir -p ${TASKS_DIR}/${TASK}
    I2B2_2010_DIR=${I2B2_DIR}/i2b2-2010

    cp ${BLUE_DIR}/data/i2b2-2010/train-docids.txt ${I2B2_2010_DIR}/
    cp ${BLUE_DIR}/data/i2b2-2010/dev-docids.txt ${I2B2_2010_DIR}/
    cp -r ${I2B2_2010_DIR}/test_data  ${I2B2_2010_DIR}/reference_standard_for_test_data/txt/
    cp -r  ${I2B2_2010_DIR}/reference_standard_for_test_data/concepts/ ${I2B2_2010_DIR}/reference_standard_for_test_data/concept/
    mkdir -p ${I2B2_2010_DIR}/original/
    cp -r  ${I2B2_2010_DIR}/reference_standard_for_test_data ${I2B2_2010_DIR}/original/
    cp -r  ${I2B2_2010_DIR}/concept_assertion_relation_training_data ${I2B2_2010_DIR}/original/

    cd BLUE_Benchmark

    PYTHONPATH='.:blue/.' python blue/gs/create_i2b2_test_gs.py \
        --input_dir ${I2B2_2010_DIR}/reference_standard_for_test_data \
        --output_dir ${I2B2_2010_DIR}

    PYTHONPATH='.:blue/.' python blue/bert/create_i2b2_bert.py  \
        --gold_directory ${I2B2_2010_DIR} \
        --output_directory ${I2B2_2010_DIR}

    cp ${I2B2_2010_DIR}/train.tsv ${TASKS_DIR}/${TASK}/train.tsv
    cp ${I2B2_2010_DIR}/dev.tsv ${TASKS_DIR}/${TASK}/dev.tsv
    cp ${I2B2_2010_DIR}/test.tsv ${TASKS_DIR}/${TASK}/test.tsv

    # make a bigger dev set
    head -n 21385 ${TASKS_DIR}/${TASK}/train.tsv > ${TASKS_DIR}/${TASK}/train_new.tsv
    cp ${TASKS_DIR}/${TASK}/dev.tsv  ${TASKS_DIR}/${TASK}/dev_new.tsv
    tail -n 776  ${TASKS_DIR}/${TASK}/train.tsv >> ${TASKS_DIR}/${TASK}/dev_new.tsv

    cd ${HERE}

fi

# Process MedNLI
TASK=MedNLI
mkdir -p ${TASKS_DIR}/${TASK}
if test -f "${MEDNLI_DIR}/mli_train_v1.jsonl"; then
    echo "#######################################"
    echo "##### Processing MedNLI #####"
    echo "#######################################"
    cp ${MEDNLI_DIR}/mli_train_v1.jsonl ${TASKS_DIR}/${TASK}/mli_train_v1.jsonl
    cp ${MEDNLI_DIR}/mli_dev_v1.jsonl ${TASKS_DIR}/${TASK}/mli_dev_v1.jsonl
    cp ${MEDNLI_DIR}/mli_test_v1.jsonl ${TASKS_DIR}/${TASK}/mli_test_v1.jsonl
fi
