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


MODEL=${1}
MAXLEN=512

# Process BioBERT datasets
DATASETS="BC2GM BC4CHEMD JNLPBA linnaeus NCBI-disease s800"
#DATASETS="s800"

echo ${MODEL}

for TASK in ${DATASETS}
do
    echo "#######################################"
    echo "##### Processing ${TASK} #####"
    echo "#######################################"


    TASK_DIR=${TASKS_DIR}/${TASK}
    PROCESSED_DATA_DIR=${TASKS_DIR}/${TASK}.model=${MODEL}.maxlen=${MAXLEN}
    mkdir -p ${TASK_DIR}
    mkdir -p ${PROCESSED_DATA_DIR}
    sed 's/\t/ /g' ${BIOBERT_DIR}/${TASK}/train.tsv > ${TASK_DIR}/train.txt.conll
    sed 's/\t/ /g' ${BIOBERT_DIR}/${TASK}/train_dev.tsv > ${TASK_DIR}/train_dev.txt.conll
    sed 's/\t/ /g' ${BIOBERT_DIR}/${TASK}/devel.tsv > ${TASK_DIR}/dev.txt.conll
    sed 's/\t/ /g' ${BIOBERT_DIR}/${TASK}/test.tsv > ${TASK_DIR}/test.txt.conll

    for split in "train" "dev" "train_dev" "test"
    do
        python3 preprocessing/clean_conll_file.py \
        --filename "${TASK_DIR}/${split}.txt.conll" \
        --model_name_or_path ${MODEL}\
        --max_len ${MAXLEN} > "${PROCESSED_DATA_DIR}/${split}.txt"
    done

    cat "${PROCESSED_DATA_DIR}/train.txt" \
       "${PROCESSED_DATA_DIR}/dev.txt"\
      "${PROCESSED_DATA_DIR}/test.txt" \
       "${PROCESSED_DATA_DIR}/train_dev.txt" \
    | cut -d " " -f 2 | grep -v "^$"| sort | uniq > ${PROCESSED_DATA_DIR}/labels1.txt
     echo -e "B\nI\nO" > ${PROCESSED_DATA_DIR}/labels.txt

done


# Process BlueBERT datasets
DATASETS="disease chem"
for TASK in ${DATASETS}
do

    echo "#######################################"
    echo "##### Processing BC5CDR-${TASK} #####"
    echo "#######################################"
    TASK_DIR=${TASKS_DIR}/BC5CDR-${TASK}
    PROCESSED_DATA_DIR=${TASKS_DIR}/BC5CDR-${TASK}.model=${MODEL}.maxlen=${MAXLEN}

    mkdir -p ${TASK_DIR}
    mkdir -p ${PROCESSED_DATA_DIR}
    sed 's/\t/ /g' ${BLUE_DIR}/bert_data/BC5CDR/${TASK}/train.tsv > ${TASK_DIR}/train.txt.conll
    sed 's/\t/ /g' ${BLUE_DIR}/bert_data/BC5CDR/${TASK}/devel.tsv > ${TASK_DIR}/dev.txt.conll
    sed 's/\t/ /g' ${BLUE_DIR}/bert_data/BC5CDR/${TASK}/test.tsv > ${TASK_DIR}/test.txt.conll

    for split in "train" "dev" "test"
    do
        python3 preprocessing/clean_conll_file.py\
                --filename "${TASK_DIR}/${split}.txt.conll"\
                --model_name_or_path ${MODEL}\
                --max_len ${MAXLEN} > "${PROCESSED_DATA_DIR}/${split}.txt"
    done

    cat "${PROCESSED_DATA_DIR}/train.txt" \
       "${PROCESSED_DATA_DIR}/dev.txt"\
      "${PROCESSED_DATA_DIR}/test.txt" \
    | cut -d " " -f 4 | grep -v "^$"| sort | uniq > ${PROCESSED_DATA_DIR}/labels.txt

done




# Process I2B2-2010
if test -d "${I2B2_DIR}/i2b2-2010/reference_standard_for_test_data"; then
    TASK="I2B22010NER"
    echo "#######################################"
    echo "##### Processing ${TASK} #####"
    echo "#######################################"
    TASK_DIR=${TASKS_DIR}/${TASK}
    PROCESSED_DATA_DIR=${TASKS_DIR}/${TASK}.model=${MODEL}.maxlen=${MAXLEN}
    mkdir -p ${TASK_DIR}
    mkdir -p ${PROCESSED_DATA_DIR}
    I2B2_2010_DIR=${I2B2_DIR}/i2b2-2010


    python preprocessing/preprocess_i2b2_2010_ner.py \
    --beth_dir ${I2B2_2010_DIR}/concept_assertion_relation_training_data/beth/ \
    --partners_dir ${I2B2_2010_DIR}/concept_assertion_relation_training_data/partners/ \
    --test_dir ${I2B2_2010_DIR}/reference_standard_for_test_data/ \
    --test_txt_dir ${I2B2_2010_DIR}/test_data/ \
    --task_dir ${TASK_DIR}\

    cp ${TASK_DIR}/merged/train.tsv  ${TASK_DIR}/train.txt.conll
    cp ${TASK_DIR}/merged/dev.tsv ${TASK_DIR}/dev.txt.conll
    cp ${TASK_DIR}/merged/test.tsv ${TASK_DIR}/test.txt.conll

    for split in "train" "dev" "test"
    do
       python3 preprocessing/clean_conll_file.py \
        --filename "${TASK_DIR}/${split}.txt.conll"\
        --model_name_or_path ${MODEL} \
        --max_len ${MAXLEN} > "${PROCESSED_DATA_DIR}/${split}.txt"
       done

    cat "${PROCESSED_DATA_DIR}/train.txt" \
       "${PROCESSED_DATA_DIR}/dev.txt"\
      "${PROCESSED_DATA_DIR}/test.txt" \
    | cut -d " " -f 2 | grep -v "^$"| sort | uniq > ${PROCESSED_DATA_DIR}/labels.txt

fi

# Process I2B2-2012
if test -d "${I2B2_DIR}/i2b2-2012/2012-07-15.original-annotation.release"; then
    TASK="I2B22012NER"
    echo "#######################################"
    echo "##### Processing ${TASK} #####"
    echo "#######################################"

    TASK_DIR=${TASKS_DIR}/${TASK}
    PROCESSED_DATA_DIR=${TASKS_DIR}/${TASK}.model=${MODEL}.maxlen=${MAXLEN}
    mkdir -p ${TASK_DIR}
    mkdir -p ${PROCESSED_DATA_DIR}
    I2B2_2012_DIR=${I2B2_DIR}/i2b2-2012


    python preprocessing/preprocess_i2b2_2012_ner.py \
    --raw_data_dir ${I2B2_2012_DIR} \
    --task_dir ${TASK_DIR}


    for split in "train" "dev" "test"
    do
       python3 preprocessing/clean_conll_file.py \
        --filename "${TASK_DIR}/${split}.txt.conll"\
        --model_name_or_path ${MODEL} \
        --max_len ${MAXLEN} > "${PROCESSED_DATA_DIR}/${split}.txt"
       done

    cat "${PROCESSED_DATA_DIR}/train.txt" \
       "${PROCESSED_DATA_DIR}/dev.txt"\
      "${PROCESSED_DATA_DIR}/test.txt" \
    | cut -d " " -f 2 | grep -v "^$"| sort | uniq > ${PROCESSED_DATA_DIR}/labels.txt

fi



# Process I2B2-2014
if test -d "${I2B2_DIR}/i2b2-2014/testing-PHI-Gold-fixed/"; then
    TASK="I2B22014NER"
    echo "#######################################"
    echo "##### Processing ${TASK} #####"
    echo "#######################################"

    TASK_DIR=${TASKS_DIR}/${TASK}
    PROCESSED_DATA_DIR=${TASKS_DIR}/${TASK}.model=${MODEL}.maxlen=${MAXLEN}
    mkdir -p ${TASK_DIR}
    mkdir -p ${PROCESSED_DATA_DIR}
    I2B2_2014_DIR=${I2B2_DIR}/i2b2-2014

    python preprocessing/preprocess_i2b2_2014_ner.py \
        --gold_set_1_dir ${I2B2_2014_DIR}/training-PHI-Gold-Set1 \
        --gold_set_2_dir ${I2B2_2014_DIR}/training-PHI-Gold-Set2 \
        --test_gold_set_dir ${I2B2_2014_DIR}/testing-PHI-Gold-fixed \
        --task_dir ${TASK_DIR}


    for split in "train" "dev" "test"
    do
       python3 preprocessing/clean_conll_file.py \
        --filename "${TASK_DIR}/${split}.txt.conll"\
        --model_name_or_path ${MODEL} \
        --max_len ${MAXLEN} > "${PROCESSED_DATA_DIR}/${split}.txt"
       done

    cat "${PROCESSED_DATA_DIR}/train.txt" \
       "${PROCESSED_DATA_DIR}/dev.txt"\
      "${PROCESSED_DATA_DIR}/test.txt" \
    | cut -d " " -f 2 | grep -v "^$"| sort | uniq > ${PROCESSED_DATA_DIR}/labels.txt

fi


