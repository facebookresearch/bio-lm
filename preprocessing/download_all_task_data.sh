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
mkdir -p ${RAW_DATA_DIR}
mkdir -p ${BLUE_DIR}
mkdir -p ${BIOBERT_DIR}
mkdir -p ${I2B2_DIR}
mkdir -p ${MEDNLI_DIR}

## get blueBERT data:
cd ${BLUE_DIR}
wget https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/bert_data.zip
wget https://github.com/ncbi-nlp/BLUE_Benchmark/releases/download/0.1/data_v0.1.zip
unzip -q bert_data.zip
unzip -q data_v0.1.zip
cd ${HERE}

# get bioBERT data:
cd ${BIOBERT_DIR}
# get NER data
gdown --id 1OletxmPYNkz2ltOr9pyT0b0iBtUWxslh
unzip -q NERdata.zip
# get RE DATA
gdown --id 1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw
unzip -q REdata.zip
cd ${HERE}

# prepare for I2B2 data
mkdir -p ${I2B2_DIR}/i2b2-2010
mkdir -p ${I2B2_DIR}/i2b2-2012
mkdir -p ${I2B2_DIR}/i2b2-2014

