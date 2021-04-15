# Biomedical and Clinical Language Models

This repository contains code and modeks to support the research paper [Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art](https://www.aclweb.org/anthology/2020.clinicalnlp-1.17/)

<br>
<p align="center">
  <img src="https://dl.fbaipublicfiles.com/MLQA/logos.png" alt="Facebook AI Research and UCL NLP"  width="60%"/>
  <br>
</p>
<br>


## Models

Pytorcch Model checkpoints are available to download below in fairseq and 🤗 Transformers format .

The overall-best RoBERTa-Large sized model from our experiments is `RoBERTa-large-PM-M3-Voc`, 
and the overall-best RoBERTa-based size model is `RoBERTa-base-PM-M3-Voc-distill-align`.


| Model | Size | Description | 🤗 Transformers Link | fairseq link | 
| ------------- | ------------- | --------- | ----|  --- | 
| RoBERTa-base-PM     | base | Pre-trained on PubMed and PMC | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-fairseq.tar.gz) | 
| RoBERTa-base-PM-Voc  | base   | Pre-trained on PubMed and PMC with a BPE Vocab learnt from PubMed| [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-Voc-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-Voc-fairseq.tar.gz) | 
| RoBERTa-base-PM-M3  |  base | Pre-trained on PubMed and PMC and MIMIC-III | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-fairseq.tar.gz) | 
| RoBERTa-base-PM-M3-Voc | base   | Pre-trained on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed| [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-fairseq.tar.gz) | 
| RoBERTa-base-PM-M3-Voc-train-longer | base   | Pre-trained on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed with an additional 50K steps | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-train-longer-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-train-longer-fairseq.tar.gz) | 
| RoBERTa-base-PM-M3-Voc-distill | base   | Base-sized model distilled from RoBERTa-large-PM-M3-Voc | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-fairseq.tar.gz) | 
| RoBERTa-base-PM-M3-Voc-distill-align | base   | Base-sized model distilled from RoBERTa-large-PM-M3-Voc with additional alignment objective| [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-base-PM-M3-Voc-distill-align-fairseq.tar.gz) | 
| RoBERTa-large-PM-M3 | large  | Pre-trained on PubMed and PMC and MIMIC-III| [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-fairseq.tar.gz) | 
| RoBERTa-large-PM-M3-Voc | large  | Pre-trained on PubMed and PMC and MIMIC-III with a BPE Vocab learnt from PubMed| [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz) | [download](https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-fairseq.tar.gz) | 

To use these models in 🤗 Transformers, (developed using Transformers version 2.6.0), download the desired model, untar it, 
and use an `AutoModel` class to load it, passing the path to the model_directory, as shown in the snippet below:

```bash
$ wget https://link/to/RoBERTa-base-PM-hf.tar.gz
$ tar -zxvf RoBERTa-base-PM-hf.tar.gz
$ python
>>> from transformers import AutoModel, AutoConfig
>>> config = AutoConfig.from_pretrained(
    "RoBERTa-base-PM-hf",
)
>>> model = AutoModel.from_pretrained("RoBERTa-base-PM-hf", config=config)
```
## Code

### Installation and Dependencies

We recommend using conda, python 3.7, and pytorch 1.4 with cuda 10.1 to match our development environment:

```bash
conda create -y -n bio-lm-env python=3.7
conda activate bio-lm-env
conda install -y pytorch==1.4 torchvision cudatoolkit=10.1 -c pytorch

# Optional, but recommended: To use fp16 training, install Apex:
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./
cd ../


# install other dependencies:
conda install scikit-learn
conda install pandas
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install transformers, and check out the appropriate commit to match our development environment
git clone git@github.com:huggingface/transformers.git
cd transformers
git reset --hard 601ac5b1dc1438f00d09696588f2deb0f045ae3b
pip install -e .
cd ..

# get BLUE Benchmark, needed for some preprocessing
git clone git@github.com:ncbi-nlp/BLUE_Benchmark.git
cd BLUE_Benchmark
git reset --hard b6216f2cb9bba209ee7028fc874123d8fd5a810c
cd ..


# get official conllevalpy
wget https://raw.githubusercontent.com/spyysalo/conlleval.py/master/conlleval.py
```

### Downloading Raw Task data:

Data Preprocessing and data is set up to match the approaches in [BLUE](https://github.com/ncbi-nlp/BLUE_Benchmark), [BioBERT](https://github.com/dmis-lab/biobert) and [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT).
The following code will download the datasets and preprocess them appropriately. 

First, download the raw data by running 
```
bash preprocessing/download_all_task_data.sh
```

This will download the raw task data from BLUE and BioBERT, and place them in `<project_root>/data/raw_data`.

There are the following exceptions for data which require signing special licenses to access:

1. MedNLI requires PhysioNET credentials, so must be applied for and downloaded separately, see [here](https://physionet.org/content/mednli/1.0.0/) for details. Once you have obtained access to the dataset,
download the mednli dataset files from PhysioNet to the directory `<project_root>/data/raw_data/mednli_raw_data`.
2. I2B2-2010 data requires signing the n2nb2 license. See [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) for details. Once you have obtained access, 
download the I2B2-2010 files `concept_assertion_relation_training_data.tar.gz`, `test_data.tar.gz` and  `reference_standard_for_test_data.tar.gz` and unzip them in the `<project_root>/data/raw_data/i2b2-raw-data/i2b2-2010` directory.
3. I2B2-2012 data requires signing the n2nb2 license. See [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) for details. Once you have obtained access, 
download the I2B2-2012 files `2012-07-15.original-annotation.release.tar.gz`, and  `2012-08-08.test-data.event-timex-groundtruth.tar.gz` and unzip them in the `<project_root>/data/raw_data/i2b2-raw-data/i2b2-2012` directory.
4. I2B2-2014 data requires signing the n2nb2 license. See [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) for details. Once you have obtained access, 
download the I2B2-2014 files `training-RiskFactors-Gold-Set1.tar.gz`, `training-RiskFactors-Gold-Set2.tar.gz` and  `testing-PHI-Gold-fixed.tar.gz` and unzip them in the `<project_root>/data/raw_data/i2b2-raw-data/i2b2-2014` directory.

### Preprocessing the Raw Task data:

The datasets then need to be preprocessed. The preprocessed data will be written to `<project root>/data/tasks`

The classification datasets can be preprocessed by running:
 ```
 bash preprocessing/preprocess_all_classification_datasets.sh
 ```

The NER datasets must be preprocessed for each model's tokenizer you want train, only to ensure that sequences are not longer than a maximum length (512 tokens):
```
# Preprocess for roberta-large's tokenizer:
bash preprocessing/preprocess_all_sequence_labelling_datasets.sh roberta-large
```

### Running Classification Experiments

Once data have been preprocessed, Classification experiments can be run using `biolm.run_classification`. An example command is below:
```bash
TASK="ChemProt"
DATADIR="data/tasks/ChemProt"
MODEL=roberta-large
MODEL_TYPE=roberta
python -m biolm.run_classification \
    --task_name ${TASK}\
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --model_name_or_path ${MODEL}\
    --tokenizer_name ${MODEL}\
    --output_dir path/to/save/model/to \
    --fp16\
    --max_seq_length 512\
    --num_train_epochs 10\
    --per_gpu_train_batch_size 8\
    --per_gpu_eval_batch_size 8\
    --save_steps 200\
    --seed 10\
    --gradient_accumulation_steps 2\
    --learning_rate 2e-5\
    --do_train\
    --do_eval\
    --warmup_steps 0\
    --overwrite_output_dir\
    --overwrite_cache
```
To see more options, print help options: `python -m biolm.run_classification -h`

To test a model on test data, use the `--do_test` option on a trained checkpoint:

```bash
TASK="ChemProt"
DATADIR="data/tasks/ChemProt"
MODEL_TYPE="roberta"
CHECKPOINT_DIR=ckpts/${TASK}
python -m biolm.run_classification \
    --task_name ${TASK}\
    --data_dir ${DATADIR}\
    --model_type ${MODEL_TYPE}\
    --model_name_or_path ${CHECKPOINT_DIR}\
    --tokenizer_name ${CHECKPOINT_DIR}\
    --output_dir ${CHECKPOINT_DIR} \
    --fp16\
    --max_seq_length 512\
    --per_gpu_eval_batch_size 8\
    --overwrite_output_dir\
    --overwrite_cache \
    --do_test
```
This will write test predictions to the model directory under the file `test_predictions.tsv` and the dataset-appropriate test set scores under `test_results.txt`

Note: GAD and EuADR are split 10 ways for cross-validation. Each fold can be run by appending the fold number, e.g. `GAD3` 

### Running Sequence Labelling Experiments
Once data have been preprocessed, Sequence Labelling experiments can be run using `biolm.run_sequence_labelling`. An example command is below:

```bash
TASK="BC5CDR-chem"
DATADIR="data/tasks/BC5CDR-chem.model=roberta-large.maxlen=512"
MODEL=roberta-large
MODEL_TYPE=roberta
python -m biolm.run_sequence_labelling \
    --data_dir ${DATADIR} \
    --model_type ${MODEL_TYPE} \
    --labels ${DATADIR}/labels.txt \
    --model_name_or_path ${MODEL} \
    --output_dir path/to/save/model/to \
    --max_seq_length  512 \
    --num_train_epochs 20 \
    --per_gpu_train_batch_size 8 \
    --save_steps 500 \
    --seed 10 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --eval_all_checkpoints
```
To see more options, print help options: `python -m biolm.run_sequence_labelling -h`

To test a model on test data, use the `--do_predict` option on a trained checkpoint:
```bash
TASK="BC5CDR-chem"
DATADIR="data/tasks/BC5CDR-chem.model=roberta-large.maxlen=512"
CHECKPOINT_DIR=ckpts/${TASK}/checkpoint-1500
MODEL_TYPE=roberta
python -m biolm.run_sequence_labelling \
    --data_dir ${DATADIR} \
    --model_type ${MODEL_TYPE} \
    --labels ${DATADIR}/labels.txt \
    --model_name_or_path ${CHECKPOINT_DIR} \
    --output_dir ${CHECKPOINT_DIR} \
    --max_seq_length  512 \
    --per_gpu_eval_batch_size 8 \
    --seed 10 \
    --do_predict
```
This will write test predictions to the model directory under the file `test_predictions.txt` and scores under `test_results.txt`.
The official CoNLL evaluation script can then be calculated on the `test_predictions.txt` file `python2 conlleval.py path/to/test_predictions.txt`.

## Citation

To cite this work, please use the following bibtex:
```
@inproceedings{lewis-etal-2020-pretrained,
    title = "Pretrained Language Models for Biomedical and Clinical Tasks: Understanding and Extending the State-of-the-Art",
    author = "Lewis, Patrick  and
      Ott, Myle  and
      Du, Jingfei  and
      Stoyanov, Veselin",
    booktitle = "Proceedings of the 3rd Clinical Natural Language Processing Workshop",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.clinicalnlp-1.17",
    pages = "146--157",
}
```

## LICENSE

The code, models and data in this repository is licenced according the [LICENSE](./LICENSE) file, with the following exceptions:
* `conlleval.py` is licensed according to the [MIT License](https://opensource.org/licenses/MIT). 
* The BLUE Benchmark, an external dependency cloned and used for some preprocessing tasks, is licenced according to the [licence](https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/LICENSE.txt) it is distributed with.
* `preprocess_i2b2_2010_ner.py`, `preprocess_i2b2_2012_ner.py`, `preprocess_i2b2_2014_ner.py` are adapted from preprocessing jupyter notebooks in  [ClinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT), and are licensed according to the [MIT License](https://opensource.org/licenses/MIT). 
* `run_classification.py` and `utils_classification.py` are licensed according the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0)
* `run_sequence_labelling.py` and `utils_sequence_labelling.py` are licensed according the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0)