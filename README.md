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

Code to run sequence classification and NER will be uploaded soon.

## Citation

To cite us, please use the following bibtex:
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

The code, models and data in this repository is licenced according the [LICENSE](./LICENSE) file.