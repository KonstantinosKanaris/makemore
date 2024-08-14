# makemore

## Table of Contents
* [Overview](#Overview)
* [Key Features](#Key--Features)
* [Data](#Data)
* [Development](#Development)
* [Execute](#Execute)


## Overview 🔍

An autoregressive character-level language model that generates more of the same type of 
content, such as names. Supports wide choice of models from bigrams all the way to a tiny GPT
language model.

The project is based on Andrej Karpathy's very insightful and constructive [lectures](https://www.youtube.com/@AndrejKarpathy).

Current model architectures are based on a few key papers:
- Bigram (a simple lookup table where one character predicts the next)
- BoW (predicts the next character based on the average of the preceding characters)
- MLP, following [Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

## Key Features 🔑

* **Custom TQDM Bar**: ```Training: [11/50] | [155/156]:   22%|███▍       , loss=2.44 [01:40<21:29]```
* **Checkpointing**: Option to resume training process from the last saved checkpoint
* **EarlyStopping**: The training process stops when the model's performance no longer improves on the validation set
* **LRScheduling**: The learning rate is reduced during training when the model's performance no longer improves on the validation set
* **Experiment Tracking**: Uses MLflow for experiment tracking

## Data 📄

The input dataset used for the training process consists of the most common 32K names found at [ssa.gov](https://www.ssa.gov/oact/babynames/)
for the year 2018. It looks like:
```
harper
evelyn
abigail
emily
elizabeth
mila
ella
...
```

## Development 🐍
Clone the repository:
```bash
$ git clone https://github.com/KonstantinosKanaris/makemore.git
```

### Set up the environment

#### Create environment
Python 3.10 is required.

- Create the environment and install the dependencies:
```bash
$ pipenv --python 3.10
$ pipenv install --dev
```
- Enable the newly-created virtual environment, with:
```bash
$ pipenv shell
```

## Execute 🚀
All the commands should be executed from the project's root directory.

### Training
```bash
$ python -m makemore -i ./data/names.txt -o names
```
The default model is a super tiny transformer decoder with 200k parameters. For additional 
configurations, refer to the `parse_arguments` function in `__main__.py`. Samples will be generated 
and printed throughout the training process.

### Sampling
For manually sampling without training, use the `--sample_only` flag:
```bash
$ python -m makemore -i ./data/names.txt -o names --sample_only
```
This will load the best model so far and generate additional samples:
```
knobe
kymana
karil
kaifany
mariah
avonz
blaeya
leris
dranklee
bofwin
adala
jahub
abdan
aanellah
julisa
rosaleus
harija
jazellyn
orin
hawaul
brisanna
ivea
zhaida
keira
shady
yamidee
will
bearli
adri
manesia
```
The default settings for the Transformer achieved a loss of ~1.98, but a much lower loss 
can be achieved with hyperparameter tuning.

## Licence

MIT