# Rommendation_model_demo

### Introduction
This demo aims to design an app recommendation model. While users looking through a product, we obtain the title of this product, then recommend 20 related products that have related titles. This demo contains two parts: Chinese title demo and English title demo. Each demo has various solutions to achieve the goal.

## ch

### Prerequisites
* python==3.7
* numpy==1.21.6
* GPU == RTX2070
* torch==1.11.0+cu113
* torchvision==0.12.0+cu113
* torchvision==0.6.1+cu92
* jieba == 0.42.1
* sklearn == 1.0.2
* gensim == 4.2.0
* pandas == 1.3.5
* scipy == 1.7.3

### Preparation
* Download Pretrained [Chinese FastText model] (https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.zh.zip)

### dataloder
$python dataloader.py

### data preprocessing
$python preprocessing_ch.py

### TF-IDF model
$python TF-IDF.py

### Word2Vec model
$python convert_ch.py
$python word2vec.py

### CountVectorizer
$python count_vectorizer.py


## en
