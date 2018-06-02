# Topic-Modelling

In this project, we implement two models for topic modeling- Latent Dirichlet Allocation (LDA) and Deep Belief Net (DBN). We use the bag-of-words and term-frequency models to produce latent representation of text data. The three files included are data_preprocessing.py, lda.py and dbn.py. 

### Requirements

  - numpy
  - scikit-learn
  - tensorflow
  - nltk
  - pandas
  - scipy
  - python 2 (for lda.py)
  - python 3 (for dbn.py)

### Dataset

The two datasets that are used in our models can be found at [20Newsgroups] and [BBCSport].

### How to run

 - data_preprocessing.py: loads raw text file of the dataset; performs basic preprocessing steps (tokenization, stemming, lemmatization) and term frequency count; the output is a .mat file with term frequencies and corresponding labels. 
 - lda.py (requires python 2): fits an lda model on the input data from sklearn library; random forest classifier is used on lda output to generate accuracy results and then visualize.
 - dbn.py (requires python 3): run this file to generate output accuracy using DBN
Note: preprocessed data files bbc_train_data.mat and 20NG_train_data.mat are included with this project.


[20Newsgroups]: <http://qwone.com/~jason/20Newsgroups/>
[BBCSport]: <http://mlg.ucd.ie/datasets/bbc.html>
