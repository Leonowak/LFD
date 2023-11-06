# How to Install Dependencies
This README provides instructions on how to install the necessary dependencies for running various models for text classification. The models mentioned include Bag of Words (BoW), Long Short-Term Memory (LSTM), and Pre-trained Language Model (PLM).

To run the models, the following dependencies are needed:
## Bag of Words:

    scikit-learn
    nltk

and can be installed using pip:

    pip install scikit-learn nltk

## LSTM

    tensorflow, numpy, scikit-learn

and can be installed using pip:

    pip install tensorflow numpy scikit-learn

## Pretrained language models

    transformers

and can be installed using pip:

    pip install tensorflow, numpy, transformers, scikit-learn

# How to Train the Models on the Data
## Random Forest (RF) with Features on OffensEval dataset:
    python bag-of-words.py -tf train.json -df dev.json -t --algorithm rf --n_estimators 300 --min_samples_split 2 -ngram 7 -char 1 -pos
## Random Forest (RF) with Features on HateXplain dataset:
    python bag-of-words.py -tf train_offenseval_hatexplain.json -df dev_offenseval_hatexplain.json -t --algorithm rf --n_estimators 300 --min_samples_split 2 -ngram 7 -char 1 -pos
## SVM with Features on OffensEval dataset:
    python bag-of-words.py -tf train.json -df dev.json -t --algorithm svm --svm_kernel linear --svm_c 2.0 -ngram 7
## SVM with Features on HateXplain dataset:
    python bag-of-words.py -tf train_offenseval_hatexplain.json -df dev_offenseval_hatexplain.json -t --algorithm svm --svm_kernel linear --svm_c 2.0 -ngram 7
## LSTM on OffensEval dataset:
    python3 LSTM.py -tf train_offens.json -df dev_offens.json -t test_offens.json -e glove.json
## LSTM on HateXplain dataset:
    python3 LSTM.py -tf train_offenseval_hatexplain.json -df dev_offenseval_hatexplain.json -t test_offenseval_hatexplain.json -e glove.json
## PLM on OffensEval dataset:
    python3 LFDfin_distilbert_base_uncased_emotion.py -tr train_offens.json -d dev_offens.json -lr 0.0001 -bs 128 -e 1 -sl 100
## PLM on HateXplain dataset:
    python3 LFDfin_distilbert_EX.py -trE train_offenseval_hatexplain.json -lr 0.0001 -bs 128 -e 1 -sl 100
