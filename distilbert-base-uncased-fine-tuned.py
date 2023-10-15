#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import random as python_random
import json
import argparse
import logging
import numpy as np
from keras.models import Sequential
from keras.initializers import Constant
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    args = parser.parse_args()
    return args

def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip()
            documents.append(" ".join(tokens.split()[3:]).strip())
            # 6-class problem: books, camera, dvd, health, music, software
            labels.append(tokens.split()[0])
    return documents, labels

def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()
    
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(Y_dev)


    lm = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=6)

    tokens_train = tokenizer(X_train, padding=True, max_length=200,truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=200,truncation=True, return_tensors="np").data

    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=0.0001)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=3, batch_size=128, validation_data=(tokens_dev, Y_dev_bin))

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        X_test, Y_test = read_corpus(args.test_file)
        Y_test = encoder.fit_transform(Y_test)
        tokens_test = tokenizer(X_test, padding=True, max_length=200, truncation=True, return_tensors="np")
        test_loss, test_accuracy = model.evaluate(tokens_test, Y_test_bin, verbose=1)
        Y_pred = model.predict(tokens_dev)["logits"]
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
        print("Test Accuracy Logits", Y_pred)
if __name__ == '__main__':
    main()
