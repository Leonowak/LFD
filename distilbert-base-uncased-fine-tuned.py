#!/usr/bin/env python

#Title: Fine-tuned Distilbert-base-uncased model for Multi-class classification tasks

#Purpose: This script is designed to run a fine-tuned version of the pre-trained Distilbert-base-uncased model on a corpus including labeled reviews.

#Functionality: It reads a txt file, transforms string labels into one-hot encodings, applies the Distilbert-base-uncased model, tokenizes the data, compiles a model and gives predicition accuracy on the respective input data set.

#Input: Three Txt files containing 5000 reviews.

#Output: Confusion Matrix and Classification Report for prediciting accuracy on multi-class labeling

#Key Features: Utilizes Keras and Transfomers for data manipulation and Scikit-Learn for preprocessing tasks.

#Use Cases: Ideal for data scientists and analysts working on machine learning projects who need to prepare their data for model training.

#Dependencies:** Requires Python, Transformers, Tensorflow, Keras and Scikit-Learn.

#Usage: Run the script with the path to the input txt file as a command-line argument.

#Author: Group 3

#Date: October 15, 2023

import random as python_random
import argparse
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.initializers import Constant
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import TextVectorization
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

    # Predict on the development set
    Y_dev_pred = model.predict(tokens_dev)["logits"]

    # Decode one-hot encoded labels
    Y_dev_pred_labels = encoder.inverse_transform(Y_dev_pred)

    # Calculate the confusion matrix and classification report
    confusion = confusion_matrix(Y_dev, Y_dev_pred_labels)
    report = classification_report(Y_dev, Y_dev_pred_labels, target_names=encoder.classes_)

    # Print the confusion matrix and classification report
    print("Confusion Matrix:\n", confusion)
    print("\nClassification Report:\n", report)

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
