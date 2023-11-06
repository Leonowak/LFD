#!/usr/bin/env python

import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import random as python_random

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_file", default='train_offens.json', type=str,
                        help="Input file to learn from (default train_offens.json)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev_offens.json',
                        help="Separate dev set to read in (default dev_offens.json)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    return parser.parse_args()

def read_corpus(corpus_file):
    """Function to read and process the data from a JSON file"""
    texts = []
    labels = []
    with open(corpus_file, encoding='utf-8-sig') as in_file:
        data = json.load(in_file)
        for entry in data:
            text = entry.get("text")
            label = entry.get("label")
            if text is not None and label in ["NOT", "OFF"]:
                texts.append(text)
                labels.append(1 if label == "OFF" else 0)
    return texts, labels

def main():
    args = create_arg_parser()
    
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    lm = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=1)

    tokens_train = tokenizer(X_train, padding=True, max_length=100, truncation=True, return_tensors="tf")
    tokens_dev = tokenizer(X_dev, padding=True, max_length=100, truncation=True, return_tensors="tf")

    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])

    model.fit(
        {'input_ids': tokens_train['input_ids'], 'attention_mask': tokens_train['attention_mask']},
        np.array(Y_train),
        verbose=1,
        epochs=1,
        batch_size=16,
        validation_data=(
            {'input_ids': tokens_dev['input_ids'], 'attention_mask': tokens_dev['attention_mask']},
            np.array(Y_dev)
        )
    )

    # Predict on the development set
    Y_dev_pred = model.predict({'input_ids': tokens_dev['input_ids'], 'attention_mask': tokens_dev['attention_mask']})["logits"]

    # Convert logits to class labels for binary classification
    Y_dev_pred_labels = (Y_dev_pred > 0).astype(int)

    # Calculate the confusion matrix and classification report for binary classification
    confusion = confusion_matrix(Y_dev, Y_dev_pred_labels)
    report = classification_report(Y_dev, Y_dev_pred_labels)

    # Print the confusion matrix and classification report
    print("Confusion Matrix:\n", confusion)
    print("\nClassification Report:\n", report)

    # Do predictions on specified test set
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file)
        tokens_test = tokenizer(X_test, padding=True, max_length=100, truncation=True, return_tensors="tf")
        test_loss, test_accuracy = model.evaluate(
            {'input_ids': tokens_test['input_ids'], 'attention_mask': tokens_test['attention_mask']},
            np.array(Y_test),
            verbose=1
        )
        Y_pred = model.predict({'input_ids': tokens_test['input_ids'], 'attention_mask': tokens_test['attention_mask']})["logits"]
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
        print("Test Predictions Logits", Y_pred)

if __name__ == '__main__':
    main()
