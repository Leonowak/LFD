#!/usr/bin/env python

import json
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import random as python_random

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train_file", default='train_offens.json', type=str, help="Input file to learn from")
    parser.add_argument("-d", "--dev_file", type=str, default='dev_offens.json', help="Separate dev set to read in")
    parser.add_argument("-t", "--test_file", type=str, default='test_offens.json', help="If added, use trained model to predict on the test set")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("-sl", "--sequence_length", type=int, default=100, help="Sequence length for padding/truncation")
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
                # Convert labels to a one-hot encoded format
                labels.append([1, 0] if label == "OFF" else [0, 1])
    return texts, np.array(labels)

def main():
    args = create_arg_parser()
    
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)

    lm = "bhadresh-savani/distilbert-base-uncased-emotion"
    tokenizer = AutoTokenizer.from_pretrained(lm)

    config = DistilBertConfig.from_pretrained(lm)
    config.num_labels = 2

    model = TFDistilBertForSequenceClassification.from_pretrained(lm, config=config, ignore_mismatched_sizes=True)
    
    tokens_train = tokenizer(X_train, padding=True, max_length=args.sequence_length, truncation=True, return_tensors="tf")
    tokens_dev = tokenizer(X_dev, padding=True, max_length=args.sequence_length, truncation=True, return_tensors="tf")

    loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optim = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(
        loss=loss_function,
        optimizer=optim,
        metrics=['accuracy']
    )

    model.fit(
        {'input_ids': tokens_train['input_ids'], 'attention_mask': tokens_train['attention_mask']},
        Y_train,
        verbose=1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(
            {'input_ids': tokens_dev['input_ids'], 'attention_mask': tokens_dev['attention_mask']},
            Y_dev
        )
    )

    # Predict on the development set
    Y_dev_pred_logits = model.predict({'input_ids': tokens_dev['input_ids'], 'attention_mask': tokens_dev['attention_mask']})["logits"]
    # Convert logits to class labels for binary classification
    Y_dev_pred_labels = np.argmax(Y_dev_pred_logits, axis=1)

    # Calculate the confusion matrix and classification report for binary classification
    confusion = confusion_matrix(np.argmax(Y_dev, axis=1), Y_dev_pred_labels)
    report = classification_report(np.argmax(Y_dev, axis=1), Y_dev_pred_labels)

    # Print the confusion matrix and classification report
    print("Confusion Matrix:\n", confusion)
    print("\nClassification Report:\n", report)

    # Do predictions on specified test set
    if args.test_file:
        X_test, Y_test = read_corpus(args.test_file)
        tokens_test = tokenizer(X_test, padding=True, max_length=100, truncation=True, return_tensors="tf")
        test_loss, test_accuracy = model.evaluate(
            {'input_ids': tokens_test['input_ids'], 'attention_mask': tokens_test['attention_mask']},
            Y_test,
            verbose=1
        )
        Y_pred_logits = model.predict({'input_ids': tokens_test['input_ids'], 'attention_mask': tokens_test['attention_mask']})["logits"]
        Y_pred_labels = np.argmax(Y_pred_logits, axis=1)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

        # Calculate the confusion matrix and classification report for binary classification on the test set
        confusion_test = confusion_matrix(np.argmax(Y_test, axis=1), Y_pred_labels)
        report_test = classification_report(np.argmax(Y_test, axis=1), Y_pred_labels)

        # Print the confusion matrix and classification report for the test set
        print("Test Confusion Matrix:\n", confusion_test)
        print("\nTest Classification Report:\n", report_test)

if __name__ == "__main__":
    main()
