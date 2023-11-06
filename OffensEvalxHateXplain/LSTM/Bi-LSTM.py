import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.initializers import Constant
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

def create_arg_parser():
    """Creates an argument parser for parsing command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--train_file", default='train.json', type=str,
                        help="Train file to learn from (default train.json)")
    parser.add_argument("-df", "--dev_file", default='dev.json', type=str,
                        help="Dev file to evaluate on (default dev.json)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="Test file to predict on")
    parser.add_argument("-e", "--embeddings", default='glove.json', type=str,
                        help="Embedding file (default glove_reviews.json)")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimizer")
    parser.add_argument("--loss_function", type=str, default='binary_crossentropy',
                        help="Loss function for model training")
    parser.add_argument("--verbose", type=int, default=2,
                        help="Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default= 100,
                        help="Number of training epochs")
    args = parser.parse_args()
    return args
    
class AttentionLayer(Layer):
    """Custom Keras layer for implementing an attention mechanism"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """Builds the layer by initializing weights"""
        last_dim = input_shape[-1]
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to AttentionLayer should be defined. Found `None`.')
        self.W = self.add_weight(name="att_weight", shape=(last_dim, 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(last_dim, 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        """Defines the logic for the layer's forward pass"""
        e = K.tanh(K.dot(x, self.W) + self.b[:K.shape(x)[1], :])
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer"""
        return (input_shape[0], input_shape[-1])

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

def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    try:
        embeddings = json.load(open(embeddings_file, 'r'))
        return {word: np.array(embeddings[word]) for word in embeddings}
    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
        exit(1)  # Exit the script with an error code
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON in '{embeddings_file}'. Check file format.")
        exit(1)  # Exit the script with an error code

def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix

def create_model(Y_train, emb_matrix, args):
    """Create and compile the neural network model"""
    learning_rate = args.learning_rate
    loss_function = args.loss_function  # Using user-specified loss function
    optim = SGD(learning_rate=learning_rate)
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = 1  

    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
    model.add(AttentionLayer())
    model.add(Dense(units=num_labels, activation="sigmoid"))  # Sigmoid activation for binary classification
    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_dev, Y_dev, args):
    """Train the neural network model and evaluate it on the dev set"""
    verbose = args.verbose
    batch_size = args.batch_size
    epochs = args.epochs
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size, validation_data=(X_dev, Y_dev))
    test_set_predict(model, X_dev, Y_dev, "dev", args)
    return model

def test_set_predict(model, X_test, Y_test, ident, args):
    """Make predictions on the test set and print classification metrics"""
    Y_pred = model.predict(X_test)
    Y_pred = (Y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    Y_test = Y_test.astype(int)
    print(f'Classification Report on own {ident} set:\n{classification_report(Y_test, Y_pred)}')
    print(f'Confusion Matrix on own {ident} set:\n{confusion_matrix(Y_test, Y_pred)}')

def main():
    """Main function that orchestrates the data processing and model training"""
    args = create_arg_parser()

    # Load the data from the JSON files
    train_data, train_labels = read_corpus(args.train_file)
    dev_data, dev_labels = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    # Vectorize the text data
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    text_ds = tf.data.Dataset.from_tensor_slices(train_data + dev_data)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Encode labels
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(train_labels)
    Y_dev_bin = encoder.transform(dev_labels)

    # Create the model
    model = create_model(Y_train_bin, emb_matrix, args)

    # Vectorize the data for training and evaluation
    X_train_vect = vectorizer(np.array(train_data)).numpy()
    X_dev_vect = vectorizer(np.array(dev_data)).numpy()

    # Train the model
    model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, args)

    # Evaluate on the test set if provided
    if args.test_file:
        test_data, test_labels = read_corpus(args.test_file)
        Y_test_bin = encoder.transform(test_labels)
        X_test_vect = vectorizer(np.array(test_data)).numpy()
        test_set_predict(model, X_test_vect, Y_test_bin, "test", args)

if __name__ == '__main__':
    main()