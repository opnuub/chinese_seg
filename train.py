import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from gensim.models import KeyedVectors


def precision(y_true, y_pred):
    """Precision metric: computes a batch-wise average of precision."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def build_model(char_size, dropout, lr, embedding_matrix, max_len, word_size, num_classes):
    model = Sequential()
    model.add(Embedding(char_size, word_size, input_length=max_len, weights=[embedding_matrix], mask_zero=True))
    model.add(Bidirectional(LSTM(char_size, dropout=dropout, recurrent_dropout=dropout, return_sequences=False), merge_mode='sum'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    optimizer = SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[precision, 'accuracy'])
    return model


def main(args):

    df = pd.read_table(args.train_input, header=None)
    label = pd.read_table(args.train_label, header=None)
    df['label'] = label[0]
    df.columns = ['character', 'label']
    df = df.sample(n=args.train_sample_size, random_state=42)  # sampling if needed

    df_val = pd.read_table(args.val_input, header=None)
    label_val = pd.read_table(args.val_label, header=None)
    df_val['val_label'] = label_val[0]
    df_val.columns = ['val_character', 'val_label']
    df_val = df_val.sample(n=args.val_sample_size, random_state=42)

    vocabulary_size = args.vocabulary_size
    max_len = args.max_sequence_length
    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(df['character'])
    sequences = tokenizer.texts_to_sequences(df['character'])
    data = pad_sequences(sequences, maxlen=max_len)

    tokenizer_val = Tokenizer(num_words=vocabulary_size)
    tokenizer_val.fit_on_texts(df_val['val_character'])
    sequences_val = tokenizer_val.texts_to_sequences(df_val['val_character'])
    val_data = pad_sequences(sequences_val, maxlen=max_len)

    train_labels = LabelEncoder().fit_transform(df.label)
    val_labels = LabelEncoder().fit_transform(df_val['val_label'])

    train_labels_counts = Counter(df.label)
    max_count = max(train_labels_counts.values())
    oversampled_data = []
    oversampled_labels = []

    for label, count in train_labels_counts.items():
        class_data = data[df['label'] == label]
        class_labels = df['label'][df['label'] == label]
        oversampled_data.extend(class_data)
        oversampled_labels.extend(class_labels)
        if count < max_count:
            n_to_oversample = max_count - count
            indices_to_duplicate = random.choices(range(len(class_data)), k=n_to_oversample)
            oversampled_data.extend(class_data[indices_to_duplicate])
            oversampled_labels.extend(class_labels.iloc[indices_to_duplicate])

    train_data_resampled = np.array(oversampled_data)
    train_labels_resampled = np.array(oversampled_labels)

    embedding_model = KeyedVectors.load_word2vec_format(args.embedding_file)
    embedding_dim = len(embedding_model[next(iter(embedding_model.vocab))])

    embedding_matrix = np.random.rand(vocabulary_size, embedding_dim)
    word_index = tokenizer.word_index
    for word, i in word_index.items():
        if i < vocabulary_size:
            try:
                embedding_vector = embedding_model.get_vector(word)
                embedding_matrix[i] = embedding_vector
            except Exception:
                pass

    num_classes = args.num_classes
    y = to_categorical(train_labels_resampled, num_classes)
    y_val_test = to_categorical(val_labels, num_classes)

    class_weights_ = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)

    os.makedirs(args.output_dir, exist_ok=True)
    weight_filepath = os.path.join(args.output_dir, "weights-improvement-{epoch:02d}-{precision:.2f}.hdf5")
    checkpoint = ModelCheckpoint(weight_filepath, monitor='precision', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger(os.path.join(args.output_dir, 'model.log'), separator=',', append=False)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    callbacks_list = [checkpoint, csv_logger, es]

    char_size = args.char_size
    dropout = args.dropout
    lr = args.learning_rate
    optimizer_name = args.optimizer
    word_size = args.word_size
    model = build_model(char_size, dropout, lr, optimizer_name, embedding_matrix, max_len, word_size, num_classes)

    history = model.fit(
        train_data_resampled,
        y,
        validation_data=(val_data, y_val_test),
        epochs=args.epochs,
        class_weight=class_weights_,
        batch_size=args.batch_size,
        callbacks=callbacks_list,
        verbose=1
    )

    model_json = model.to_json()
    with open(os.path.join(args.output_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(args.output_dir, 'model_weights.h5'), overwrite=True)

    plt.figure()
    plt.plot(history.history.get('accuracy', history.history.get('acc')))
    plt.plot(history.history.get('val_accuracy', history.history.get('val_acc')))
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['precision'])
    plt.plot(history.history.get('val_precision', []))
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    y_pred = np.argmax(model.predict(validation_data_resampled[:args.num_predict]), axis=-1)
    y_true = validation_labels_resampled[:args.num_predict]
    report = classification_report(y_true, y_pred, target_names=[str(i) for i in range(num_classes)])
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a text classification model using SMOTE and pre-trained embeddings.")
    parser.add_argument('--train-input', type=str, required=True,
                        help="Path to the training input text file.")
    parser.add_argument('--train-label', type=str, required=True,
                        help="Path to the training label file.")
    parser.add_argument('--val-input', type=str, required=True,
                        help="Path to the validation input text file.")
    parser.add_argument('--val-label', type=str, required=True,
                        help="Path to the validation label file.")
    parser.add_argument('--embedding-file', type=str, required=True,
                        help="Path to the embedding model file in word2vec format.")
    parser.add_argument('--output-dir', type=str, default="resources",
                        help="Directory for saving the model, weights, and logs.")
    parser.add_argument('--train-sample-size', type=int, default=50000,
                        help="Sample size for training data (for speed; use full data by setting a high value).")
    parser.add_argument('--val-sample-size', type=int, default=20000,
                        help="Sample size for validation data (for speed; use full data by setting a high value).")
    parser.add_argument('--vocabulary-size', type=int, default=128,
                        help="Vocabulary size for tokenizer.")
    parser.add_argument('--max-sequence-length', type=int, default=50,
                        help="Maximum sequence length for padding.")
    parser.add_argument('--num-classes', type=int, default=4,
                        help="Number of target classes.")
    parser.add_argument('--char-size', type=int, default=256,
                        help="Number of units in the embedding and LSTM layers.")
    parser.add_argument('--dropout', type=float, default=0.2,
                        help="Dropout rate for the model.")
    parser.add_argument('--learning-rate', type=float, default=0.04,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="Optimizer type (e.g., 'sgd').")
    parser.add_argument('--word-size', type=int, default=100,
                        help="Dimension of the output space for the Embedding layer.")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of epochs for training.")
    parser.add_argument('--batch-size', type=int, default=10,
                        help="Batch size for training.")
    parser.add_argument('--num-predict', type=int, default=500,
                        help="Number of samples to use in the prediction for classification report.")

    args = parser.parse_args()
    main(args)
