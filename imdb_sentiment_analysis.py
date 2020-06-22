__author__ = 'Akshit Agarwal'
__email__ = 'akshit@email.arizona.edu'
__date__ = '2020-06-22'
__dataset__ = 'http://ai.stanford.edu/~amaas/data/sentiment/'
__connect__ = 'https://www.linkedin.com/in/akshit-agarwal93/'

import os
import re

from keras import Sequential
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from pandas import DataFrame, concat
from sklearn.model_selection import train_test_split


def cleanhtml(raw_html):
    """Removes irrelevant html tags from the review string using regular expressions (re)
    :param raw_html: review wih html string
    :return: clean review
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def get_reviews_data(folder):
    """This function loops through each txt files of positive and negative reviews, converts it to a pandas Datframe
     with columns 'Review' and 'Sentiment, and shuffles the dataframe
    :param folder: folder with txt files
    :return: pandas DataFrame
    """

    pos_df = DataFrame()
    neg_df = DataFrame()
    for sentiment in ['pos', 'neg']:
        print(f'''Fetching {sentiment} reviews...''')
        fpath = os.path.join(folder, sentiment)
        directory = os.fsencode(fpath)
        train_list = []
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                fpath = os.path.join(folder, sentiment, filename)
                with open(fpath, 'r', encoding='utf8') as readfile:
                    str_review = readfile.read().replace('\n', '')
                    str_review = cleanhtml(str_review)
                    train_list.append(str_review)
        if sentiment == 'pos':
            pos_df = DataFrame(train_list)
            pos_df['sentiment'] = 1
        else:
            neg_df = DataFrame(train_list)
            neg_df['sentiment'] = 0
    data = concat([pos_df, neg_df])
    data.columns = ['Review', 'Sentiment']
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffles the dataset
    print(data.head())
    return data


def preprocess_text_data(data, max_features, max_len):
    """This function prepares the dataset for applying ML Algorithm by converting the text data into numbers
    :param data:
    :param max_features:the maximum number of words to keep, based on word frequency. Only the most common words are kept
    :param max_len: maximum length of sequemces generated in text_to_sequences
    :return: numpy array of sequence of numbers (converted from textual data)
    """
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    reviews_list = data['Review']
    tokenizer.fit_on_texts(reviews_list)
    print(tokenizer.word_index)
    """This is the vocabulary size of the dataset. It is the number of unique words in our dataset. This should be the
    input of the embedding layer (input_dim) in the network."""
    X = tokenizer.texts_to_sequences(reviews_list)
    X = pad_sequences(X, maxlen=max_len)
    Y = data.loc[:, ['Sentiment']].values
    return X, Y


def create_baseline_model(X, input_dim, output_dim):
    """This function creates a LSTM Network with an Embedding Layer
    :param X: train set
    :param input_dim: Input to the Embedding Layer. This should be greater than or equal to the vocabulary size.
    :param output_dim: Dimension of the dense embedding.
    :return:Sequential Model
    """
    model = Sequential()
    # print(tokenizer.word_index)
    model.add(Embedding(input_dim, output_dim, input_length=X.shape[1]))
    model.add(Dropout(0.2))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, go_backwards=False))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.9))
    model.add(Dense(1, activation='sigmoid'))  # For binary classification
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    print(model.summary())
    return model


def split_and_fit_train_set(X, Y, model, test_size, batch_size, epochs):
    """This function splits dataset into training and validation data and fits the model with a specified batch_size
    and epochs
    :param X: Input data
    :param Y: Inpot Labels [1=positive, 0=negative]
    :param model: Sequential model
    :param test_size: ratio of validation/test data to the entire dataset
    :param batch_size: number of samples processed before the model is updated
    :param epochs: no of times input is fed forward and backward to the network
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(X_test, Y_test))
    print('model is fit...')


def evaluate_on_test_set(model, data, max_features, max_len, batch_size):
    """This function preporcesses the test data, evaluates the model performance and prints the
        score and accuracy of the model
     :returns score, accuracy: Model score and model accuracy on the validation set"""

    tokenizer = Tokenizer(num_words=max_features, split=' ')
    reviews_list = data['Review']
    tokenizer.fit_on_texts(reviews_list)
    X = tokenizer.texts_to_sequences(reviews_list)
    X = pad_sequences(X, maxlen=max_len)
    Y = data.loc[:, ['Sentiment']].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    score, accuracy = model.evaluate(X_train, Y_train, batch_size=batch_size, verbose=2)
    return score, accuracy


def main():
    # Location of datasets
    train_path = r"C:\Users\akshitagarwal\Desktop\Keras\datasets\IMDB Reviews Dataset\train"
    val_path = r"C:\Users\akshitagarwal\Desktop\Keras\datasets\IMDB Reviews Dataset\val"

    train_data = get_reviews_data(train_path)
    X, Y = preprocess_text_data(train_data, max_features=1200, max_len=50)
    model = create_baseline_model(X, 90000, 16)
    split_and_fit_train_set(X, Y, model, 0.2, batch_size=32, epochs=50)
    val_data = get_reviews_data(val_path)
    evaluate_on_test_set(model, val_data, max_features=1200, max_len=50, batch_size=32)
    print('program execution finished')


if __name__ == '__main__':
    main()
