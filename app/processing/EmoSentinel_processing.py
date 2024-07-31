import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,GRU,RNN,SimpleRNN
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import datetime
import joblib
import re
from nltk.corpus import stopwords
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


mental_health_df = pd.read_csv("D:/Machine_Learning/Datasets/MENTAL_Health/mental_health.csv")
mental_health_df = mental_health_df.dropna()
encoder = LabelEncoder()
mental_health_df['encoded_label'] = encoder.fit_transform(mental_health_df['status'])

print("TF VERSION",tf.__version__)
import tensorflow.keras
print("keras version",tensorflow.keras.__version__)

def plot_metrics(history):
    metrics = ['loss', 'accuracy']#, 'precision', 'recall']
    x = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    for metric in metrics:
        plt.plot(history.history[metric], label=f'train_{metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f'Train and Validation {metric.capitalize()} Over Epochs')
        path = os.path.join(os.path.dirname(__file__), 'model')
        plt.savefig(f'{path}{metric}_{x}_plot.png')
        plt.clf()
def prepare_data():
    # Tokenize the text
    texts = mental_health_df['statement'].values
    labels = mental_health_df['encoded_label'].values
    max_features = 10000  # Number of words to consider as features
    maxlen = 500  # Cut texts after this number of words
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=maxlen)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test
def emo_sentinel_train(model_details):
    x_train, x_test, y_train, y_test = prepare_data()
    # To continue training the next day, load the model and weights
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_full_updated.h5')
    print(model_path)
    model = tf.keras.models.load_model(model_path)
    checkpoint_filepath =  os.path.join(os.path.dirname(__file__),  'model', 'model_checkpoint.h5')
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath,
                                          save_weights_only=True,
                                          save_best_only=True,
                                          monitor='val_loss',
                                          mode='min')
    # Optionally, load the best weights if you used the checkpoint
    model.load_weights(checkpoint_filepath)

    # Continue training for another 5 epochs
    history = model.fit(x_train, y_train,
                        batch_size=model_details['batch_size'],
                        epochs=model_details['epochs'],
                        validation_data=(x_test, y_test),
                        callbacks=[checkpoint_callback])

    # Save the updated model
    model.save(model_path)
    #plot_metrics(history)
    res_metrics = []
    metrics = ['loss', 'accuracy']
    for metric in metrics:
        res_metrics.append(history.history[metric])

    return res_metrics


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    # text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


def emo_sentinel_base_detect(text):
    text_clean = clean_text(text)
    model_path_svm = os.path.join(os.path.dirname(__file__), 'model', 'MentalHealthFlag_svm.pkl')
    svm = joblib.load(model_path_svm)
    result = svm.predict([text_clean])
    print("SVM RES::" ,result[0])
    booleanRes = False
    if result[0] == 1:
        booleanRes =  True
    return booleanRes#result[0]
