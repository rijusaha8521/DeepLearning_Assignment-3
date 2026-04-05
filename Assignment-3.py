import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# Sample data
texts = ["I love this movie", "This is bad", "Amazing experience", "Worst film ever"]
labels = [1, 0, 1, 0]

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
X = pad_sequences(sequences, maxlen=5)
y = np.array(labels)

# RNN Model
rnn_model = Sequential([
    Embedding(5000, 32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X, y, epochs=5)

# LSTM Model
lstm_model = Sequential([
    Embedding(5000, 32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X, y, epochs=5)

print("Training completed for both RNN and LSTM models.")