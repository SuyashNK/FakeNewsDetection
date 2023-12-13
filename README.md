# FakeNewsDetection



## Fake News Detection: Data Science Project

Welcome to the `FakeNews.ipynb` Colaboratory notebook, a comprehensive project aimed at detecting fake news using Data Science techniques. In this notebook, we leverage machine learning, specifically a combination of Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs), to create a robust model for distinguishing between real and fake news.

### Importing Packages

The project begins by importing essential libraries, including TensorFlow and Keras, for building and training the machine learning model. Additional packages such as NumPy, Pandas, and Scikit-Learn are utilized for data manipulation and preprocessing.

```python
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import json
import csv
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import pprint
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

tf.disable_eager_execution()
data = pd.read_csv("/content/drive/MyDrive/data/news.csv")
data.head()
```

### Data Preprocessing

The notebook then focuses on preparing the dataset for training. This involves dropping unnecessary columns, encoding labels, and tokenizing the textual data for input into the model.

```python
data = data.drop(["Unnamed: 0"], axis=1)
le = preprocessing.LabelEncoder()
le.fit(data['label'])
data['label'] = le.transform(data['label'])

# Tokenization
# ... (code snippet for tokenization)
```

### Model Architecture

The heart of the project lies in creating a robust model architecture. The following layers form the basis of the model:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size1+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```

### Model Training

The model is trained over 50 epochs with the training dataset, and the training progress is visualized with accuracy metrics.

```python
num_epochs = 50
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
```

### Model Evaluation and Prediction

To assess the model's performance, a sample text is provided for validation, demonstrating the practical application of the trained model.

```python
# Providing the sample text and validating
X = "Suyash is going to be the next big thing"

# Detection
sequences = tokenizer1.texts_to_sequences([X])[0]
sequences = pad_sequences([sequences], maxlen=54, padding=padding_type, truncating=trunc_type)

if model.predict(sequences, verbose=0)[0][0] >= 0.5:
    print("This news is REAL")
else:
    print("This news is FAKE")
```

Feel free to explore and adapt this notebook for your own fake news detection projects. If you have any questions or need assistance, please don't hesitate to reach out.

---

*Note: Adjust paths and parameters based on your dataset and requirements.*
