import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer


def preprocess_data():
  lemmatizer = WordNetLemmatizer()
  intents = json.loads(open('intents.json').read())['intents']
  
  words = []
  classes = []
  documents = []
  ignore_letters = ['?', '!', '.', ',']

  for intent in intents:
    for pattern in intent['patterns']:
      word_list = nltk.word_tokenize(pattern)
      words.extend(word_list)
      documents.append((word_list, intent['tag']))
      if intent['tag'] not in classes:
        classes.append(intent['tag'])

  words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
  words = sorted(set(words))
  classes = sorted(set(classes))

  pickle.dump(words, open('words.pkl', 'wb'))
  pickle.dump(classes, open('classes.pkl', 'wb'))
  return words, classes, documents

def create_training_data(words, classes, documents):
  training = []
  output_empty = [0] * len(classes)

  for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [WordNetLemmatizer().lemmatize(word.lower()) for word in word_patterns]
    for word in words:
      bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

  random.shuffle(training)
  training = np.array(training)

  train_x = training[:, :len(words)]
  train_y = training[:, len(words):]
  return train_x, train_y

def build_model(train_x, train_y):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

  sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  model.fit(train_x, train_y, epochs=150, batch_size=5, verbose=1)
  model.save('chatbot_model.h5')
  print('Done')

if __name__ == "__main__":
  words, classes, documents = preprocess_data()
  train_x, train_y = create_training_data(words, classes, documents)
  build_model(train_x, train_y)
