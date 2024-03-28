import json
import pickle
import numpy as np
import nltk
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())['intents']
lemmatizer = WordNetLemmatizer()


def prepare_user_input(sentence):
    words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word) for word in words]  


def create_bag_of_words(sentence_words):
    bag = np.zeros(len(words))
    for word in sentence_words:
        if word in words:
            bag[words.index(word)] = 1
    return bag


def classify_intent(clean_sentence):
    bog = create_bag_of_words(clean_sentence)
    prediction = model.predict(bog.reshape(1, -1))[0]
    results = [[i, r] for i, r in enumerate(prediction) if r > 0.25] 
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def retrieve_response(predicted_intents, intents_json):
    tag = predicted_intents[0]['intent']
    for intent in intents_json:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])


def start_chatbot():
    print("Starting Chatbot")
    while True:
        user_input = input(">> ")
        clean_words = prepare_user_input(user_input)
        predicted_intents = classify_intent(clean_words)
        response = retrieve_response(predicted_intents, intents)
        print(response)


if __name__ == "__main__":
    start_chatbot()