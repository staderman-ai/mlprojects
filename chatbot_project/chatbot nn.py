#!/usr/bin/env python
# coding: utf-8


import nltk

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import numpy as np


words = []
tags = []
documents = []
non_words = ['?', '!']

data_file = open('training_data.json').read()
chat_data = json.loads(data_file)


for intent in chat_data['training_data']: # take each section of the training data
    for pattern in intent['patterns']: # take each pattern in the current section
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])
        
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in non_words]


words = sorted(list(set(words)))
tags = sorted(list(set(tags)))


print(words)


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

print(documents[0])


# create training data
training_data = []

for document in documents:
    class_labels = [0] * len(tags)
    bag = [0] * len(words)
    for w in document[0]:
        for i in range(len(words)):
            if words[i] == lemmatizer.lemmatize(w.lower()):
                bag[i] = 1
                break
    class_labels[tags.index(document[1])] = 1
    training_data.append([bag, class_labels])
        
random.shuffle(training_data)
training_data = np.array(training_data)
x_train = list(training_data[:,0])
y_train = list(training_data[:,1])
    
len(y_train[0])



model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(len(x_train[0]),)))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(x_train, y_train, epochs=200, batch_size=10, verbose=1)


model.save('abhijitm_chatbot.h5', hist)


len(x_train[0])


np.array(x_train[0]).reshape(1,90)


tags[model.predict_classes(np.array(x_train[0]).reshape(1,90))[0]]



# use the model and predict a response based on the users input
def clean_sentence(sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = [lemmatizer.lemmatize(w.lower()) for w in sentence if w not in non_words]
    return sentence

def create_model_input(sentence):
    sentence = clean_sentence(sentence)
    bag = [0] * len(words)
    for word in sentence:
        for i in range(len(words)):
            if words[i] == word:
                bag[i] = 1
                break
    return bag

def predict_tag(sentence):
    bag = create_model_input(sentence)
    return model.predict_classes(np.array(bag).reshape(1,90))[0]

def get_response(sentence):
    tag = tags[predict_tag(sentence)]
    for intent in chat_data['training_data']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    


get_response('how are you')


import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = get_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()




