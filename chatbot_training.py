import nltk
import snowballstemmer
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import numpy
import json


with open(r"data.json",encoding='utf-8') as file:
    data = json.load(file)
    print(data)

stemmer = snowballstemmer.stemmer('turkish')
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stemWord(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))


labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stemWord(w.lower()) for w in doc]

    for w in words:

        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
model = Sequential()
model.add(Dense(16,input_shape=(len(training[0]),),activation="relu"))
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(5,activation="softmax"))
model.summary()
model.compile(Adam(lr=.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(training, output,epochs=300, verbose=2,batch_size=4)

model.save('my_model.h5')  # create
