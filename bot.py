import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

# utilisation d'un dictionnaire pour représenter un fichier JSON d'intentions
data = {"intents": [
             {"tag": "greeting",
              "patterns": ["Hello", "La forme?", "yo", "Salut", "ça roule?"],
              "responses": ["Salut à toi!", "Hello", "Comment vas tu?", "Salutations!", "Enchanté"],
             },
             {"tag": "age",
              "patterns": ["Quel âge as-tu?", "C'est quand ton anniversaire?", "Quand es-tu né?"],
              "responses": ["J'ai 25 ans", "Je suis né en 1996", "Ma date d'anniversaire est le 3 juillet et je suis né en 1996", "03/07/1996"]
             },
             {"tag": "appartenance",
              "patterns": ["Tu es dans quelle école ?","Tu étudies où ?","Tu fais quoi dans la vie ?"],
              "responses": ["Je suis étudiant au campus Ynov Sophia", "Je suis à Sophia Ynov Campus", "J'étudie à Ynov"]
             },
             {"tag": "name",
              "patterns": ["Quel est ton prénom?", "Comment tu t'appelles?", "Qui es-tu?"],
              "responses": ["Mon prénom est Miki", "Je suis Miki", "Miki"]
             },
             {"tag": "goodbye",
              "patterns": [ "bye", "Salut", "see ya", "adios", "cya"],
              "responses": ["Je n'ai pas compris votre question"]
             },
             {"tag": "school",
              "patterns": [ "C'est quoi Ynov ?","Que sais tu de Ynov ?"],
              "responses": ["Ynov est une école supérieure spécialisée dans les domaines du numérique, de la créativité et de l'innovation, formant les futurs talents de l'industrie technologique."]
             },
             {"tag": "nom_ecole",
             "patterns": ["Quel est le nom de ton école?", "Comment s'appelle ton école?", "C'est quoi le nom de ton école?"],
             "responses": ["Mon école s'appelle Ynov", "L'école où je suis inscrit se nomme Ynov", "Ynov"]
             },
             {"tag": "localisation",
             "patterns": ["Où est située ton école?", "Dans quelle ville se trouve ton école?", "C'est où Ynov?"],
             "responses": ["Ynov est située dans plusieurs villes en France, dont Bordeaux, Lyon, Paris, Aix-en-Provence, etc.", "Mon école a plusieurs campus répartis dans différentes villes françaises."]
             },
             {"tag": "programmes",
             "patterns": ["Quels programmes propose ton école?", "Quels types de formations sont disponibles à Ynov?", "Peux-tu me parler des filières à Ynov?"],
             "responses": ["Ynov propose une variété de programmes dans les domaines du numérique, de la créativité et de l'innovation, tels que le développement web, la cybersécurité, le design graphique, l'animation 3D, etc."]
             },
             {"tag": "inscription",
             "patterns": ["Comment s'inscrire à Ynov?", "Quelles sont les modalités d'admission à Ynov?", "Peux-tu m'aider à m'inscrire à Ynov?","Comment s'y inscrire ?","Comment je peux m'y inscrire ?"],
             "responses": ["Pour s'inscrire à Ynov, il faut généralement remplir un formulaire en ligne sur le site officiel de l'école et suivre les instructions fournies par l'administration.", "Les modalités d'admission varient selon les programmes et les campus, il est conseillé de consulter le site web de l'école ou de contacter l'administration pour plus d'informations."]
             },
             {"tag": "services",
             "patterns": ["Quels sont les services disponibles à Ynov?", "Ynov offre-t-elle des services aux étudiants?", "Peux-tu me parler des infrastructures de Ynov?"],
             "responses": ["Ynov propose une gamme de services aux étudiants, y compris des espaces de travail collaboratifs, des laboratoires équipés, un réseau d'anciens élèves, des opportunités de stage et d'emploi, etc."]
             },
             {"tag": "histoire",
             "patterns": ["Peux-tu me raconter l'histoire d'Ynov?", "Comment Ynov a-t-elle été fondée?", "Quelle est l'histoire de Ynov?"],
             "responses": ["Ynov a été fondée en 2009 par Benoît Raphaël. Initialement connue sous le nom de 'Web@cadémie', elle proposait des formations axées sur le développement web. Au fil des années, l'école a élargi son offre de formations pour inclure les domaines de la création numérique et de l'innovation. Aujourd'hui, Ynov est devenue l'une des principales écoles supérieures dans les domaines du numérique en France, offrant une variété de programmes de qualité et formant les futurs talents de l'industrie."]
             }
        
]}

# initialisation de lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()
# création des listes
words = []
classes = []
doc_X = []
doc_y = []
# parcourir avec une boucle For toutes les intentions
# tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
# le tag associé à l'intention sont ajoutés aux listes correspondantes
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    
    # ajouter le tag aux classes s'il n'est pas déjà là 
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatiser tous les mots du vocabulaire et les convertir en minuscule
# si les mots n'apparaissent pas dans la ponctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
# trier le vocabulaire et les classes par ordre alphabétique et prendre le
# set pour s'assurer qu'il n'y a pas de doublons
words = sorted(set(words))
classes = sorted(set(classes))

# liste pour les données d'entraînement
training = []
out_empty = [0] * len(classes)
# création du modèle d'ensemble de mots
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    # marque l'index de la classe à laquelle le pattern atguel est associé à
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1
    # ajoute le one hot encoded BoW et les classes associées à la liste training
    training.append([bow, output_row])
# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)
# séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# définition de quelques paramètres
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 200

# modèle Deep Learning
model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

# entraînement du modèle
model.fit(x=train_X, y=train_y, epochs=200, verbose=1)

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens
def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)
def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list
def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

  # lancement du chatbot
while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)