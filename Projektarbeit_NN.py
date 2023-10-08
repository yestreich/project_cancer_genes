'''

= Projektarbeit - Neuronale Netze - binäre Klassifikation von Kolorektalkrebs =

'''

#%% Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils 

from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout 
from keras import regularizers

from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
 
import tensorflow
import keras

#%% Datensatz aufbereiten

dataset = pd.read_csv("02_Neuronale Netze/04_Woche 4/Projektarbeit/Colorectal_GSE44861.csv")
print(dataset.shape) # 105 Datenreihen, 22.279 Features

#Sample-Nummerierung wird nicht gebraucht
dataset = dataset.drop("samples", axis = 1)

#Datensatz in Features (X) und Zielvariablen (y) aufteilen
X = dataset.iloc[:, 1:].astype("float64")
y = np.array(dataset["type"])

#tensorflow.random.set_seed(7)
#seed(6)
random.seed(23)

#%% Train-Test-Split

#Datensatz für Training und Testen aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 2, stratify = y)

#Trainingsdaten für Training und Validierung aufteilen
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, 
                                                  random_state = 2, stratify=y_train)

#%% Scaling

#MinMaxScaler, weil ReLU später benutzt wird
scaler = MinMaxScaler() # Kann neue Input-Werte ins negative ziehen. 
                        # Jenachdem wie die Range der Rohdaten ist.

#Scaler mit Trainings-Input "formen" und auf alle Inputs anwenden
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

#%% PCA - Dimensionen reduzieren

#22277 anfängliche Dimensionen auf 10 Dimensionen brechen
pca = PCA(n_components=10)

X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)
X_val_new = pca.transform(X_val)

# Wie viel Inhalt stecken in 10 Dimensionen?
print(sum(pca.explained_variance_ratio_)) 
# -> Der neue Datensatz kann 66.3% des Inhalts abbilden.
# 1/3 Daten"verlust"

#%% Encoding

# Kodierung der Output-strings ("tumoral", "normal") als Zahlen
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
y_val = encoder.transform(y_val)

#Binäres Encoding : aus 1 wird [0,1] (quasi OneHotEncoding)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)

# %% Modell bauen

# Menge an Input- & Outputknoten vordefinieren
net_input = X_train_new.shape[1]
net_output = y_train.shape[1]

def model_net():
    net = Sequential() 
    net.add(Dense(8, input_dim = net_input, activation = "relu",
                  #Regularizer L1
                  kernel_regularizer=regularizers.L1(l1=0.1),
                  #Gewichtsinitialisierung
                  kernel_initializer=keras.initializers.glorot_uniform(seed=123)))
    net.add(Dense(8, activation = "relu",
                  #Regularizer L1
                  kernel_regularizer=regularizers.L1(l1=0.1),
                  #Gewichtsinitialisierung
                  kernel_initializer=keras.initializers.glorot_uniform(seed=123)))
    net.add(Dropout(0.25))
    net.add(Dense(8, activation = "relu",
                  #Regularizer L1
                  kernel_regularizer=regularizers.L1(l1=0.1),
                  #Gewichtsinitialisierung
                  kernel_initializer=keras.initializers.glorot_uniform(seed=123)))
    net.add(Dropout(0.25))

    # Output-Layer, softmax für Wahrscheinlichkeit
    net.add(Dense(net_output, activation = "softmax")) 
    #Zusammenstellung
    #categorical_crossentropy gewählt weil zwei Outputklassen
    net.compile(loss = "categorical_crossentropy", 
                optimizer = keras.optimizers.Adam(learning_rate=0.001), 
                metrics=["accuracy"]) 
    return net

#Modell bauen
model = KerasClassifier(model = model_net)

#%% GridSearchCV

 #Kurzer Testlauf mit Epochen ist schon geschehen. Dieser Bereich deckt viel ab
epochs = [50, 100, 150, 200]
# Nur 67 Datenreihen in Trainingsdaten. Größere Batches sind kaum sinnvoll
batches = [1, 2, 4, 8, 16] 

param_grid = dict(epochs = epochs, 
                  batch_size = batches)

kfold = KFold(n_splits=3, shuffle=True) # Zu wenig Daten um mehr splits zu machen

#Kreuzvalidierung : Welche Kombination aus Epochen und Batches ist die Beste
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = kfold)
grid.result = grid.fit(X_train_new, y_train)

#Ausgabe der besten Kombination und aller anderen Kombinationen
print(grid.result.best_score_)
print(grid.result.best_params_)

means = grid.cv_results_['mean_test_score']
params = grid.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))

#%% Trainieren

#EarlyStoping einbauen
stop_early = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                                      patience=3)

#Trainieren mit bester Kombination aus GridSearchCV
trained = model.fit(X_train_new, y_train,
                    epochs = 150,
                    batch_size = 2,
                    verbose = 1,
                    validation_data = (X_val_new, y_val), 
                    callbacks = [stop_early]
                    ) 
#%% Kurven plotten

# fit macht callback.history-Typ. Darin stehen die einzelnen Werte pro Epoche 
# -> Plottbar

#fig = plt.figure(figsize=(9, 4))
plt.plot(trained.history_["loss"])
plt.plot(trained.history_["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["loss", "val_loss"])
plt.title("Loss of training data and validation data")
plt.savefig("02_Neuronale Netze/04_Woche 4/Projektarbeit/01_Loss")
plt.show()

#fig = plt.figure(figsize=(6, 5))
plt.plot(trained.history_["accuracy"])
plt.plot(trained.history_["val_accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["accuracy", "val_accuracy"])
plt.title("Accuracy of training data and validation data")
plt.savefig("02_Neuronale Netze/04_Woche 4/Projektarbeit/02_Accuracy")
plt.show()

#%% Testen & Testscore

#Test-Score
test_accuracy  = trained.score(X_test_new, y_test)
print("Test Accuracy:", test_accuracy)

#%% Simulation: neue Daten

Xnew = np.random.uniform(low=4.2, high=16.1, size=(1, 22277))

Xnew = scaler.transform(Xnew)
Xnew = pca.transform(Xnew)
ynew = model.predict(Xnew)  

# invert normalize
inv = np.argmax(ynew)
ynew = encoder.inverse_transform([inv])
print(ynew[0])

