#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:34:56 2024

@author: administrator
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:13:35 2024

@author: petea
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from IPython.display import Audio
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
import tensorflow

from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

#Features=pd.read_csv('features_savee.csv')
#Features=pd.read_csv('features_tess_128mels.csv')
#Features=pd.read_csv('features_ravdess_128mels.csv')
nr_epoci=150
batch_size=64
learning_rate=0.0001
filename="GRU_7clase_leakyRelu_ravdess"+str(nr_epoci)+"_"+str(learning_rate)+"_"+str(batch_size)
Features=pd.read_csv('7clase/features_ravdess_7clase_128mel.csv')
Features.head()
#separarea caracteristicilor
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
#re-etichetarea valorilor in 3 categorii



#transfomarea etichetelor categorice intr-un format numeric
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

#x_train.shape, y_train.shape, x_test.shape, y_test.shape
#scalarea datelor
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#x_train.shape, y_train.shape, x_test.shape, y_test.shape

#se adauga o noua dimensiune pentru a fi compatibila cu modelul CNN
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


import tensorflow.keras.layers as L
from tensorflow.keras.layers import LeakyReLU



model = Sequential([
    # Convolutional layers for feature extraction
    L.Conv1D(128, kernel_size=7, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)),
    L.BatchNormalization(),
    L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

    L.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

    L.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

    L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    L.BatchNormalization(),
    L.MaxPooling1D(pool_size=3, strides=2, padding='same'),

    # Replace Flatten with GRU layers
    L.Bidirectional(L.GRU(128, return_sequences=True)),
    L.Dropout(0.3),
    L.Bidirectional(L.GRU(128)),

    # Fully connected layers
    L.Dense(256, activation='relu'),
    L.BatchNormalization(),
    L.Dense(Y.shape[1], activation='softmax')
])


# model = Sequential([
#     # Convolutional layers for feature extraction
#     L.Conv1D(128, kernel_size=7, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

#     L.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

#     L.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

#     L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=3, strides=2, padding='same'),

#     # Additional convolutional layer for more feature extraction
#     L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=3, strides=2, padding='same'),

#     # Bidirectional GRU layers
#     L.Bidirectional(L.GRU(128, return_sequences=True)),
#     L.Dropout(0.3),
#     L.Bidirectional(L.GRU(128, return_sequences=True)),
#     L.Dropout(0.3),
#     L.Bidirectional(L.GRU(128)),

#     # Fully connected layers
#     # L.Dense(256, activation='relu'),
#     # L.BatchNormalization(),
#     # L.Dropout(0.3),
#     # L.Dense(128, activation='relu'),
#     # L.BatchNormalization(),
#     # L.Dropout(0.3),
#     # L.Dense(3, activation='softmax')
#     L.Dense(256),
#     LeakyReLU(alpha=0.01),  # Leaky ReLU instead of ReLU
#     L.BatchNormalization(),
#     L.Dropout(0.3),
#     L.Dense(128),
#     LeakyReLU(alpha=0.01),  # Leaky ReLU instead of ReLU
#     L.BatchNormalization(),
#     L.Dropout(0.3),
#     L.Dense(Y.shape[1], activation='softmax')
# ])


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define ReduceLROnPlateau callback
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=learning_rate)

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nr_epoci, validation_data=(x_test, y_test), callbacks=[rlrp])

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.layers import LeakyReLU

# model = Sequential([
#     # Convolutional layers for feature extraction
#     L.Conv1D(128, kernel_size=7, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

#     L.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

#     L.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=5, strides=2, padding='same'),

#     L.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=3, strides=2, padding='same'),

#     # Additional convolutional layer for more feature extraction
#     L.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
#     L.BatchNormalization(),
#     L.MaxPooling1D(pool_size=3, strides=2, padding='same'),

#     # Bidirectional GRU layers
#     L.Bidirectional(L.GRU(128, return_sequences=True)),
#     L.Dropout(0.3),
#     L.Bidirectional(L.GRU(128, return_sequences=True)),
#     L.Dropout(0.3),
#     L.Bidirectional(L.GRU(128)),

#     # Fully connected layers with Leaky ReLU activations
#     L.Dense(256),
#     LeakyReLU(alpha=0.01),  # Leaky ReLU instead of ReLU
#     L.BatchNormalization(),
#     L.Dropout(0.3),
#     L.Dense(128),
#     LeakyReLU(alpha=0.01),  # Leaky ReLU instead of ReLU
#     L.BatchNormalization(),
#     L.Dropout(0.3),
#     L.Dense(3, activation='softmax')
# ])

# # Compile the model with a more advanced optimizer
# optimizer = Adam(learning_rate=learning_rate)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # # Define ReduceLROnPlateau callback
# rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=learning_rate)

# # Train the model with the new callbacks
# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nr_epoci, batch_size=batch_size, callbacks=[rlrp])


model.summary()

print("LSTM \nAcuratetea modelului pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(nr_epoci)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']
fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Loss antrenare')
ax[0].plot(epochs , test_loss , label = 'Loss testare')
ax[0].set_title('Loss Antrenare si Testare')
ax[0].legend()
ax[0].set_xlabel("Epoci")
ax[1].plot(epochs , train_acc , label = 'Acuratete antrenare')
ax[1].plot(epochs , test_acc , label = 'Acuratete testare')
ax[1].set_title('Acuratete Antrenare si Testare')
ax[1].legend()
ax[1].set_xlabel("Epoci")
plt.savefig(filename+"_curba.png")
plt.show()

#se fac predictii pe datele de testare
pred_test = model.predict(x_test)
#se inverseaza codificarea pentru a obtine etichetele originale
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)
#se creeaza un DataFrame pentru a compara etichetele prezise cu cele reale
df = pd.DataFrame(columns=['Etichete prezise', 'Etichetele reale'])
df['Etichete prezise'] = y_pred.flatten()
df['Etichetele reale'] = y_test.flatten()
df.head(10)
#matricea de confuzie
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (12, 10))
cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Matricea de confuzie', size=20)
plt.xlabel('Etichetele prezise', size=14)
plt.ylabel('Etichetele reale', size=14)
plt.savefig(filename+"_cm.png")
plt.show()
#raportul de clasificare
report=classification_report(y_test, y_pred, output_dict=True)

df=pd.DataFrame(report).transpose()
df.to_csv(filename+"_cr.txt");
print(classification_report(y_test, y_pred))




