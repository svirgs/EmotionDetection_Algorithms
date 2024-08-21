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
filename="features_all_3clase_128mels_lstm2"
Features=pd.read_csv('3clase/features_128mels/features_all_3clase.csv')
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

#model 5
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the model
model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1], 1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Define ReduceLROnPlateau callback
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.00001)

# Train the model
history = model.fit(x_train, y_train, batch_size=16, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

model.summary()

#history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

print("LSTM \nAcuratetea modelului pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(100)]
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
plt.savefig(filename+"_m5_curba.png")
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
plt.savefig(filename+"_m5_cm.png")
plt.show()
#raportul de clasificare
report=classification_report(y_test, y_pred, output_dict=True)

df=pd.DataFrame(report).transpose()
df.to_csv(filename+"_m5_cr.txt");
print(classification_report(y_test, y_pred))




