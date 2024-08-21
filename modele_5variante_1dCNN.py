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
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

#Features=pd.read_csv('features_savee.csv')
#Features=pd.read_csv('features_tess_128mels.csv')
#Features=pd.read_csv('features_ravdess_128mels.csv')
filename="features_all_7clase_128mel"
Features=pd.read_csv('7clase/features_all_7clase_128mels.csv')
Features.head()
#separarea caracteristicilor
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
#re-etichetarea valorilor in 3 categorii
'''
Y = np.where(np.isin(Y, ['sad', 'fear','angry','disgust']), 'negativa', Y)
Y = np.where(np.isin(Y, ['happy', 'surprise']), 'pozitiva', Y)
Y = np.where(np.isin(Y, ['calm', 'neutral','unknown']), 'neutra', Y)
'''
'''
Y = np.where(np.isin(Y, ['negative']), 'negativa', Y)
Y = np.where(np.isin(Y, ['positive']), 'pozitiva', Y)
Y = np.where(np.isin(Y, ['neutral']), 'neutra', Y)
'''
'''
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

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

#model1

model = Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.3))

# Modifică stratul final pentru a avea 3 unități
model.add(Dense(units=8, activation='softmax'))
# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.00001)
#rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.00001)
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

print("1\nAcuratetea modelului 1 pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

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
plt.savefig(filename+"m1_curba.png")
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
plt.savefig(filename+"_m1_cm.png")
plt.show()
#raportul de clasificare
report=classification_report(y_test, y_pred, output_dict=True)

df=pd.DataFrame(report).transpose()
df.to_csv(filename+"_m1_cr.txt");
print(classification_report(y_test, y_pred))
'''
'''
Features=pd.read_csv('7clase/features_all_7clase_128mels.csv')
Features.head()
#separarea caracteristicilor
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values

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
'''
#model 2
'''
from keras.layers import Input, Flatten, Dropout, Activation
model = Sequential()
model.add(Conv1D(256, 5,padding='same',input_shape=(x_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())

# Modifică stratul final pentru a avea 3 unități
model.add(Dense(units=8, activation='softmax'))
# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.00001)
#rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.00001)
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

print("2\nAcuratetea modelului 2 pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

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
plt.savefig(filename+"m2_curba.png")
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
plt.savefig(filename+"_m2_cm.png")
plt.show()
#raportul de clasificare
report=classification_report(y_test, y_pred, output_dict=True)

df=pd.DataFrame(report).transpose()
df.to_csv(filename+"_m2_cr.txt");
print(classification_report(y_test, y_pred))

'''
'''
Features=pd.read_csv('7clase/features_all_7clase_128mels.csv')
Features.head()
#separarea caracteristicilor
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
#re-etichetarea valorilor in 3 categorii

# Y = np.where(np.isin(Y, ['sad', 'fear','angry','disgust']), 'negativa', Y)
# Y = np.where(np.isin(Y, ['happy', 'surprise']), 'pozitiva', Y)
# Y = np.where(np.isin(Y, ['calm', 'neutral','unknown']), 'neutra', Y)
# 
# Y = np.where(np.isin(Y, ['negative']), 'negativa', Y)
# Y = np.where(np.isin(Y, ['positive']), 'pozitiva', Y)
# Y = np.where(np.isin(Y, ['neutral']), 'neutra', Y)


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
#model 3

from keras.layers import Input, Flatten, Dropout, Activation
model = Sequential()
model.add(Conv1D(64, 5,padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(256, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Flatten())

# Modifică stratul final pentru a avea 3 unități
model.add(Dense(units=8, activation='softmax'))
# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.00001)
#rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.00001)
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

print("3\nAcuratetea modelului 3 pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

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
plt.savefig(filename+"m3_curba.png")
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
plt.savefig(filename+"_m3_cm.png")
plt.show()
#raportul de clasificare
report=classification_report(y_test, y_pred, output_dict=True)

df=pd.DataFrame(report).transpose()
df.to_csv(filename+"_m3_cr.txt");
print(classification_report(y_test, y_pred))

Features=pd.read_csv('7clase/features_all_7clase_128mels.csv')
Features.head()
#separarea caracteristicilor
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
#re-etichetarea valorilor in 3 categorii

# Y = np.where(np.isin(Y, ['sad', 'fear','angry','disgust']), 'negativa', Y)
# Y = np.where(np.isin(Y, ['happy', 'surprise']), 'pozitiva', Y)
# Y = np.where(np.isin(Y, ['calm', 'neutral','unknown']), 'neutra', Y)
# 
# 
# Y = np.where(np.isin(Y, ['negative']), 'negativa', Y)
# Y = np.where(np.isin(Y, ['positive']), 'pozitiva', Y)
# Y = np.where(np.isin(Y, ['neutral']), 'neutra', Y)


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

#model 4 
from keras.layers import Input, Flatten, Dropout, Activation
model = Sequential()
model.add(Conv1D(256, 5,padding='same', input_shape=(x_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
# Modifică stratul final pentru a avea 3 unități
model.add(Dense(units=8, activation='softmax'))
# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.00001)
#rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.00001)
history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])

print("4\nAcuratetea modelului 4 pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

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
plt.savefig(filename+"m4_curba.png")
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
plt.savefig(filename+"_m4_cm.png")
plt.show()
#raportul de clasificare
report=classification_report(y_test, y_pred, output_dict=True)

df=pd.DataFrame(report).transpose()
df.to_csv(filename+"_m4_cr.txt");
print(classification_report(y_test, y_pred))

'''

Features=pd.read_csv('7clase/features_all_7clase_128mels.csv')
Features.head()
#separarea caracteristicilor
X = Features.iloc[: ,:-1].values
Y = Features['labels'].values
#re-etichetarea valorilor in 3 categorii

# Y = np.where(np.isin(Y, ['sad', 'fear','angry','disgust']), 'negativa', Y)
# Y = np.where(np.isin(Y, ['happy', 'surprise']), 'pozitiva', Y)
# Y = np.where(np.isin(Y, ['calm', 'neutral','unknown']), 'neutra', Y)


# Y = np.where(np.isin(Y, ['negative']), 'negativa', Y)
# Y = np.where(np.isin(Y, ['positive']), 'pozitiva', Y)
# Y = np.where(np.isin(Y, ['neutral']), 'neutra', Y)


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
import tensorflow.keras.layers as L

model = Sequential([
    L.Conv1D(256,kernel_size=5, strides=1,padding='same', activation='relu', input_shape=(x_train.shape[1], 1)),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),

    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    L.Dropout(0.2),  

    L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),

    L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=5,strides=2,padding='same'),
    L.Dropout(0.2), 

    L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
    L.BatchNormalization(),
    L.MaxPool1D(pool_size=3,strides=2,padding='same'),
    L.Dropout(0.2), 

    L.Flatten(),
    L.Dense(256,activation='relu'),
    L.BatchNormalization(),
    #L.Dense(3,activation='softmax')
])

# Modifică stratul final pentru a avea 3 unități
model.add(Dense(units=8, activation='softmax'))
# Compilarea modelului
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=0, patience=2, min_lr=0.00001)
#rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.00001)
#history=model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])
history=model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks=[rlrp])


print("5\nAcuratetea modelului pe date de testare: " , model.evaluate(x_test,y_test)[1]*100 , "%")

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


