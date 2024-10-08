# Paths for data.
Ravdess = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
Crema = "/kaggle/input/cremad/AudioWAV/"
Tess = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
Savee = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()

crema_directory_list = os.listdir(Crema)

file_emotion = []
file_path = []

for file in crema_directory_list:
    # storing file paths
    file_path.append(Crema + file)
    # storing file emotions
    part=file.split('_')
    if part[2] == 'SAD':
        file_emotion.append('sad')
    elif part[2] == 'ANG':
        file_emotion.append('angry')
    elif part[2] == 'DIS':
        file_emotion.append('disgust')
    elif part[2] == 'FEA':
        file_emotion.append('fear')
    elif part[2] == 'HAP':
        file_emotion.append('happy')
    elif part[2] == 'NEU':
        file_emotion.append('neutral')
    else:
        file_emotion.append('Unknown')
        
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()


tess_directory_list = os.listdir(Tess)

file_emotion = []
file_path = []

for dir in tess_directory_list:
    directories = os.listdir(Tess + dir)
    for file in directories:
        part = file.split('.')[0]
        part = part.split('_')[2]
        if part=='ps':
            file_emotion.append('surprise')
        else:
            file_emotion.append(part)
        file_path.append(Tess + dir + '/' + file)
        
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.head()

savee_directory_list = os.listdir(Savee)

file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')
        
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.head()

linkcode
# creare DataFrame cu toate seturile de date
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis = 0)
data_path.to_csv("data_path.csv",index=False)
data_path.shape

rows_to_drop = []

# Iterăm prin DataFrame folosind iterația cu .iterrows()
for index, row in data_path.iterrows():
    if row['Emotions'] == 'disgust':
        rows_to_drop.append(index)

# Eliminăm rândurile corespunzătoare din DataFrame
data_path.drop(rows_to_drop, inplace=True)
data_path.shape

import pandas as pd

# Presupunând că data_path este DataFrame-ul tău și 'Emotions' este o coloană din acesta

# Inițializăm lista pentru rândurile care trebuie eliminate
rows_to_drop = []

# Iterăm prin DataFrame și adăugăm indecșii rândurilor care trebuie eliminate pentru 'sad'
i = 0
for index, row in data_path.iterrows():
    if row['Emotions'] == 'sad':
        rows_to_drop.append(index)
        i += 1
    if i > 1000:
        break  # Sărim din buclă după ce am adăugat 1000 de rânduri pentru 'sad'

# Iterăm din nou prin DataFrame și adăugăm indecșii rândurilor care trebuie eliminate pentru 'angry'
i = 0
for index, row in data_path.iterrows():
    if row['Emotions'] == 'angry':
        rows_to_drop.append(index)
        i += 1
    if i > 1000:
        break  # Sărim din buclă după ce am adăugat 1000 de rânduri pentru 'angry'
i = 0
for index, row in data_path.iterrows():
    if row['Emotions'] == 'fear':
        rows_to_drop.append(index)
        i += 1
    if i > 1000:
        break  # Sărim din buclă după ce am adăugat 1000 de rânduri pentru 'angry'
# Eliminăm rândurile corespunzătoare din DataFrame
data_path.drop(rows_to_drop, inplace=True)

# Verificăm dimensiunea DataFrame-ului după eliminare
print(data_path.shape)
