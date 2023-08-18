
import streamlit as st
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization,
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,
                          Dropout)
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from keras.initializers import glorot_uniform
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img,img_to_array
from streamlit import caching

st.write("Music Genre Recognition App") #to createa a script and write all the code for app
st.write("This is a Web App to predict genre of music")
file = st.sidebar.file_uploader("Please Upload Mp3 Audio File Here or Use Demo Of App Below using Preloaded Music",type=["mp3"])

from PIL import Image
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize


class_labels = ['blues', #music genre types
                'classical',
                'country',
                'disco',
                'hiphop',
                'metal',
                'pop',
                'reggae',
                'rock']


def GenreModel(input_shape=(288, 432, 4), classes=9):
  X_input = Input(input_shape)

  X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
  X = BatchNormalization(axis=3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Conv2D(256, kernel_size=(3, 3), strides=(1, 1))(X)
  X = BatchNormalization(axis=-1)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((2, 2))(X)

  X = Flatten()(X)

  # X = Dropout(rate=0.3)(X)

  # X = Dense(256,activation='relu')(X)

  # X = Dropout(rate=0.4)(X)

  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=9))(X)

  model = Model(inputs=X_input, outputs=X, name='GenreModel')

  return model

model = GenreModel(input_shape=(288,432,4),classes=9)
model.load_weights("genre.h5")


def convert_mp3_to_wav(music_file): # a function which is converting mp3 audio files into .wav files,
  sound = AudioSegment.from_mp3(music_file) # because Librosa works only with .wav files.
  sound.export("music_file.wav", format="wav")


def extract_relevant(wav_file, t1, t2): # it's a function which extracts 3 sec of audio from our music
  wav = AudioSegment.from_wav(wav_file)
  wav = wav[1000 * t1:1000 * t2]
  wav.export("extracted.wav", format='wav')


def create_melspectrogram(wav_file): # creates a mel spectrogram
  y, sr = librosa.load(wav_file, duration=5)
  mels = librosa.feature.melspectrogram(y=y, sr=sr)

  fig = plt.Figure()
  canvas = FigureCanvas(fig)
  p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
  plt.savefig('melspectrogram.png')


def predict(image_data, model): # to predict the genre of music using mel spectrogram generated above
  # image = image_data.resize((288,432))
  image = img_to_array(image_data)

  image = np.reshape(image, (1, 288, 432, 4))

  prediction = model.predict(image / 255)

  prediction = prediction.reshape((9,))

  class_label = np.argmax(prediction)

  return class_label, prediction
#class_label is the label with highest probability, and prediction captures the probability distribution over all classes

if file is None: # we will merge all the functions we wrote to display the final output on our web app
  st.text("Please upload an mp3 file")
else:
  convert_mp3_to_wav(file)
  extract_relevant("music_file.wav",40,50)
  create_melspectrogram("extracted.wav")
  image_data =   load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
  class_label,prediction = predict(image_data,model)
  st.write("## The Genre of Song is "+class_labels[class_label])












