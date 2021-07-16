#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sieć neuronowa typu LSTM pisząca tekst przez wybór kolejnych liter
"""
import os
import warnings
warnings.filterwarnings('ignore')

import joblib
import random
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

os.chdir('.../Work')


## Wykonanie ramki danych uczących dla sieci LSTM
def data_creation( file, inp_text_len=4000, n_letters=10 ):
    with open(file, 'r') as f:
        voc = f.read().replace('\n', ' ').translate({ord(e): None for e in '„”-/—'}).\
        lower().replace('tom', '').replace('rozdział', '')
        voc = ' '.join(voc.split()) 
    df = pd.get_dummies([e for e in voc[39:inp_text_len] ])
   
    X = []; y = [];
    for i in range(len(df)-n_letters):
        X.append(df.iloc[i:i+n_letters,:].values)
        y.append(df.iloc[i+n_letters,:].values)
    
    X = np.array( X ); y = np.array( y )
    print( 'X:', X.shape, '  y:', y.shape )
    return X, y, df


## Budowa modelu
def model_build( X_shape, y_shape ):
    model = Sequential()
    model.add(LSTM(units = y_shape[1]*3, return_sequences = True, input_shape = (X_shape[1], X_shape[2])))
    model.add(LSTM(units = y_shape[1]*2, return_sequences = True))
    model.add(Dropout(0.1))
    model.add(LSTM(units = y_shape[1]*2)) 
    model.add(Dense(units = y_shape[1], activation='softmax')) # relu sigmoid softmax
    model.compile(loss='categorical_crossentropy', optimizer='adam') # adam sgd   mean_squared_error categorical_crossentropy
    print(model.summary())
    return model


## Wykonanie ramki danych z tekstu startowego
def frame_maker( start_str, n_letters ):
    start_array = []
    for e in start_str:
        start_array = start_array + [df.columns==e ]
    
    start_array = np.array(start_array).astype(int)
    start_array.shape
    
    X_start = []; 
    for i in range(len(start_array)-n_letters):
        X_start.append(start_array[i:i+n_letters,:])
    
    X_start = np.array(X_start)
    return X_start


## Zmiana losowej litery w łańcuchu na inną o mniejszej częstości następstwa, dla uniknięcia zapętlenia sieci
def letter_change( l ):
    ind = df[df.iloc[ :, l ]==1].index
    if ind[-1] == len(df)-1:
        ind = ind[:-1]
    else: pass

    new_letter = df.iloc[( ind+1 ).tolist(),:].sum().sort_values(axis=0, ascending=False).index[random.choice([1,2,3])]
    return (df.columns==new_letter).argmax()


## Predykcja następnej litery i dodanie jej do danych wejściowych kolejnej predykcji
def story_writing( X_start, pred_len = 1000, retro_start = -10, diff_letter = 0 ): # 1/0 - włącz zmianę losowej litery
    X_new = X_start[retro_start, :, :]; story = ''
    for i in range(pred_len):
        pred = model.predict_classes( X_new.reshape(-1, X_start.shape[1], X_start.shape[2] ) )
        new_line = np.zeros(y.shape[1])
        
        if diff_letter==1 and (i+1) % random.randint(200, 222) == 0: 
            new_line[ letter_change( pred[0] ) ]
        else:
            new_line[ pred[0] ] = 1
            
        X_new = np.row_stack(( np.delete(X_new, 0, axis=0), new_line)) 
        story+=df.columns[pred[0]]
    return story


## #######################################
## Uruchomienie, nauka i testowanie modelu

n_letters = 10

if __name__ == "__main__":
    X, y, df = data_creation( os.getcwd()+'/krzyzacy1.txt', 4000, n_letters )
  
    model = model_build( X.shape, y.shape )
    model.fit(X, y, 
              epochs = 60, 
              batch_size = 10, 
              verbose = 1, 
              callbacks=[ EarlyStopping( monitor='val_loss', mode='min', min_delta=0.005, 
                                        restore_best_weights=True, verbose=1, patience=2 ) ] )
    #joblib.dump(model,'model_lstm.pkl')
    #model = joblib.load('model_lstm.pkl')
   
    start_str = input( 'Wprowadź tekst początkowy dla bota, coś w stylu średniowiecza >20 zn: ' ).lower()
    
    print( start_str, '\n'*2, story_writing( frame_maker( start_str, n_letters ) , 1200, -8, 1 ) )

    #start_str = 'w gospodzie pod lutym turem, należącej do opactwa, siedziało kilku ludzi, \
    #słuchając opowiadania wojaka bywalca, który z dalekich stron przybywszy, prawił im o przygodach, \
    #jakich na wojnie i w czasie podróży doznał. człek był brodaty, w sile wieku, pleczysty, prawie ogromny, \
    #ale wychudły; włosy nosił ujęte w pątlik'.lower()
