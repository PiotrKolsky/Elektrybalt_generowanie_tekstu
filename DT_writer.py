
import os
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

###################################################
# tekst uczący
os.chdir('../Robocze')
with open('pan-tadeusz-czyli-ostatni-zajazd-na-litwie.txt', 'r') as f:
    voc = f.read().replace('\n', ' ')

# słowniki znaków
word_list = list([[i,w] for i,w in enumerate(voc)])
word_dict = dict([(w, i) for i,w in enumerate(set(voc))])
word_dict_inv = {v: k for k, v in word_dict.items()}

# przypisanie do listy znaków ich numerów wg słownika
for e in word_list[:]:
    e[0] = word_dict[e[1]]

# funkcja ucząco-pisząca
def DT_writer(text, n_letters=8, max_d=30, text_len=1000):
    # tablice liter do treningu
    X = []; y = [];
    for j in range(len(word_list)-n_letters):
        temp = []
        for i in range(j, j+n_letters):
            temp.append(word_list[i][0])
        X.append(temp)
        y.append(word_list[i+1][0])
    # trening klasyfikatora
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=max_d) #kryterium podziału gałęzi: gini lub entropy
    #classifier = RandomForestClassifier(criterion='gini', max_depth=max_d) #alternatywny algorytm
    classifier.fit(np.array(X), y)
    # predycja
    y_pred = []
    X_pred = [word_dict[e] for e in text[-n_letters:]]
    for i in range(text_len):
        if i % random.randint(round(text_len/10),round(text_len/8)) == 0:
            y_temp = random.choice(list(word_dict.values())) #losowe znaki dla uniknięcia zapętlenia
        else:
            y_temp = classifier.predict([X_pred])[0]
            y_pred.append(y_temp)
        X_pred = X_pred[1:]
        X_pred.append(y_temp) 
    # formatowanie tekstu w liniach po 100 znaków
    for e in y_pred:
        text = text + word_dict_inv[e]
    text_f = ''
    for e in text:
        text_f = text_f + e
        if len(text_f) % 100 == 0:
            text_f = text_f + '\n'
    print(text_f)

# składnia polecenia: 'tekst inicjujący', ilość liter analizowanych przed prognozowaną, głębokość drzewa, długość pisanego tekstu.
DT_writer('Jako tam szlachta na Litwie żyła? ', 8, 25, 1500)
############################################################
