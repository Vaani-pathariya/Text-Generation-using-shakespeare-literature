from tabnanny import verbose
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import random
import tensorflow as tf
from keras import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from keras.layers import Activation,Dense
from keras.optimizers import RMSprop
filepath=tf.keras.utils.get_file('shakespare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()
text=text[300000:800000]
charachters=sorted(set(text))
char_to_index=dict((c,i)for i,c in enumerate (charachters))
index_to_char=dict((i,c)for i,c in enumerate (charachters))
seq_length=40
step_size=3
'''
sentences=[]
next_characters=[]

for i in range(0,len(text)-seq_length,step_size):
    sentences.append(text[i:i+seq_length])
    next_characters.append(text[i+seq_length])
x=np.zeros((len(sentences),seq_length,len(charachters)),dtype=np.float32)
y=np.zeros((len(sentences),len(charachters)),dtype=np.float32)
for i,sentence in enumerate(sentences):
    for t,charachter in enumerate(sentence):
        x[i,t,char_to_index[charachter]]=1
    y[i,char_to_index[next_characters[i]]]=1
    '''
new_model = tf.keras.models.load_model('textgenerator.model')
def sample(preds,temperature=1.0):
    preds=np.asarray(preds).astype("float64")
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)
def generate_text(length,temperature):
    start_index=random.randint(0,len(text)-seq_length-1)
    generated=""
    sentence=text[start_index: start_index+ seq_length]
    generated+= sentence
    for i in range(length):
        x= np.zeros((1,seq_length,len(charachters)))
        for t,charachter in enumerate(sentence):
            x[0,t,char_to_index[charachter]]=1
        predictions=new_model.predict(x,verbose=0)[0]
        next_index=sample(predictions,temperature)
        next_character=index_to_char[next_index]
        generated+=next_character
        sentence=sentence[1:]+next_character
    return generated
print("=========0.2=========")
print (generate_text(300,0.2))
print("=========0.4=========")
print (generate_text(300,0.4))
print("=========0.6=========")
print (generate_text(300,0.6))
print("=========0.8=========")
print (generate_text(300,0.8))
print("=========1.0=========")
print (generate_text(300,1.0))

