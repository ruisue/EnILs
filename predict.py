

import matplotlib.pyplot as plt
import math
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Embedding, Convolution1D, Dropout, Activation, MaxPooling1D
from Bio import SeqIO
import pandas as pd
from nltk import trigrams, bigrams
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from tensorflow.keras.layers import Bidirectional, LSTM, BatchNormalization
import re
import tensorflow as tf
import numpy as np
from itertools import chain
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, precision_recall_curve
# from tensorflow.keras import SGD, Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from itertools import chain
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split

from Bio import SeqIO
import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from itertools import chain
from sklearn.metrics import accuracy_score

from nltk import trigrams, bigrams

from gensim.models import Word2Vec
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

import re

from sklearn.metrics import roc_curve, precision_recall_curve

from sklearn.metrics import auc


from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np

from sklearn.model_selection import train_test_split

##############word2vec
texts = []
for index, record in enumerate(SeqIO.parse('all.fasta', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1]
        #temp_str = temp_str + " " +item[0]
    texts.append(temp_str)

seq = []
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts:
    doc = re.sub(stop, '', doc)
    seq.append(doc.split())

w2v_model = Word2Vec.load('ampep30word2vecsize70win5min1bi.model')
embedding_matrix = w2v_model.wv.vectors

xtrain = []
for i in seq:
    # print(i)
    a = w2v_model.wv[i]#原来是wvz[i]显示not subscriptable wvz.wv是查看这个词的词向量
    xtrain.append(a)
word = np.array(xtrain)
X_data1=np.array(word)


XXW=X_data1[3058:,:]

YY=np.array([1] * 94 + [0] * 94, dtype='float32')
#############glove
f = open('glovedeepampep30vectorswindow3size70.txt','r')
embeddings_index={}
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()
Vector_list=[]
for i in seq:
    vector =[]
    for j in i:
        vector.append(embeddings_index[j])
    vector1 = pd.DataFrame(vector)
    Vector_list.append(vector1)
empty1 = []
for j in Vector_list:
    fla = np.array(j).flatten().tolist()
    empty1.append(fla)
Xtrain=np.array(empty1)
Xtrain1 = Xtrain.reshape((Xtrain.shape[0], 29, 70))

XXG=Xtrain1[3058:,:]

####RF
def read_feature(filepath):
    feature = pd.read_csv(filepath, header = None,index_col = False)
    feature = np.array(feature)
    x_train_feature = np.array(feature[1:, 1:len(feature[0])])
#    y_train_feature = feature[:, 0]
    return x_train_feature
def read_feature3(filepath):
    feature3 = pd.read_csv(filepath, header = None,index_col = False)
    feature3 = np.array(feature3)
    x_train_feature3 = np.array(feature3[1:,2:])
#    y_train_feature = feature[:, 0]
    return x_train_feature3

x1ptrain=read_feature('AAC-3246.csv')
x2ptrain=read_feature('DPC-3246.csv')
x3ptrain=read_feature('ABC-3246.csv')
x4ptrain=read_feature('RRI-3246.csv')
x5ptrain=read_feature('DDOR-3246.csv')
x6ptrain=read_feature3('SE-3246.csv')
x7ptrain=read_feature3('SER-3246.csv')
x8ptrain=read_feature('PAAC-3246.csv')
x9ptrain=read_feature('QSO-3246.csv')

x_sum11=np.concatenate((x1ptrain,x2ptrain,x3ptrain,x4ptrain,x5ptrain,x6ptrain,x7ptrain,x8ptrain,x9ptrain),axis=1)


X_data11=np.array(x_sum11)
XXRF=X_data11[3058:,:]

model1 = tf.keras.models.load_model('models/' + str(k) + 'model_wtv-size70-glove-size70-RF559dim--WTV_WUzhe-earlystop_deep-ampep30(1).h5')
model2 = tf.keras.models.load_model('models/' + str(k) + 'model_wtv-size70-glove-size70-RF559dim--GLOVE_WUzhe-earlystop_deep-ampep30(1).h5')
model3 = joblib.load('models/' + str(k) + 'model_wtv-size70-glove-size70-RF559dim--RF_WUzhe-earlystop_deep-ampep30(1).pkl')
model4 = joblib.load('models/' + str(k) + 'model_wtv-size70-glove-size70-RF559dim--SVM_WUzhe-earlystop_deep-ampep30(1).pkl')

pred111 = model1.predict(XXW)
pred222 = model2.predict(XXG)
# pred333 = model3.predict(x_test111)
pred333 = model3.predict_proba(XXRF)[:, 1]
pred444 = model4.predict_proba(XXRF)[:, 1]
final = []
for l in range(0, len(pred111)):
    finalpred111 = (pred111[l] + pred222[l] + pred333[l]+ pred444[l]) / 4
    final.append(finalpred111)
final1 = np.array(final)
