import joblib
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, BatchNormalization
from keras.layers import Flatten, Dense, Embedding, Convolution1D, Dropout, Activation, MaxPooling1D
from Bio import SeqIO
import pandas as pd
from scipy import interp
import matplotlib.pyplot as plt
from nltk import trigrams, bigrams
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from keras.layers import Bidirectional, LSTM, BatchNormalization
import re
import tensorflow as tf
import numpy as np
from itertools import chain
from keras import backend as K
from sklearn.metrics import roc_curve, precision_recall_curve
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.models import Model
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from itertools import chain
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split

a=pd.read_csv('E:/run/il-6/SMOTE/smo100_p292xuwt5_3bi.csv')
b=pd.read_csv('E:/run/il-6/pos292neg2393s_xulie+wtv(win5min3bi).csv',header=None)
pos=np.array(a)
pos_neg=np.array(b)
smo=pd.read_csv('SMOTE/smo300.csv')        #过采样的样本
train_pos=pd.read_csv('TRAIN_POS.csv')    #训练集的292个正样本

neg1=pos_neg[2100:2685,:]
XX=np.concatenate((train_pos,neg1))
YY=np.array([1] * 292+[0] * 585)

kf = StratifiedKFold(n_splits=5 ,shuffle=True, random_state=111)
empty=[]
cvSn=[]
cvSp=[]
cvAcc=[]
cvauc=[]
SN = []
SP = []
ACC = []
AUC = []
tprs = []
aucs1 = []
mean_fpr = np.linspace(0, 1, 100)

##########三个分类器集成
n = 0
k=1
for train_index, test_index in kf.split(XX,YY):#分层交叉验证，正集分k份，负集分k份
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train1, x_test1 = XX[train_index], XX[test_index]
    y_train1, y_test1 = YY[train_index], YY[test_index]
    x_sum12 = np.concatenate((smo, x_train1), axis=0)
    y1 = np.array([1] * 300)
    label = np.concatenate((y1, y_train1), axis=0)
    print(x_train1.shape)
    print(x_test1.shape)
    x_train11 = x_sum12.reshape((x_sum12.shape[0], x_sum12.shape[1], 1))
    x_test11 = x_test1.reshape((x_test1.shape[0], x_test1.shape[1], 1))
    model1 = XGBClassifier(colsample_bytree=0.99682,gamma=6.37535,
                           learning_rate=0.07313,max_depth=7,
                           min_child_weight=9.03839,n_estimators=125,subsample=0.91606)  # random_state=1,learning_rate=0.01     n_estimators=228,max_depth=12
    model2 = RandomForestClassifier(n_estimators=206,max_depth=7,max_features=0.1,min_samples_split=18)  ##n_estimators=220,max_depth=15

    model1.fit(x_sum12, label)
    model2.fit(x_sum12, label)



    model = Sequential()
    model.add(Convolution1D(filters=59, kernel_size=6, strides=1, padding='valid',
                            activation='relu', input_shape=(1625, 1)))    ###未用kernel_initializer='he_normal',
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Convolution1D(filters=55, kernel_size=4, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(60, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(73, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(92, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train11, label, epochs=100, validation_data=(x_test11, y_test1), verbose=1, shuffle=True)

    model.save(str(k)+'model_average.h5')
    joblib.dump(model1, str(k)+'xgb_average.pkl')
    joblib.dump(model2, str(k)+'rf_average.pkl')
    k=k+1

    pred1=model1.predict(x_test1)
    pred2=model2.predict(x_test1)
    pred3=model.predict_classes(x_test11)

    pred111 = model1.predict_proba(x_test1)[:, 1]
    pred222 = model2.predict_proba(x_test1)[:, 1]
    pred333 = model.predict(x_test11)
    final = []
    for l in range(0, len(pred111)):
        finalpred111 = (pred111[l] + pred222[l] + pred333[l]) / 3
        final.append(finalpred111)
    final1 = np.array(final)
    c1 = list(chain(*final1))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []

    for i in range(len(y_test1)):
        if final1[i] > 0.5 and y_test1[i] == 1:
            TP += 1
        if final1[i] > 0.5 and y_test1[i] == 0:
            FP += 1
            FP_index.append(i)
        if final1[i] < 0.5 and y_test1[i] == 1:
            FN += 1
            FN_index.append(i)
        if final1[i] < 0.5 and y_test1[i] == 0:
            TN += 1
    Sn1 = TP / (TP + FN)
    Sp1 = TN / (FP + TN)
    Acc1 = (TP + TN) / (TP + FP + TN + FN)
    fpr, tpr, thresholds = roc_curve(y_test1, final1)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs1.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (n, roc_auc))
    n += 1
    SN.append(Sn1 * 100)
    SP.append(Sp1 * 100)
    ACC.append(Acc1 * 100)
    AUC.append(aucs1)

print('SN', SN)
print('SP', SP)
print('ACC', ACC)
print('AUC', AUC)

meanSN = np.mean(SN)
meanSP = np.mean(SP)
meanACC = np.mean(ACC)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
meanAUC = auc(mean_fpr, mean_tpr)  # 计算平均AUC值

print("meanSN", meanSN)
print("meanSP", meanSP)
print("meanACC", meanACC)
print("meanAUC", meanAUC)

