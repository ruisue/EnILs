import matplotlib.pyplot as plt
import pandas as pd
import joblib
from itertools import chain
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
from tensorflow.keras.models import *
import numpy as np

b=pd.read_csv('pos73neg598s.csv',header=None)


pos_neg=np.array(b)
XX=pos_neg
YY=np.array([1] * 73+[0] * 598)
k=5
#####load model
model1 = joblib.load( 'models/' +'xgb_average.pkl')
model2 = joblib.load( 'models/' +'rf_average.pkl')
model3 = load_model('models/'+'model_average.h5')

X_data11=XX.reshape((XX.shape[0],XX.shape[1],1))

pred111=model1.predict_proba(XX)[:,1]
pred222=model2.predict_proba(XX)[:,1]
pred333=model3.predict(X_data11)
final=[]
for l in  range(0,len(pred111)):
    finalpred111 = (pred111[l] + pred222[l] + pred333[l]) / 3
    final.append(finalpred111)
final1=np.array(final)
c1 = list(chain(*final1))

def Twoclassfy_evalu(Y_test, y_predict):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []
    for i in range(len(Y_test)):
        if y_predict[i] > 0.5 and Y_test[i] == 1:
            TP += 1
        if y_predict[i] > 0.5 and Y_test[i] == 0:
            FP += 1
            FP_index.append(i)
        if y_predict[i] < 0.5 and Y_test[i] == 1:
            FN += 1
            FN_index.append(i)
        if y_predict[i] < 0.5 and Y_test[i] == 0:
            TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    # MCC = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    Acc = (TP + TN) / (TP + FP + TN + FN)
    fpr, tpr, thresholds = roc_curve(Y_test, y_predict)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05]) #####plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])#####plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    print('Sn', Sn)
    print('Sp', Sp)
    print('ACC', Acc)
    print('roc_auc', roc_auc)
    return Sn, Sp, Acc


if __name__ == '__main__':
    Twoclassfy_evalu(YY, c1)
