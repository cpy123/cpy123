import pandas as pd
import numpy as np
import sys
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import json
import traceback
import joblib

class Result:
    precision = 0
    recall = 0
    accuracy = 0
    fMeaure=[]
    rocArea=[]
params = {}
params['model'] = '/Users/dell/Desktop/rf.model'
params['test'] = 'C:/Users/dell/Desktop/out1.csv'
params['opath'] = '/Users/dell/Desktop/outt.csv'
argvs = sys.argv
try:
    for i in range(len(argvs)):
        if i < 1:
            continue
        if argvs[i].split('=')[1] == 'None':
            params[argvs[i].split('=')[0]] = None
        else:
            Type = type(params[argvs[i].split('=')[0]])
            params[argvs[i].split('=')[0]] = Type(argvs[i].split('=')[1])


    model = joblib.load(params['model'])

    test_csv = pd.read_csv(params['test'])

    test_feature = test_csv.drop(['label'], axis=1)

    y_test = test_csv['label']

    y_pred = model.predict(test_feature)

    y_pred_proba = y_pred


    res = {}
    res['precision'] = precision_score(y_test, y_pred,average='macro')
    res['recall'] = recall_score(y_test,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_test, y_pred)
    res['fMeasure'] = f1_score(y_test, y_pred,average='macro')
    res['rocArea'] = 0
    # res['featureImportances'] = clf.feature_importances_.tolist()
    print(json.dumps(res))
    predict_df = pd.DataFrame(y_pred)
    predict_df.columns = ['predict']
    predict_df.to_csv(params['opath'],index=False)
except Exception as e:
    print(e)