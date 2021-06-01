from sklearn.ensemble import IsolationForest
import csv
import sys
import numpy as np
from scipy import stats
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 特征预处理----------异常值检测

params = {}
params['n_estimators'] = 100
params['max_samples'] ='auto'
params['contamination'] = 0.1
params['max_features'] = 1.0
params['path'] = 'C:/Users/dell/Desktop/paderborn-train/traindata_N15_M07_F10.csv'
params['opath'] = 'C:/Users/dell/Desktop/out.csv'
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
    with open(params['path'], 'r') as f:
    #1.创建阅读器对象
        reader = csv.reader(f)
    #2.读取文件第一行数据
        head_row=next(reader)
    data_attribute = []
    for item in head_row:
        data_attribute.append(item)

 #读取数据并删除最后一列标签
    tn = pd.read_csv(params['path'])
    tn.dropna(inplace=True)
    train = np.array(tn)
    train_x = train[:, :-1]

#存标签
    train_y = train[:,-1]
    train_y = np.array(train_y)

#对所有数据行进行异常检测
    train_x = np.array(train_x)
    clf = IsolationForest(n_estimators=params['n_estimators'],
                      max_samples=params['max_samples'],
                      contamination=params['contamination'],
                      max_features=params['max_features'],
                      bootstrap=False, n_jobs=1, random_state=None,
                      verbose=0).fit(train_x)
#pred存入的是每一行数据的预测值，是1或者-1
    pred = clf.predict(train_x)
    normal = train_x[pred == 1]
    abnormal = train_x[pred == -1]

    print (pred.size)
#删除pred为-1的行数据
    df = pd.DataFrame(pd.read_csv(params['path']))[0:pred.
         size]
    df['pred']=pred

#data2=data1[-data1.sorce.isin([61])]
    df2 = df[-df.pred.isin([-1])]
    df2 = df2.drop(['pred'],axis=1)

#将清洗之后的数据存入csv文件
    data_out = df2.iloc[:,:].values
    csvfile2 = open(params['opath'],'w',newline='')
    writer = csv.writer(csvfile2)
    writer.writerow(data_attribute)   #存属性
    m = len(data_out)
    print(m)
    for i in range(m):
        writer.writerow(data_out[i])
except Exception as e:
    print(e)

df_out_columns = ['time_mean','time_std','time_max','time_min','time_rms',
                  'time_ptp','time_median','time_iqr','time_pr',
                  'time_skew','time_kurtosis','time_var','time_amp',
                  'time_smr','time_wavefactor','time_peakfactor',
                  'time_pulse','time_margin','1X','2X','3X','1XRatio',
                  '2XRatio','3XRatio']
label_columns = ['label']
full_columns = df_out_columns + label_columns

# 计算特征值
def featureget(df_line):
    #提取时域特征
    time_mean = df_line.mean()
    time_std = df_line.std()
    time_max = df_line.max()
    time_min = df_line.min()
    time_rms = np.sqrt(np.square(df_line).mean().astype(np.float64))
    time_ptp = np.asarray(df_line).ptp()
    time_median = np.median(df_line)
    time_iqr = np.percentile(df_line,75)-np.percentile(df_line,25)
    time_pr = np.percentile(df_line,90)-np.percentile(df_line,10)
    time_skew = stats.skew(df_line)
    time_kurtosis = stats.kurtosis(df_line)
    time_var = np.var(df_line)
    time_amp = np.abs(df_line).mean()
    time_smr = np.square(np.sqrt(np.abs(df_line).astype(np.float64)).mean())
#下面四个特征需要注意分母为0或接近0问题，可能会发生报错
    time_wavefactor = time_rms/time_amp
    time_peakfactor = time_max/time_rms
    time_pulse = time_max/time_amp
    time_margin = time_max/time_smr
#提取频域特征倍频能量以及能量占比
    plist_raw = np.fft.fft(list(df_line), n=1024)
    plist = np.abs(plist_raw)
    plist_energy = (np.square(plist)).sum()
#在傅里叶变换结果中，在32点处的幅值为一倍频幅值，64点处幅值为二倍频幅值，96点处为三倍频幅值，因此提取这三处幅值并计算能量占比
    return_list = [
    time_mean,time_std,time_max,time_min,time_rms,time_ptp,
    time_median,time_iqr,time_pr,time_skew,time_kurtosis,
    time_var,time_amp,time_smr,time_wavefactor,time_peakfactor,
    time_pulse,time_margin,plist[32], plist[64], plist[96],
    np.square(plist[32]) / plist_energy,
    np.square(plist[64]) / plist_energy,
    np.square(plist[96]) / plist_energy
    ]
    return return_list



# 写文件
params = {}
params['path'] = 'C:/Users/dell/Desktop/out.csv'
params['opath'] = 'C:/Users/dell/Desktop/out1.csv'
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
    df_normal= pd.read_csv(params['path'])
    feature_normal = []
    for i in range(0,df_normal.shape[0]):
        feature_line = []
        #对整合后数据每个传感器采集的波形循环提取特征
        feature_line.extend(featureget(df_normal.iloc[i,1:2560]))
        feature_line.append(df_normal['label'][i])
        feature_normal.append(feature_line)
#输出到特定文件中
    feature_normal = pd.DataFrame(feature_normal, columns=full_columns)
    feature_normal.to_csv(params['opath'], index=False)
except Exception as e:
    print(e)