import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
from model import modeling
from prediction import prediction

dataset = pd.read_csv('Tata_Global_Stock_Price_Train.csv')

class processing:
    
    def pre_processing(self):
        
        train_set = dataset.iloc[:, 1:2].values
        train_set = sc.fit_transform(train_set)
        
        X_train = []
        y_train = []
        for i in range(60, 1199):
            X_train.append(train_set[i-60:i, 0])
            y_train.append(train_set[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return X_train, y_train


obj = processing()
X_train, y_train = obj.pre_processing()
model_obj = modeling()
reg = model_obj.model(X_train, y_train)
pred_obj = prediction()
pred_obj.pred(dataset, sc, reg)



