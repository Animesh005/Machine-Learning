import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import csv
from sklearn.linear_model import LogisticRegression


dataFrame = pd.read_csv("train.csv")
missing_data = np.any(np.isnan(dataFrame))

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)


if(missing_data == True):
    imr.fit(dataFrame)
    imputed_data = imr.transform(dataFrame.values)

    x_value = imputed_data.iloc[:, 1:24].values
    y_value = imputed_data.iloc[:, 25].values

else:
    x_value = dataFrame.iloc[:, 1:24].values
    y_value = dataFrame.iloc[:, 25].values

kf = KFold(n_splits=5)


for train_index, test_index in kf.split(x_value):
    x_train, x_test = x_value[train_index], x_value[test_index]
    y_train, y_test = y_value[train_index], y_value[test_index]

#y_train = y_train.reshape(-1, 1)
#y_test = y_test.reshape(-1, 1)

mms = MinMaxScaler()

x_train_norm = mms.fit_transform(x_train)
x_test_norm = mms.fit_transform(x_test)


regr = LogisticRegression()


regr.fit(x_train_norm, y_train)

test_data = pd.read_csv("test.csv")

x_test_data = test_data.iloc[:, 1:24].values
obs_nos = test_data.iloc[:, 0].values

x_test_data_norm = mms.fit_transform(x_test_data)

prediction = regr.predict(x_test_data_norm)

with open('animesh2.csv', 'w', newline='') as f:
    mywriter = csv.writer(f)

    mywriter.writerow(['Observation', 'Energy'])

    for i in range(len(prediction)):
        mywriter.writerow([obs_nos[i], prediction[i]])

