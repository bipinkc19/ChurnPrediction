
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.simplefilter('ignore')

# os.chdir('C:\\Users\\97798\\Desktop\\udemy\\Churn_Prediction')

dataframe = pd.read_csv("Churn_Modelling.csv")

df = dataframe.drop(columns = ["RowNumber", "CustomerId", "Surname"])

df_X = df[df.columns.difference(['Exited'])]
df_Y = df["Exited"]
dummies = pd.get_dummies(df_X[['Geography', 'Gender']])
df_X = df.drop(columns = ['Geography', 'Gender'])
df_X = pd.concat([df_X, dummies], axis = 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_X = scaler.fit_transform(df_X)
df_X = pd.DataFrame(df_X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size = 0.2, random_state = 7)

import keras
from keras.models import Sequential 
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 14))
classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))
classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))

classifier.compile(optimizer = keras.optimizers.Adam(lr = 0.001), loss = "binary_crossentropy", metrics = ["accuracy"])

classifier.fit(x = X_train, y = Y_train, epochs = 5, validation_split=0.2)

def classification_metrics(actual, pred, msg):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(actual, pred)

    plt.figure()
    ax= plt.subplot()
    sns.heatmap(cm, annot = True, fmt = 'g')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['active', 'terminated']) 
    ax.yaxis.set_ticklabels(['active', 'terminated'])   
    plt.show()       
    sensitivity = cm[1][1]/(cm[1][0] + cm[1][1])
    specifity = cm[0][0]/(cm[0][0] + cm[0][1])
    accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)
    
    print(msg, '\n')
    print('accuracy:    ', round(accuracy,2), 
      '\nsensitivity: ', round(sensitivity,2), 
      '\nspecifity:   ', round(specifity,2))

y_pred = classifier.predict_classes(X_test)
classification_metrics(actual = Y_test, pred = y_pred, msg = "On test data")

