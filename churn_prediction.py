
# coding: utf-8

# ## Importing dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


# ## Loading the DataSet

# In[2]:


dataframe = pd.read_csv("Churn_Modelling.csv")
dataframe.head()


# ## Dropping the columns RowNumber, CustomerId and Surname as they don't play any role in predicting 

# In[3]:


df = dataframe.drop(columns = ["RowNumber", "CustomerId", "Surname"])


# ## Seperating the Value we want to predict
# - Setting all the other attributes a X Input
# - Exited a Y because we wan't to predict if a customer will exit in the coming days or not

# ## One Hot Encoding
# - The data categorical value are no good to machine so we label it a numbers
# - But giving them numbers makes some high priority and other low without any reason
# - So we one hot encode them creating seperating columns
# - like gender_male a column and gender_female another

# In[4]:


df_X = df.drop(columns = ["Exited"])
df_Y = df["Exited"]
dummies = pd.get_dummies(df_X[['Geography', 'Gender']])
df_X = df_X.drop(columns = ['Geography', 'Gender'])
df_X = pd.concat([df_X, dummies], axis = 1)
display(df_X.head())
display(df_Y.head())


# # Standard Scaling then so that all the attributes will have equal weights at input

# In[5]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_X = scaler.fit_transform(df_X)
df_X = pd.DataFrame(df_X)


# ## Splitting them to Train and Test data
# - Test data is never shown to model except at last so that the real accuracy can be measured

# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size = 0.2, random_state = 7)


# ## Building a neural network

# In[7]:


import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Dropout


# In[8]:


classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 13))
classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))
classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))


# ## Compiling the network with optimizers and loss

# In[9]:


classifier.compile(optimizer = keras.optimizers.Adam(lr = 0.001), loss = "binary_crossentropy", metrics = ["accuracy"])


# ## Fitting the model with back propagation

# In[ ]:


classifier.fit(x = X_train, y = Y_train, epochs = 100, batch_size = 10, validation_split=0.2)


# ## Checking the model accuracy

# In[11]:


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


# In[12]:


y_pred = classifier.predict_classes(X_test)
classification_metrics(actual = Y_test, pred = y_pred, msg = "On test data")


# ## Wrapping the keras in Sklearn to do k-fold validation

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 13))
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))
    classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = keras.optimizers.Adam(lr = 0.001), loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)

display(accuracies.mean())


# In[14]:


accuracies.mean()


# ## Hyper parameter tuning through Grid Search

# In[ ]:


from sklearn.model_selection import GridSearchCV
def build_classifier(optimizers):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 13))
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))
    classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizers, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [10, 25, 32],
              'epochs': [100, 150],
              'optimizers':["adam", "rmsprop"]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# ## The best parameters obtained

# In[16]:


display(best_parameters)
display(best_accuracy)


# ## Choosing learning rate with above obtained parameters
# - Using 5 fold validation for faster computation

# In[ ]:


from sklearn.model_selection import GridSearchCV
def build_classifier(lrs):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 13))
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))
    classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = keras.optimizers.adam(lr = lrs), loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 150)
parameters = {'lrs':[0.1, 0.01, 0.001, 0.003, 0.0001, 0.0003]}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)
grid_search = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[19]:


display(best_parameters)
display(best_accuracy)


# ## Using all the parameters and doing an 10 fold validation

# In[22]:


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = 'relu', input_dim = 13))
    classifier.add(Dense(output_dim = 6, init = "uniform", activation = "relu"))
    classifier.add(Dense(output_dim = 1, init = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = keras.optimizers.adam(lr = 0.01), loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 150)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)

display(accuracies.mean())


# In[23]:


display(accuracies.mean())

