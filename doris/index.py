import streamlit as st
import numpy as np
import pandas as pd
import pandas as pd2
import random
from sklearn.model_selection import train_test_split

##Loading models into memeory:
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

##Loading evaluation tools into memeory:
from sklearn import model_selection
from sklearn. preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import matplotlib.pyplot as plt


header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()

with header:
    st.title('AMT Fraud Detection')
    #st.text('This project is just to toy around with somethings')

with dataset:
    st.header('Dataset')
    st.text('The dataset is customized and cannot be viewed publicly')

    atm = pd.read_csv('atm.csv')
    st.write(atm.head(5))
    scaler = MinMaxScaler()

    fraud = pd.DataFrame(atm['fraud'].value_counts()).head(50)
    st.bar_chart(fraud)
    X = atm.drop(columns='fraud', axis=1)
    Y = atm['fraud']


    #Scaling the dataset
    for col in X:
        scaler = MinMaxScaler()
        X[col] = scaler.fit_transform(X[[col]])
    #st.write(X.head(5))

    dist_col, bal_col = st.columns(2)
   
    #Balancing the training Data
    dist_col.subheader('Distribution of transaction')
    dist_col.write(atm['fraud'].value_counts())

    normal = atm[atm.fraud == 0]
    fraud = atm[atm.fraud == 1]
    normal_sample = normal.sample(n = 217)
    #st.write(normal_sample)
    new_atm = pd.concat([normal_sample, fraud], axis=0)
    new_atm['fraud'].value_counts()
    bal_col.subheader('Balancing the training dataset')
    bal_col.write(new_atm['fraud'].value_counts())
    
    X = new_atm.drop(columns='fraud', axis=1)
    Y = new_atm['fraud']


with model_training:
    st.header('Developing the training model here!')
    
    st.write('SPLITTING THE FEATURES AND LABELS INTO TRAINING AND TEST DATASETS:')
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2,stratify=Y,random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    y_pred_proba = model.predict_proba(X_test)[::,1]
    logis_fpr, logis_tpr, _ = metrics.roc_curve(Y_test,  y_pred_proba)
    logis_auc = metrics.roc_auc_score(Y_test, y_pred_proba)

    # #create ROC curve
    # fig = plt.figure()
    # plt.plot(logis_fpr,logis_tpr,label="AUC="+str(logis_auc))
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.legend(loc=4)
    # plt.show()
    # st.pyplot(fig)

    #Accuracy on training data:
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    #Accuracy on testing data:
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    select_col, display_col = st.columns(2)
   
    select_col.subheader('Training Accuracy:')
    select_col.write(training_data_accuracy)

    display_col.subheader('Testing Accuracy:')
    #display_col.write('Testing Accuracy:')
    display_col.write(test_data_accuracy)

    select_col.subheader('Confusion Matrix:')
    select_col.write(confusion_matrix(X_test_prediction, Y_test))

    


if st.button('Submit'):
    #model = build_model()
    time = random.randint(0, 8)/10
    ac = random.randint(0, 1)
    amount = random.randint(1000, 10000)
    dist = random.randint(0, 850)
    
    scaler = MinMaxScaler()
    x = [{'time': time, 'ac': ac, 'd': dist, 'amount': amount}]
    x_df = pd.DataFrame.from_dict(x)
    st.write(x_df)
    #x_df = scaler.fit_transform(x_df)
    x_df = scaler.fit_transform([[time, ac, dist, amount]])
    #st.write(x_df)
    prediction = model.predict(x_df)
    
    if prediction == 1:
        target = "Fraud"
        msg = "Fraud is suspected in this transaction"
    else:
        target = "Normal"
        msg = "This transaction is normal"

    #msg = "Transaction is " + target
    #st.write('Transaction is ', target)

    st.success(msg)

    #display_col.write(matrix)

    
