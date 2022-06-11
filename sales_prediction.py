import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.header('Future Sales Prediction')
df_columns = st.sidebar.multiselect('Pick columns [These Columns will be used for Training]', ["TV", "Radio", "Newspaper", "Sales"])
ml_model_select = st.sidebar.selectbox('Pick a Regression model', ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Decision Tree', 'Random forest'])
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
data1 = data[data.columns if df_columns == [] else df_columns]
st.text("Advertising Data")
df_head = st.slider('No of Rows', 0, 199)
st.dataframe(data1.head(df_head))
fig_select = st.selectbox('Pick a column to plot against sales', ["TV", "Radio", "Newspaper"])
fig = plt.figure(figsize=(10, 4))
sns.scatterplot(data=data, x="Sales", y="TV" if fig_select == [] else fig_select)
st.pyplot(fig)
if 'Sales' in data1.columns:
    data1.drop(['Sales'], axis = 1, inplace = True) 
if data1.size == 0:
    data1 = data[["TV", "Radio", "Newspaper"]]
X = data1
Y = data['Sales']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
if ml_model_select == 'Linear Regression':
    reg = LinearRegression().fit(x_train, y_train)
    st.subheader(f"Accuracy : {reg.score(x_test, y_test) * 100} %")
if ml_model_select == 'Ridge Regression':
    alpha = st.slider('Hyperparameter : Alpha', 0.01, 0.5)
    ridgeReg = Ridge(alpha=alpha, normalize=True)
    ridgeReg.fit(x_train,y_train)
    y_pred = ridgeReg.predict(x_test)
    st.subheader(f"Accuracy : {ridgeReg.score(x_test, y_test) * 100} %")
if ml_model_select == 'Lasso Regression':
    alpha = st.slider('Hyperparameter : Alpha', 0.01, 0.5)
    lassoReg = Lasso(alpha=alpha, normalize=True)
    lassoReg.fit(x_train,y_train)
    pred = lassoReg.predict(x_test)
    st.subheader(f"Accuracy : {lassoReg.score(x_test, y_test) * 100} %")
if ml_model_select == 'Decision Tree':
    reg = DecisionTreeRegressor()
    reg.fit(x_train,y_train)
    pred = reg.predict(x_test)
    st.subheader(f"Accuracy : {reg.score(x_test, y_test) * 100} %")
if ml_model_select == 'Random forest':
    estimators = st.slider('Hyperparameter : n_estimators', 10, 1000)
    reg = RandomForestRegressor(n_estimators = estimators)
    reg.fit(x_train,y_train)
    pred = reg.predict(x_test)
    st.subheader(f"Accuracy : {reg.score(x_test, y_test) * 100} %")
# st.stop()
