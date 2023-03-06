# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:51:15 2021

@author: kavan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
path = r"C:\Users\kavan\OneDrive\Documents\The Project\databackup1.csv"
df= pd.read_csv(path)
dum_df = df['DOTW']
dum_df1 = df['Month']

dum_df = pd.get_dummies(df, columns=['DOTW', 'Month'])


#enc = OneHotEncoder(handle_unknown='ignore')
#enc_df = pd.DataFrame(enc.fit_transform(df[['DOTW']]).toarray())
#df= df.join(enc_df)

path1 = r"C:\Users\kavan\OneDrive\Documents\The Project\possibledata6.csv"
dum_df.to_csv(path1)

X = dum_df[['DOTW_Friday', 'DOTW_Monday','DOTW_Saturday', 'DOTW_Sunday', 'DOTW_Thursday',	'DOTW_Tuesday',	'DOTW_Wednesday', 'Month_April','Month_August',	'Month_December','Month_February', 'Month_January',	'Month_July', 'Month_June',	'Month_March', 'Month_May',	'Month_November','Month_October','Month_September', 'GITP', 'KIL', 'Federal-Holiday']]
y = dum_df[['Tickets']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
degree=1
polyreg_scaled=make_pipeline(PolynomialFeatures(degree),scaler,LinearRegression())
polyreg_scaled.fit(X_train, y_train)
y_pred = polyreg_scaled.predict(X_test)

new_input = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

def getDOTW():
    
    day = input("What is the day of the week of the day you want to predict?")
    if day == "Sunday":
        np.put(new_input, 3, 1)
    elif day == "Monday":
        np.put(new_input, 1, 1)
    elif day == "Tuesday":
        np.put(new_input, 5, 1)
    elif day == "Wednesday":
        np.put(new_input, 6, 1)
    elif day == "Thursday":
        np.put(new_input, 4, 1)
    elif day == "Friday":
        np.put(new_input, 0, 1)
    elif day == "Saturday":
        np.put(new_input, 2, 1)
    else:
        print("You didn't enter a proper DOTW, please check you capitalization and spelling and try again.")
        getDOTW()

def getMonth():
    month = input("What month are you predicting?")
    if month == "January":
        np.put(new_input, 11, 1)
    elif month == "February":
        np.put(new_input, 10, 1)
    elif month == "March":
        np.put(new_input, 14, 1)
    elif month == "April":
        np.put(new_input, 7, 1)
    elif month == "May":
        np.put(new_input, 15, 1)
    elif month == "June":
        np.put(new_input, 13, 1)
    elif month == "July":
        np.put(new_input, 12, 1)
    elif month == "August":
        np.put(new_input, 8, 1)
    elif month == "September":
        np.put(new_input, 18, 1)
    elif month == "October":
        np.put(new_input, 17, 1)
    elif month == "November":
        np.put(new_input, 16, 1)
    elif month == "December":
        np.put(new_input, 9, 1)
    else:
        print("Please check spelling and capitalization and try again.")
        getMonth()

def glow():
    glow = input("Is the day a Glow in the Park?")
    if glow == "Yes":
        np.put(new_input, 19, 1)

def kil():
    kil = input("Is the day a Keep it Lit?")
    if kil == "Yes":
        np.put(new_input, 20, 1)

def fed():
    fed = input("Is the day a federal holiday?")
    if fed == "Yes":
        np.put(new_input, 21, 1)

def prediction():
    #numpy.put(a, ind, v, mode='raise')
    # get prediction for new input
    newnew_input = np.reshape(new_input, (-1, 22))
    new_output = polyreg_scaled.predict(newnew_input)
    # summarize input and output
    print(new_input, new_output)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print("The predicted number of climbers is ", new_output)
    print("Prediction complete. Restart the program to make another prediction.")
def main():
    getDOTW()
    getMonth()
    glow()
    kil()
    fed()
    prediction()
main()