import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from random_forest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

loan = pd.read_csv('./data/loan.csv')

loan = loan.rename(columns={
    ' no_of_dependents': 'no_of_dependents',
    ' education': 'education',
    ' self_employed': 'self_employed',
    ' income_annum': 'income_annum',
    ' loan_amount': 'loan_amount', 
    ' loan_term': 'loan_term', 
    ' cibil_score': 'cibil_score',
    ' residential_assets_value': 'residential_assets_value', 
    ' commercial_assets_value': 'commercial_assets_value', 
    ' luxury_assets_value': 'luxury_assets_value',
    ' bank_asset_value': 'bank_asset_value',
    ' loan_status': 'loan_status'
})

loan['education'] = loan['education'].map({' Graduate': 1, ' Not Graduate': 0,'Graduate':1,'Not Graduate':0})
loan['self_employed'] = loan['self_employed'].map({' Yes': 1, ' No': 0,'Yes': 1,'No': 0})
loan['loan_status'] = loan['loan_status'].map({' Approved': 1, ' Rejected': 0,'Approved': 1, 'Rejected': 0})

np.random.seed(42)
random_indices = np.random.permutation(loan.index)
loan = loan.loc[random_indices]

##################
# Approval Model #
##################

X = loan.drop(['loan_status', 'loan_amount', 'loan_id'], axis=1).values
y = loan['loan_status'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_approval = RandomForest(n_trees=19,max_depth=15,random_state=42)
print("RF Approval model is running.")
rf_approval.fit(X_train, y_train)
approval_accuracy = accuracy_score(y_test, rf_approval.predict(X_test))
print("Approval Accuracy", approval_accuracy)
joblib.dump(rf_approval, "models/approval.joblib")

################
# Amount Model #
################
amount = loan[loan['loan_status']==1]
amount.drop(['loan_status','loan_id'], axis=1)
A = amount.drop('loan_amount', axis=1)
b = amount['loan_amount']
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2, random_state=42)
rf_amount = RandomForestRegressor(random_state=42)
print("RF Amount model is running.")
rf_amount.fit(A_train, b_train)
amount_accuracy = accuracy_score(b_test, rf_amount.predict(A_test))
print("Amount Accuracy", amount_accuracy)
joblib.dump(rf_amount, "models/amount.joblib")
