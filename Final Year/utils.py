import numpy as np
from joblib import load

approval_model = load("./models/approval.joblib")
amount_model = load("./models/amount.joblib")

def make_predictions(
    no_of_dependents, 
    education, 
    self_employed, 
    income_annum,
    loan_term, 
    cibil_score, 
    residential_assets_value,
    commercial_assets_value, 
    luxury_assets_value, 
    bank_asset_value
):
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    user_data = np.array([
        no_of_dependents, 
        education, 
        self_employed, 
        income_annum,
        loan_term, 
        cibil_score, 
        residential_assets_value,
        commercial_assets_value, 
        luxury_assets_value, 
        bank_asset_value
    ]).reshape(1, -1)

    prediction = approval_model.predict(user_data)[0]

    if prediction == 1:
        return f"Loan is Approved. Predicted Loan Amount = Nprs {round(amount_model.predict(user_data)[0])}"
    return "Loan is Rejected."