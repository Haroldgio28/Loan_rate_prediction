import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import RandomizedSearchCV
import pickle


#Load Data
df = pd.read_csv('./data/raw.csv')
df = df[['emp_title','emp_length','state','homeownership','annual_income','verified_income','debt_to_income','annual_income_joint','verification_income_joint','debt_to_income_joint','delinq_2y','months_since_last_delinq','earliest_credit_line','inquiries_last_12m','total_credit_lines','open_credit_lines','total_credit_limit','total_credit_utilized','num_collections_last_12m','num_historical_failed_to_pay','months_since_90d_late','current_accounts_delinq','total_collection_amount_ever','current_installment_accounts','accounts_opened_24m','months_since_last_credit_inquiry','num_satisfactory_accounts','num_accounts_120d_past_due','num_accounts_30d_past_due','num_active_debit_accounts','total_debit_limit','num_total_cc_accounts','num_open_cc_accounts','num_cc_carrying_balance','num_mort_accounts','account_never_delinq_percent','tax_liens','public_record_bankrupt','loan_purpose','application_type','loan_amount','term','interest_rate']]

features = ['emp_title','emp_length','state','homeownership','annual_income','verified_income','debt_to_income','annual_income_joint','verification_income_joint','debt_to_income_joint','delinq_2y','months_since_last_delinq','earliest_credit_line','inquiries_last_12m','total_credit_lines','open_credit_lines','total_credit_limit','total_credit_utilized','num_collections_last_12m','num_historical_failed_to_pay','months_since_90d_late','current_accounts_delinq','total_collection_amount_ever','current_installment_accounts','accounts_opened_24m','months_since_last_credit_inquiry','num_satisfactory_accounts','num_accounts_120d_past_due','num_accounts_30d_past_due','num_active_debit_accounts','total_debit_limit','num_total_cc_accounts','num_open_cc_accounts','num_cc_carrying_balance','num_mort_accounts','account_never_delinq_percent','tax_liens','public_record_bankrupt','loan_purpose','application_type','loan_amount','term']
target = 'interest_rate'

#Data Preparation

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.strip().str.replace(' ', '_')

# Delete with too many missing values and no prediction power
del df['verification_income_joint']
del df['annual_income_joint']
del df['debt_to_income_joint']

# Filling missing values with 0
df['months_since_last_delinq'] = df['months_since_last_delinq'].fillna(0)
df['months_since_90d_late'] = df['months_since_90d_late'].fillna(0)
df['months_since_last_credit_inquiry'] = df['months_since_last_credit_inquiry'].fillna(0)
df['num_accounts_120d_past_due'] = df['num_accounts_120d_past_due'].fillna(0)
df['emp_title'] = df['emp_title'].fillna('missing')
df['emp_length'] = df['emp_length'].fillna(0)
df['debt_to_income'] = df['debt_to_income'].fillna(0)


# Splitting data into train and test

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train[target].values
del df_full_train[target]

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train[target].values
y_val = df_val[target].values
y_test = df_test[target].values

del df_train[target]
del df_val[target]
del df_test[target]

train_dicts = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

# Best parameters resulting of tunning

boost_rounds = 9
params = {'objective': 'reg:squarederror', 'max_depth': 5, 'eta': 0.3}


dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

best_model = xgb.train(
    params,
    dtrain,
    num_boost_round=boost_rounds
)


print('-------------------------------------')
# Evaluar el modelo
y_pred = best_model.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE on the test set: {rmse}")

# Evaluar el modelo
y_pred = best_model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"RMSE on the validation set: {rmse}")


# Save model 
output_file = 'model1.0.bin'

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv,model), f_out)
print('\n')
print(f'The model is saved to {output_file}')

