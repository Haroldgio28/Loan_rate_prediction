
import requests
# import pandas as pd
# import numpy as np


# df_test = pd.read_parquet('./data/test.parquet')
# sample = np.random.randint(0, len(df_test))
# customer = df_test.iloc[sample]
# cust = customer.to_dict()
cust = {'emp_title': 'graduate_assistant',
 'emp_length': 3.0,
 'state': 'nm',
 'homeownership': 'rent',
 'annual_income': 37000.0,
 'verified_income': 'source_verified',
 'debt_to_income': 23.86,
 'delinq_2y': 0,
 'months_since_last_delinq': 0.0,
 'earliest_credit_line': 2007,
 'inquiries_last_12m': 0,
 'total_credit_lines': 20,
 'open_credit_lines': 18,
 'total_credit_limit': 86302,
 'total_credit_utilized': 75053,
 'num_collections_last_12m': 0,
 'num_historical_failed_to_pay': 2,
 'months_since_90d_late': 0.0,
 'current_accounts_delinq': 0,
 'total_collection_amount_ever': 0,
 'current_installment_accounts': 6,
 'accounts_opened_24m': 4,
 'months_since_last_credit_inquiry': 16.0,
 'num_satisfactory_accounts': 18,
 'num_accounts_120d_past_due': 0.0,
 'num_accounts_30d_past_due': 0,
 'num_active_debit_accounts': 4,
 'total_debit_limit': 16900,
 'num_total_cc_accounts': 13,
 'num_open_cc_accounts': 12,
 'num_cc_carrying_balance': 6,
 'num_mort_accounts': 0,
 'account_never_delinq_percent': 100.0,
 'tax_liens': 2,
 'public_record_bankrupt': 0,
 'loan_purpose': 'credit_card',
 'application_type': 'individual',
 'loan_amount': 10000,
 'term': 36,
 'interest_rate': 6.72}


url = 'http://localhost:9696/predict'


def rate_pred(customer_id,cust,url):
    amount = cust['loan_amount']
    response = requests.post(url, json=cust).json()
    rate = round(response['rate'],2)
    print(f'The customer {customer_id} will receive a loan of${amount} with a rate {rate}%')



rate_pred('xyz-123',cust,url)


