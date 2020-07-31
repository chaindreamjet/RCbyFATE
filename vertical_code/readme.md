Guest side
==
### agg_application

- process data, including padding missing value, one-hot

### agg_bureau

- process data, including padding missing value, groupby, numerical, category

### agg_bureau_balance
- process data, including padding missing value, groupby, numerical, category

Host side
==
### agg_credit_card
- process data, including padding missing value, groupby, numerical, category

### agg_installment_payment
- process data, including padding missing value, groupby, numerical, category

###  agg_POS_CASH_balance
- process data, including padding missing value, groupby, numerical, category

### agg_previous_application
- process data, including padding missing value, groupby, numerical, category

Training
==

###  feature selection
- merge data together, and using xgboost do feature selection

### vertical
- do xgboost model training

### resources of dataset
Our original data for vertical federal learning is from: https://www.kaggle.com/c/home-credit-default-risk

Also, we put the dataset after we processed at google drive, and here is the link: https://drive.google.com/file/d/1n9mMukIeUyaW57tL4C4KmDu1myZ02nfK/view?usp=sharing
