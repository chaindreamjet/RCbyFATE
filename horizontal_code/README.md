### feature_extract0

- process the information in user_info table
- Use One_hot Encode

### feature_extract1

- process the information in bill_detail table
- extract features

### feature_extract2

- process the information in browse_history table
- get the statistical characterization and extract features
- combie this table with before tables, and get the data with dimension 45,000 * 900

### feature_selection

- Use XGBoost model to do the feature selection
- pick top 100 features (feature importance)

### horizontal_training

- divide the data into Bank A training set, Bank B training set, and test set
- train the model in local machine, get auc in different situations


### resources of dataset
Our original data for horizontal federal learning is from: 
https://www.dcjingsai.com/v2/cmptDetail.html?id=176

Also, we put the dataset after we processed at google drive, and here is the link:
https://drive.google.com/file/d/1WqNODomhVV9xmTM7kzeqtBwSyMdQFeBj/view?usp=sharing
