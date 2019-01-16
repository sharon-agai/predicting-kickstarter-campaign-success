
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score
import sys

# Reading in the data and improving readability
data = pd.read_csv(sys.argv[1])
data.columns = data.columns.str.replace(' ', '_')
data.country = data.country.replace(to_replace='N,0"', value='NO')

# Separating out success and failure columns - don't need all 6 states
successdata = data[data.state == 'successful']
faildata = data[data.state == 'failed']
alldata = pd.concat([successdata, faildata])

# Change launched and deadline to datetime format and create a new column called duration
alldata.launched = pd.to_datetime(alldata.launched, format='%Y-%m-%d %H:%M:%S')
alldata.deadline = pd.to_datetime(alldata.deadline, format='%Y-%m-%d %H:%M:%S')
alldata['duration']=alldata['deadline']-alldata['launched']
alldata.duration = alldata.duration.dt.days

# Drop rows with many NaNs and duplicate rows next, as well as columns we won't need.
alldata.drop_duplicates()
alldata.isnull().sum()
alldata.drop(columns=['ID', 'category', 'currency', 'deadline', 'goal', 'launched', 'name', 'pledged', 'usd_pledged'], inplace=True)
alldata.isnull().sum()

# Create the Random Forest Classification Algorithm
def RandomForest(randomState, X_train, X_test, y_train, y_test):
    global classifier
    global cm
    # Creation and fit
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Prediction
    y_pred = classifier.predict(X_test)

    # Important metrics
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)


# Creating dummy variables for categories - one-hot encoding
alldata_enc = pd.get_dummies(alldata, columns=['state', 'main_category', 'country'])

# Create a training set and a test set
X = alldata_enc.drop(['state_successful', 'state_failed'], axis='columns').values
y = alldata_enc.state_successful.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# Initialize accuracy metrics
scorer = make_scorer(accuracy_score)

# The magic! (classifier runs with 10-fold cross-validation)
RandomForest(0, X_train, X_test, y_train, y_test)
crossvalidation = np.mean(cross_val_score(classifier, X, y, cv=10))

# Extracting and viewing feature importances - which features cause the greatest increase in residuals when permuted?
feature_importances = classifier.feature_importances_

feature_importances_df = pd.DataFrame(feature_importances,
                                   index= alldata_enc.drop(['state_successful', 'state_failed'], axis='columns').columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

keyfeatures_df = feature_importances_df[:4]
keyfeaturenames = np.array(keyfeatures_df.index)

# Print statements 
print('Cross-validated performance: ', np.mean(crossvalidation))
print('Confusion matrix: ')
print(cm)
print('Most important features: ')
print(keyfeatures_df)
