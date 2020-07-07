import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import os
print(os.listdir("../input")) 

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('logreg', LogisticRegression())]

# Read input data using pandas read_csv function
data = pd.read_csv("../input/train.csv")
df = pd.read_csv("../input/test.csv")

X = data.drop(['PAID_NEXT_MONTH'], axis = 1)
y = data['PAID_NEXT_MONTH'].values
Xt = df.drop(['PAID_NEXT_MONTH'], axis = 1)

# Use scikit-learn's train_test_split package to split the train data among 
# training and testing with 70% and 30% data for each respectively.
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3, random_state = 42)
# Input data files are available in the "../input/" directory.

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Fit pipeline to training data
pipeline.fit(X_train, y_train)

# use trained pipeline to predict customer bill payment
y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))

y_pred_test = pipeline.predict(Xt)

# create dataframe for submission to challenge
d = {'ID': df['ID'].values, 'PAID_NEXT_MONTH' : y_pred_test}
df_final = pd.DataFrame(data = d)

# debug check
print(df_final.head())

# submission to challenge
df_final.to_csv('submission.csv')