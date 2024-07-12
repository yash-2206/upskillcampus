#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.impute import SimpleImputer


# In[2]:


df1 = pd.read_csv('datafile (1).csv')
df2 = pd.read_csv('datafile (2).csv')
df3 = pd.read_csv('datafile (3).csv')
df4 = pd.read_csv('datafile.csv')


# In[3]:


df2.columns = df2.columns.str.strip()
df4.columns = df4.columns.str.strip()


# In[4]:


df1['Crop'] = df1['Crop'].str.strip().str.lower()
df2['Crop'] = df2['Crop'].str.strip().str.lower()
df3['Crop'] = df3['Crop'].str.strip().str.lower()
df4['Crop'] = df4['Crop'].str.strip().str.lower()


# In[5]:


merged_df = df1.merge(df2, on='Crop', how='inner')
merged_df = merged_df.merge(df3, on='Crop', how='inner')
merged_df = merged_df.merge(df4, on='Crop', how='inner')


# In[6]:


print(merged_df.head())


# In[7]:


print(merged_df.columns)


# In[8]:


# Handle missing values
# Example: Fill missing values with the mean of the column
merged_df.fillna(merged_df.mean(), inplace=True)


# In[9]:


def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.where((data < lower_bound) | (data > upper_bound))


# In[10]:


# Detect outliers in numerical columns and remove them
for col in merged_df.select_dtypes(include=[np.number]).columns:
    outliers_indices = detect_outliers_iqr(merged_df[col])
    merged_df = merged_df.drop(outliers_indices[0])


# In[11]:


# Convert categorical variables
# Identify categorical columns
categorical_cols = merged_df.select_dtypes(include=['object']).columns


# In[12]:


# Assume 'Yield (Quintal/ Hectare)' is the target variable
X = merged_df.drop('Yield (Quintal/ Hectare) ', axis=1)
y = merged_df['Yield (Quintal/ Hectare) ']


# In[13]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', SimpleImputer(strategy='mean'), X.select_dtypes(include=[np.number]).columns)
    ],
    remainder='passthrough'
)


# In[14]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# Create a preprocessing and training pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler())
])


# In[16]:


# Preprocess and scale the training and testing data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)


# In[17]:


model = GradientBoostingRegressor()


# In[18]:


model.fit(X_train_processed, y_train)


# In[19]:


y_pred = model.predict(X_test_processed)


# In[20]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[21]:


results = {
    'Gradient Boosting Regressor': {
        'MAE': mae,
        'MSE': mse,
        'R2': r2
    }
}


# In[22]:


results

