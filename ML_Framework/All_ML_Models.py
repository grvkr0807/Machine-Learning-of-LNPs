#!/usr/bin/env python
# coding: utf-8

# # ML Models

# In[4]:


import numpy as np

# Random Forest Regression
def ML_Model_RF_Regression(X, X_train, X_test, y_train, y_test, num_estimators, num_neighbors, num_random):
    from sklearn.ensemble import RandomForestRegressor
    
    # Create and train a Random Forest Regression model
    model = RandomForestRegressor(n_estimators= num_estimators, random_state= num_random)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_full= model.predict(X)
    
    return y_pred, y_pred_full



# Random Forest Classification
def ML_Model_RF_Classification(X, X_train, X_test, y_train, y_test, num_estimators, num_neighbors, num_random):
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train a Random Forest Regression model
    model = RandomForestClassifier(n_estimators= num_estimators, random_state= num_random)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_full= model.predict(X)
    
    return y_pred, y_pred_full



# SVM Classification
def ML_Model_SVM_Classification(X, X_train, X_test, y_train, y_test, num_estimators, num_neighbors, num_random):
    from sklearn.svm import SVC
    
    # Create and train a Random Forest Regression model
    model = SVC(kernel='linear', random_state= num_random)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_full= model.predict(X)
    
    return y_pred, y_pred_full



# kNN Classification
def ML_Model_kNN_Classification(X, X_train, X_test, y_train, y_test, num_estimators, num_neighbors, num_random):
    from sklearn.neighbors import KNeighborsClassifier
    
    # Create and train a Random Forest Regression model
    model = KNeighborsClassifier(n_neighbors= num_neighbors)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_full= model.predict(X)
    
    return y_pred, y_pred_full



# Gradient Boosting Classification
def ML_Model_GB_Classification(X, X_train, X_test, y_train, y_test, num_estimators, num_neighbors, num_random):
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Create and train a Random Forest Regression model
    model = GradientBoostingClassifier(n_estimators= num_estimators, random_state= num_random)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_full= model.predict(X)
    
    return y_pred, y_pred_full



# Logistic Regression Classification
def ML_Model_LR_Classification(X, X_train, X_test, y_train, y_test, num_estimators, num_neighbors, num_random):
    from sklearn.linear_model import LogisticRegression
    
    # Create and train a Random Forest Regression model
    model = LogisticRegression(max_iter= num_estimators, random_state= num_random)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    y_pred_full= model.predict(X)
    
    return y_pred, y_pred_full



