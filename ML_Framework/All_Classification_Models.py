#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# # Binary Classifier

# In[11]:


def Binary_Classifier(y_test, y_pred, low, mid, high):
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score, mean_absolute_percentage_error, matthews_corrcoef
    from scipy.stats import pearsonr, spearmanr
    
    # Define the target value categories based on the updated ranges
    category_ranges = [(low, mid), (mid, high)]  # Two ranges
    categories = [1, 2]  # Two categories
    
    # Classify the predictions into categories based on the updated ranges
    y_classified = []
    
    for prediction in y_pred:
        for i, (start, end) in enumerate(category_ranges):
            if start <= prediction < end:
                y_classified.append(categories[i])
                break
    
    # Define the true categories based on the target value ranges
    true_categories = []
    
    for prediction in y_test:
        for i, (start, end) in enumerate(category_ranges):
            if start <= prediction < end:
                true_categories.append(categories[i])
                break
    
    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(true_categories, y_classified)
    balanced_accuracy = balanced_accuracy_score(true_categories, y_classified)
    precision = precision_score(true_categories, y_classified, average='weighted', labels=np.unique(y_classified))
    recall = recall_score(true_categories, y_classified, average='weighted', labels=np.unique(y_classified))
    f1 = f1_score(true_categories, y_classified, average='weighted', labels=np.unique(y_classified))
    roc_auc = roc_auc_score(true_categories, y_classified)
    r2 = r2_score(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    MCC= matthews_corrcoef(true_categories, y_classified)
    PCC, _ = pearsonr(y_test, y_pred)
    SCC, _ = spearmanr(y_test, y_pred)
    
    return accuracy, balanced_accuracy, precision, recall, f1, roc_auc, r2, MAPE, MCC, PCC, SCC


# # Multi-Class Classifier

# In[12]:


def MultiClass_Classifier(y_test, y_pred, low, mid1, mid2, mid3, high):
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score, r2_score, mean_absolute_percentage_error, matthews_corrcoef
    from scipy.stats import pearsonr, spearmanr
    
    # Define the target value categories based on the updated ranges
    category_ranges = [(low, mid1), (mid1, mid2), (mid2, mid3), (mid3, high)]  # Four ranges
    categories = [1, 2, 3, 4]  # Four categories
    
    # Classify the predictions into categories based on the updated ranges
    y_classified = []
    
    for prediction in y_pred:
        for i, (start, end) in enumerate(category_ranges):
            if start <= prediction < end:
                y_classified.append(categories[i])
                break
    
    # Define the true categories based on the target value ranges
    true_categories = []
    
    for prediction in y_test:
        for i, (start, end) in enumerate(category_ranges):
            if start <= prediction < end:
                true_categories.append(categories[i])
                break
    
    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(true_categories, y_classified)
    balanced_accuracy = balanced_accuracy_score(true_categories, y_classified)
    precision = precision_score(true_categories, y_classified, average='weighted', labels=np.unique(y_classified))
    recall = recall_score(true_categories, y_classified, average='weighted', labels=np.unique(y_classified))
    f1 = f1_score(true_categories, y_classified, average='weighted', labels=np.unique(y_classified))
    r2 = r2_score(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    MCC= matthews_corrcoef(true_categories, y_classified)
    PCC, _ = pearsonr(y_test, y_pred)
    SCC, _ = spearmanr(y_test, y_pred)
    
    return accuracy, balanced_accuracy, precision, recall, f1, r2, MAPE, MCC, PCC, SCC
    


# In[3]:


# !jupyter nbconvert --to script All_Classification_Models.ipynb

