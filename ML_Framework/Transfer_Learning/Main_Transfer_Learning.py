#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import gc
import torch
gc.collect()
torch.cuda.empty_cache()

#Ramanujan1729,push

# Loading Libraries
import pandas as pd
import numpy as np
import pickle
import os
import copy
import time

from All_Featurizers import *
from All_ML_Models import *

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import KFold

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem




# In[ ]:


featurizer_map={'RDKit_NG':[RDKit_Descriptors_NG, RDKit_FPGenerator_NG, RDKit_AtomPairGenerator_NG, RDKit_MorganGenerator_NG],
                'RDKit_3DDescriptors_NG': [RDKit_3DDescriptors_NG],
                'Descriptastorus_NG': [Descriptastorus_RDKit2DNormalized_NG],
                'Pytorch_NG': [PytorchGeometric_NG],
                'DeepChem_NG': [DeepChem_MolGraphConv_NG, DeepChem_DMPNN_NG],
                'All_Featurizers_NG': [RDKit_Descriptors_NG, Descriptastorus_RDKit2DNormalized_NG, RDKit_FPGenerator_NG, RDKit_AtomPairGenerator_NG, RDKit_MorganGenerator_NG, PytorchGeometric_NG, DeepChem_MolGraphConv_NG, DeepChem_DMPNN_NG],
                
                'Pytorch_GG': [PytorchGeometric_GG],
                'DeepChem_GG': [DeepChem_MolGraphConv_GG, DeepChem_DMPNN_GG],
                'All_Featurizers_GG': [PytorchGeometric_GG, DeepChem_MolGraphConv_GG, DeepChem_DMPNN_GG]
               }

target_map={'NG':[generate_targets_NG],
            'GG': [generate_targets_GG]
           }


ml_model_map= {'All_Models_NG':[ML_Model_RF, ML_Model_ExtraTrees, ML_Model_GB],
               'ML_Model_GCN': [ML_Model_GCN],
               'ML_Model_GAT': [ML_Model_GAT],
               'All_Models_GG': [ML_Model_GCN, ML_Model_GAT],
              }


# File paths and other parameters
Nbins= 2
property= 'Activity'
target_style= 'NG'
featurizer_style= 'All_Featurizers_'+target_style
ml_model_style= 'All_Models_'+target_style


# Names of files that contain SMILES, composition, and target property 
# smiles_file = rf"C:\Users\grvkr\Box\Gaurav Kumar\Projects\SAR_NM\Codes\ML_Framework\Data_Combined\Transfer_Learning\InVivo_SMILES_All.xlsx"
smiles_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/InVivo_SMILES_All.xlsx"
smiles_sheet_name= "SMILES"

# target_file = rf"C:\Users\grvkr\Box\Gaurav Kumar\Projects\SAR_NM\Codes\ML_Framework\Data_Combined\Transfer_Learning\InVivo_{property}_All.xlsx"
target_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/InVivo_{property}_All.xlsx"
target_sheet_name= f"{Nbins} bins"


# dataX_filepath = rf"C:\Users\grvkr\Box\Gaurav Kumar\Projects\SAR_NM\Codes\ML_Framework\Data_Combined\Transfer_Learning\{property}_pickle_files\dataX_dict_all_{featurizer_style}.pkl"
# datay_filepath = rf"C:\Users\grvkr\Box\Gaurav Kumar\Projects\SAR_NM\Codes\ML_Framework\Data_Combined\Transfer_Learning\{property}_pickle_files\datay_all_{Nbins}bins_{target_style}.pkl"
dataX_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/dataX_transfer_dict_all_{featurizer_style}.pkl"
datay_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/datay_all_{Nbins}bins_{target_style}.pkl"
scaffolds_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/Scaffolds.pkl"


model_output_path = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files"
prediction_output_path = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_Predictions {Nbins} bins.xlsx"
metrics_output_path = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_Metrics {Nbins} bins.xlsx"


# In[ ]:


# Load datasets

smiles_df = pd.read_excel(smiles_filepath, sheet_name= smiles_sheet_name)
smiles_list = smiles_df.iloc[:, 0].dropna().tolist()

target_df = pd.read_excel(target_filepath, sheet_name= target_sheet_name)

Nrows, Ncolumns = smiles_df.shape
Nconstituents= (Ncolumns-6)//2


with open(dataX_filepath, 'rb') as file:
    print('Reading feature variable')
    dataX_transfer_dict_all = pickle.load(file)
    
with open(datay_filepath, 'rb') as file1:
    print('Reading target variable')
    datay_all= pickle.load(file1)


# In[ ]:


def generate_scaffold(smiles):
    """Generate Bemis-Murcko scaffold from SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else None


if os.path.exists(scaffolds_filepath):
    with open(scaffolds_filepath, 'rb') as file1:
        print('Reading scaffolds file')
        scaffolds= pickle.load(file1)
else:
    scaffolds = [generate_scaffold(sm) for sm in smiles_list]
    with open(scaffolds_filepath, 'wb') as file1:
        print('Writing scaffolds file')
        pickle.dump(scaffolds, file1)
    


# In[ ]:


def save_predictions(datay_test, datay_dict_pred, fold_idx, output_file):
    
    results_sheet_name= f'Fold {fold_idx + 1}'
    new_output_df = pd.DataFrame()
    column_name= 'y Test'
    new_output_df[column_name] = datay_test
    for ml_model_name, featurizers in datay_dict_pred.items():
        for featurizer_name, predictions in featurizers.items():
            column_name = f"{ml_model_name}_{featurizer_name}"
            new_output_df[column_name] = predictions

    if os.path.exists(output_file):
        with pd.ExcelFile(output_file, engine='openpyxl') as xls:
            if results_sheet_name in xls.sheet_names:
                existing_data_df = pd.read_excel(xls, sheet_name=results_sheet_name)
                # Concatenate existing data with new data
                combined_df = pd.concat([existing_data_df, new_output_df], axis=1)
            else:
                combined_df = new_output_df
    else:
        combined_df = new_output_df
    
    # Write the combined DataFrame to the Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=results_sheet_name, index=False)
    
    
    print(f"Finished")



def save_metrics(datay_dict_pred, datay_test, classes, fold_idx, output_file):
    rows = []
    results_sheet_name= f'Fold {fold_idx + 1}'
    for ml_model_name, featurizers in datay_dict_pred.items():
        for featurizer_name, predictions in featurizers.items():
            row = {'ML_Model': ml_model_name, 'Featurizer': featurizer_name}
            for cls in classes:
                y_true_binary = [1 if y == cls else 0 for y in datay_test]
                y_pred_binary = [1 if y == cls else 0 for y in predictions]
                row[f'Accuracy_{cls}'] = accuracy_score(y_true_binary, y_pred_binary)
                row[f'Precision_{cls}'] = precision_score(y_true_binary, y_pred_binary)
                row[f'Recall_{cls}'] = recall_score(y_true_binary, y_pred_binary)
                row[f'F1 Score_{cls}'] = f1_score(y_true_binary, y_pred_binary)
                row[f'MCC_{cls}'] = matthews_corrcoef(y_true_binary, y_pred_binary)
                row[f'AUC_{cls}'] = roc_auc_score(y_true_binary, y_pred_binary) if len(np.unique(y_true_binary)) > 1 else np.nan
            row['Kappa'] = cohen_kappa_score(datay_test, predictions)
            rows.append(row)

    results_df = pd.DataFrame(rows)
    
    if os.path.exists(output_file):
        with pd.ExcelFile(output_file, engine='openpyxl') as xls:
            if results_sheet_name in xls.sheet_names:
                existing_data_df = pd.read_excel(xls, sheet_name=results_sheet_name)
                # Concatenate existing data with new data
                combined_df = pd.concat([existing_data_df, results_df], ignore_index=True)
            else:
                combined_df = results_df
    else:
        combined_df = results_df
    
    # Write the combined DataFrame to the Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        combined_df.to_excel(writer, sheet_name=results_sheet_name, index=False)
    
    print(f"Finished")





# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(scaffolds)):
    print(f"\nFold {fold_idx + 1}")
    datay_train, datay_test = datay_all[train_idx], datay_all[test_idx]

    datay_dict_pred = {}
    for ml_model_name, featurizer_dict in dataX_transfer_dict_all.items():
        for featurizer_name, features in featurizer_dict.items():
            print(featurizer_name)
            features_train, features_test = features.iloc[train_idx], features.iloc[test_idx]
            
            start_time= time.time()
            model_fn = globals()[ml_model_name]
            print(ml_model_name)
            model, predictions = model_fn(features_train, features_test, datay_train, datay_test)
            datay_dict_pred.setdefault(ml_model_name, {})[featurizer_name] = predictions

            # Save trained model to a pickle file
            model_filename = f"{model_output_path}/Trained_Model_{Nbins} bins_{featurizer_name}_{ml_model_name}_Fold{fold_idx + 1}.pkl"
            with open(model_filename, 'wb') as model_file:
                pickle.dump(model, model_file)
            print(f"Model saved: {model_filename}")
            
            end_time= time.time()
            total_time= end_time - start_time
            print(f"Total time spent on this ML model: {total_time:.2f} seconds")


    save_predictions(datay_test, datay_dict_pred, fold_idx, prediction_output_path)
    save_metrics(datay_dict_pred, datay_test, np.unique(datay_all), fold_idx, metrics_output_path)

print("\n✅ Scaffold-Based Cross-Validation Completed and All Outputs Saved.")


# In[1]:


# !jupyter nbconvert --to script Main_Transfer_Learning.ipynb


# In[ ]:




