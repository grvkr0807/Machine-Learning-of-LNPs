#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading Libraries
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import os

from All_Featurizers import *
from All_ML_Models import *

from sklearn.model_selection import KFold

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem

import copy
import torch
import time


print("Finished loading libraries", flush=True)


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
dataX_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/dataX_dict_all_{featurizer_style}.pkl"
datay_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/datay_all_{Nbins}bins_{target_style}.pkl"
# scaffolds_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/Scaffolds.pkl"

base_model_path = rf"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/5Fold_CV/{property}_pickle_files"

dataX_transfer_filepath= f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/dataX_transfer_dict_all_{featurizer_style}.pkl"
pred_probs_filepath= f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/pred_probs_dict_all_{featurizer_style}.pkl"



# In[ ]:


# Load datasets

smiles_df = pd.read_excel(smiles_filepath, sheet_name= smiles_sheet_name)
smiles_list = smiles_df.iloc[:, 0].dropna().tolist()
size= smiles_df['size']
pdi= smiles_df['pdi']
zeta= smiles_df['zeta']

target_df = pd.read_excel(target_filepath, sheet_name= target_sheet_name)

Nrows, Ncolumns = smiles_df.shape
Nconstituents= (Ncolumns-6)//2


with open(dataX_filepath, 'rb') as file:
    print('Reading feature variable')
    dataX_dict_all = pickle.load(file)
    
with open(datay_filepath, 'rb') as file1:
    print('Reading target variable')
    datay_all= pickle.load(file1)


# In[ ]:


invivo_pred_probs = {}
dataX_transfer_dict_all= {}
for featurizer_name, features in dataX_dict_all.items():
    print(featurizer_name)
    for model_fn in ml_model_map[ml_model_style]:
        start_time= time.time()
        
        ml_model_name= model_fn.__name__
        print(ml_model_name)
        fold_preds= []
        for fold_idx in range(0,5):
            base_model_file= f"{base_model_path}/Trained_Model_{Nbins} bins_{featurizer_name}_{ml_model_name}_Fold{fold_idx + 1}.pkl"
            with open(base_model_file, 'rb') as f:
                base_model = pickle.load(f)
            probs= base_model.predict_proba(features)[:, 1]  # predicted probabilities for class 1
            fold_preds.append(probs)
        fold_avg_pred_probs = np.mean(fold_preds, axis=0) # average the infomration learned over 5 folds    
        invivo_pred_probs.setdefault(ml_model_name, {})[featurizer_name] = fold_avg_pred_probs
        
        X_transfer = pd.DataFrame({
            'Model_Prediction': fold_avg_pred_probs,
            'Size': size,
            'PDI': pdi,
            'Zeta_Potential': zeta
        })
        dataX_transfer_dict_all.setdefault(ml_model_name, {})[featurizer_name] = X_transfer
            
        end_time= time.time()
        total_time= end_time - start_time
        print(f"Total time spent on this ML model: {total_time:.2f} seconds")


with open(dataX_transfer_filepath, 'wb') as file1:
    print('Writing transfer featurized variable')
    pickle.dump(dataX_transfer_dict_all, file1)

with open(pred_probs_filepath, 'wb') as file1:
    print('Writing in vivo predicted probabilities')
    pickle.dump(invivo_pred_probs, file1)





# In[ ]:


# !jupyter nbconvert --to script Main_Featurization_Transfer.ipynb

