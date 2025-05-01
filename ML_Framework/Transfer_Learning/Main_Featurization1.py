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
                'All_Featurizers_NG': [RDKit_Descriptors_NG, Descriptastorus_RDKit2DNormalized_NG, RDKit_3DDescriptors_NG, RDKit_FPGenerator_NG, RDKit_AtomPairGenerator_NG, RDKit_MorganGenerator_NG, PytorchGeometric_NG, DeepChem_MolGraphConv_NG, DeepChem_DMPNN_NG],
                
                'Pytorch_GG': [PytorchGeometric_GG],
                'DeepChem_GG': [DeepChem_MolGraphConv_GG, DeepChem_DMPNN_GG],
                'All_Featurizers_GG': [PytorchGeometric_GG, DeepChem_MolGraphConv_GG, DeepChem_DMPNN_GG]
               }

target_map={'NG':[generate_targets_NG],
            'GG': [generate_targets_GG]
           }


ml_model_map= {'All_Models_NG':[ML_Model_RF, ML_Model_ExtraTrees, ML_Model_GB, ML_Model_SVM_Linear],
               'ML_Model_GCN': [ML_Model_GCN],
               'ML_Model_GAT': [ML_Model_GAT],
               'All_Models_GG': [ML_Model_GCN, ML_Model_GAT],
              }


# File paths and other parameters
Nbins= 2
property= 'Activity'
target_style= 'NG'
featurizer_style= 'All_Featurizers_'+target_style


# Names of files that contain SMILES, composition, and target property 
smiles_file = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/InVivo_SMILES_All.xlsx"
smiles_sheet_name= "SMILES"

target_file = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/InVivo_{property}_All.xlsx"
target_sheet_name= f"{Nbins} bins"


dataX_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/dataX_dict_all_{featurizer_style}.pkl"
datay_filepath = f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/datay_all_{Nbins}bins_{target_style}.pkl"


# In[ ]:


# Load datasets

smiles_df = pd.read_excel(smiles_file, sheet_name= smiles_sheet_name)

target_df = pd.read_excel(target_file, sheet_name= target_sheet_name)

Nrows, Ncolumns = smiles_df.shape

Nconstituents= (Ncolumns-6)//2

print("Finished loading data", flush=True)


# In[ ]:


start_time= time.time()

if os.path.exists(datay_filepath):
    with open(datay_filepath, 'rb') as file1:
        print('Reading target variable', flush=True)
        datay_all= pickle.load(file1)

else:
    for tt in target_map[target_style]:
        datay_all= tt(target_df, Nrows)

    with open(datay_filepath, 'wb') as file1:
        print('Writing target variable', flush=True)
        pickle.dump(datay_all, file1)
    
end_time = time.time()  # End timer
total_time = end_time - start_time
print(f"\nTotal wall time spent on this step: {total_time:.2f} seconds", flush=True)


# In[ ]:


start_time_main= time.time()

if os.path.exists(dataX_filepath):
    with open(dataX_filepath, 'rb') as file1:
        print('Reading featurized variable', flush=True)
        dataX_dict_all= pickle.load(file1)

else:
    # Perform featurization using for loop to implement multiple featurization techniques
    # Initialize dictionaries to store the output
    dataX_dict_all = {} 
    
    # Iterate over the featurizer functions based on the featurizer_style
    for ff in featurizer_map[featurizer_style]:
        start_time= time.time()
        featurizer_name = ff.__name__  # Get the featurizer function name
        print(featurizer_name, flush=True)
        # Call the function
        dataX_dict_all[featurizer_name] = ff(smiles_df, Nrows, Nconstituents)
        end_time= time.time()
        total_time= end_time - start_time
        print(f"Total wall time spent on this function: {total_time:.2f} seconds", flush=True)
    
    
    ###################   Generate Hybrid Features   ###################
    
    n_remove = Nconstituents_training + 3  # number of global feature values (e.g. 4 for composition + 3 for RNA_type, lipid_to_RNA, dosage)
    
    if featurizer_style == 'All_Featurizers_NG':
        start_time= time.time()
        print('Working on hybrid descriptors and graphs', flush=True)
        descriptorX = dataX_dict_all['RDKit_Descriptors_NG']
        descriptorX = descriptorX[:, :-n_remove]   # Remove variables already included in the graph data list
        graphX = dataX_dict_all['PytorchGeometric_NG']
        hybridX = np.concatenate((descriptorX, graphX), axis=1)
        dataX_dict_all['Hybrid_Descriptors_Graph_NG'] = hybridX
        end_time= time.time()
        total_time= end_time - start_time
        print(f"Total wall time spent on this function: {total_time:.2f} seconds", flush=True)
    
        start_time= time.time()
        print('Working on hybrid fingerprints and graphs', flush=True)
        fpX = dataX_dict_all['RDKit_FPGenerator_NG']
        fpX = fpX[:, :-n_remove]   # Remove variables already included in the graph data list
        hybridX = np.concatenate((fpX, graphX), axis=1)
        dataX_dict_all['Hybrid_Fingerprints_Graph_NG'] = hybridX
        end_time= time.time()
        total_time= end_time - start_time
        print(f"Total wall time spent on this function: {total_time:.2f} seconds", flush=True)
    
    elif featurizer_style == 'All_Featurizers_GG':
        # First load descriptors and fingerprints generated in NG featurizers
        with open(f"/home/kumar542/SAR_NM/Codes/ML_Framework/Data_Combined/Transfer_Learning/{property}_pickle_files/dataX_dict_all_All_Featurizers_NG.pkl", 'rb') as file1:
            dataX_dict_all_NG = pickle.load(file1)
        
        start_time= time.time()
        print('Working on hybrid descriptors and graphs', flush=True)
        descriptorX = torch.tensor(dataX_dict_all_NG['RDKit_Descriptors_NG'], dtype=torch.float)
        dataX_dict_all['Hybrid_Descriptors_Graph_GG']= copy.deepcopy(dataX_dict_all['PytorchGeometric_GG'])
        for idx, graph in enumerate(dataX_dict_all['Hybrid_Descriptors_Graph_GG']):
            # Each graph.global_features becomes a 2D tensor [1, hybrid_dim]
            graph.global_features = torch.clone(descriptorX[idx].unsqueeze(0)).contiguous()
            
        end_time= time.time()
        total_time= end_time - start_time
        print(f"Total wall time spent on this function: {total_time:.2f} seconds", flush=True)
    
        # --- Hybrid Fingerprints ---
        start_time= time.time()
        print('Working on hybrid fingerprints and graphs', flush=True)
        fpX = torch.tensor(dataX_dict_all_NG['RDKit_FPGenerator_NG'], dtype=torch.float)
        dataX_dict_all['Hybrid_Fingerprints_Graph_GG']= copy.deepcopy(dataX_dict_all['PytorchGeometric_GG'])
        for idx, graph in enumerate(dataX_dict_all['Hybrid_Fingerprints_Graph_GG']):
            # Each graph.global_features becomes a 2D tensor [1, hybrid_dim]
            graph.global_features = torch.clone(fpX[idx].unsqueeze(0)).contiguous()
            
        end_time= time.time()
        total_time= end_time - start_time
        print(f"Total wall time spent on this function: {total_time:.2f} seconds", flush=True)

    with open(dataX_filepath, 'wb') as file1:
        print('Writing featurized variable')
        pickle.dump(dataX_dict_all, file1)


end_time_main = time.time()  # End timer
total_time_main = end_time_main - start_time_main
print(f"\nTotal wall time spent on this step: {total_time_main:.2f} seconds", flush=True)


# In[1]:


# !jupyter nbconvert --to script Main_Featurization1.ipynb


# In[ ]:




