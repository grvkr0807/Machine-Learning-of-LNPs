#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator, AllChem, Descriptors3D
from descriptastorus.descriptors import rdNormalizedDescriptors

import torch
from torch_geometric.utils import from_smiles
from torch_geometric.data import Data, DataLoader

import deepchem as dc


# # First set of all featurizers for non-graph based ML models (*_NG such as RF, SVM, XGBoost)

# # Function to generate target variables

# In[ ]:


def generate_targets_NG(target_df, Nrows):
    y= []
    for i in range(Nrows):
        target = target_df.iloc[i, 0]
        y.append(target)

    return np.array(y)
    


# # RDKit Descriptors and Fingerprints

# In[ ]:


# RDKit Descriptor Featurizer

def RDKit_Descriptors_NG(smiles_df, Nrows, Nconstituents):

    X = []
    ipc_index = [42]
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            composition.append(amount)

            RDLogger.DisableLog('rdApp.error')   # Suppress RDKit errors
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog('rdApp.error')  # Re-enable errors after parsing
            
            if mol is not None:
                features = np.array([value for value in Descriptors.CalcMolDescriptors(mol).values()])
                features = np.delete(features, ipc_index)
            else:
                features= np.zeros(209)

            features_full.extend(features)
        
        # Flatten all features into a single list and append metadata
        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)
    
    return np.array(X)




def RDKit_FPGenerator_NG(smiles_df, Nrows, Nconstituents):

    fp_size= 1024
    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)

    X = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            composition.append(amount)
            
            RDLogger.DisableLog('rdApp.error')   # Suppress RDKit errors
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog('rdApp.error')  # Re-enable errors after parsing

            if mol is not None:
                features = np.array(fpgen.GetFingerprint(mol))
            else:
                features= np.zeros(fp_size)

            features_full.extend(features)
        
        # Flatten all features into a single list and append metadata
        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)
    
    return np.array(X)




def RDKit_AtomPairGenerator_NG(smiles_df, Nrows, Nconstituents):

    fp_size= 1024
    fpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size)

    X = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            composition.append(amount)
            
            RDLogger.DisableLog('rdApp.error')   # Suppress RDKit errors
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog('rdApp.error')  # Re-enable errors after parsing

            if mol is not None:
                features = np.array(fpgen.GetFingerprint(mol))
            else:
                features= np.zeros(fp_size)

            features_full.extend(features)
        
        # Flatten all features into a single list and append metadata
        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)
    
    return np.array(X)




def RDKit_MorganGenerator_NG(smiles_df, Nrows, Nconstituents):

    fp_size= 1024
    fpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=fp_size)

    X = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            composition.append(amount)
            
            RDLogger.DisableLog('rdApp.error')   # Suppress RDKit errors
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog('rdApp.error')  # Re-enable errors after parsing

            if mol is not None:
                features = np.array(fpgen.GetFingerprint(mol))
            else:
                features= np.zeros(fp_size)

            features_full.extend(features)
        
        # Flatten all features into a single list and append metadata
        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)
    
    return np.array(X)


# # RDKit 3D Descriptors

# In[ ]:


def RDKit_3DDescriptors_NG(smiles_df, Nrows, Nconstituents):
    X = []
    RDLogger.DisableLog('rdApp.error')

    for i in range(Nrows):
        print(f"Working on {i}")
        composition = []
        features_full = []

        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]

        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            composition.append(amount)

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                features = np.zeros(11)
            else:
                mol = Chem.AddHs(mol)

                # Customizable ETKDG parameters for complex molecules
                params = AllChem.ETKDGv3()
                params.useRandomCoords = True
                params.maxAttempts = 10
                params.pruneRmsThresh = 0.05
                params.randomSeed = 42
                
                try:
                    result = AllChem.EmbedMolecule(mol, params)
                    if result != 0:
                        print(f"Failed to embed molecule: {smiles}")
                        features = np.zeros(11)
                    else:
                        try:
                            AllChem.UFFOptimizeMolecule(mol)
                            features = np.array(list(Descriptors3D.CalcMolDescriptors3D(mol).values()))
                        except Exception as e:
                            print(f"Optimization failed: {smiles}, Error: {e}")
                            features = np.zeros(11)
                except Exception as e:
                    print(f"Error during embedding: {smiles}, Error: {e}")
                    features = np.zeros(11)

            features_full.extend(features)

        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)

    RDLogger.EnableLog('rdApp.error')
    return np.array(X)


# # Descriptastorus Featurizers

# In[ ]:


def Descriptastorus_RDKit2DNormalized_NG(smiles_df, Nrows, Nconstituents):
    
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    
    X = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            composition.append(amount)
            
            RDLogger.DisableLog('rdApp.error')   # Suppress RDKit errors
            mol = Chem.MolFromSmiles(smiles)
            RDLogger.EnableLog('rdApp.error')  # Re-enable errors after parsing

            if mol is not None:
                results = generator.process(smiles)
                processed, features = results[0], np.array(results[1:])
            else:
                features= np.zeros(200)  # size of normalized descriptors

            features_full.extend(features)
        
        # Flatten all features into a single list and append metadata
        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)
    
    return np.array(X)





# # Pytorch geometric Featurizer

# In[ ]:


def PytorchGeometric_NG(smiles_df, Nrows, Nconstituents):

    X = []

    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]

        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2 * j]
            amount = smiles_df.iloc[i, 2 * j + 1]
            composition.append(amount)
            
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                constituent_graph = from_smiles(smiles)
                node_features = constituent_graph.x.numpy()  # Node features
                edge_features = constituent_graph.edge_attr.numpy()  # Edge features
                

                node_mean = np.mean(node_features, axis=0)
                node_std = np.std(node_features, axis=0)
                node_min = np.min(node_features, axis=0)
                node_max = np.max(node_features, axis=0)
            
                edge_mean = np.mean(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_std = np.std(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_min = np.min(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_max = np.max(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                
                features = np.hstack((node_mean, node_std, node_min, node_max, 
                          edge_mean, edge_std, edge_min, edge_max))

            else:
                features= np.zeros(48)   # size of pytorch features after nincluding statistics

            features_full.extend(features)

        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)

    return np.array(X)


# # DeepChem Featurizers

# In[ ]:


def DeepChem_MolGraphConv_NG(smiles_df, Nrows, Nconstituents):

    featurizer= dc.feat.MolGraphConvFeaturizer(use_edges=True, use_partial_charge=True)
    
    # Initialize empty lists for features and target values
    X = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2 * j]
            amount = smiles_df.iloc[i, 2 * j + 1]
            composition.append(amount)
            
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                constituent_graph = featurizer.featurize(smiles)
                node_features = constituent_graph[0].node_features
                edge_features= constituent_graph[0].edge_features
                
                node_mean = np.mean(node_features, axis=0)
                node_std = np.std(node_features, axis=0)
                node_min = np.min(node_features, axis=0)
                node_max = np.max(node_features, axis=0)
            
                edge_mean = np.mean(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_std = np.std(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_min = np.min(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_max = np.max(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                
                features = np.hstack((node_mean, node_std, node_min, node_max, 
                          edge_mean, edge_std, edge_min, edge_max))
            else:
                features= np.zeros(168)

            features_full.extend(features)

        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)

    return np.array(X)



def DeepChem_DMPNN_NG(smiles_df, Nrows, Nconstituents):

    featurizer= dc.feat.DMPNNFeaturizer()
    
    # Initialize empty lists for features and target values
    X = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        composition= []  # Initialize an empty list to store LNP composition
        features_full= []  # Initialize an empty list to store descriptors for all constituents
        
        # Get composition, RNA_type, lipid_to_RNA, dosage, and target value
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]
        
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2 * j]
            amount = smiles_df.iloc[i, 2 * j + 1]
            composition.append(amount)
            
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:
                constituent_graph = featurizer.featurize(smiles)
                node_features = constituent_graph[0].node_features
                edge_features= constituent_graph[0].edge_features
                
                node_mean = np.mean(node_features, axis=0)
                node_std = np.std(node_features, axis=0)
                node_min = np.min(node_features, axis=0)
                node_max = np.max(node_features, axis=0)
            
                edge_mean = np.mean(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_std = np.std(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_min = np.min(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                edge_max = np.max(edge_features, axis=0) if edge_features is not None else np.zeros(node_features.shape[1])
                
                features = np.hstack((node_mean, node_std, node_min, node_max, 
                          edge_mean, edge_std, edge_min, edge_max))
            else:
                features= np.zeros(588)

            features_full.extend(features)

        final_features = np.concatenate([features_full, composition, [RNA_type, lipid_to_RNA, dosage]])
        X.append(final_features)

    return np.array(X)



# # Set of all featurizers for graph based ML models (*_GG, GCN and GAT)

# # Function to generate target variables

# In[ ]:


def generate_targets_GG(target_df, Nrows):
    y= []
    for i in range(Nrows):
        target = target_df.iloc[i, 0]
        y.append(float(target))

    return torch.tensor(y, dtype=torch.long)
    


# # Create Pytorch Data for Graph based ML models

# In[ ]:


def PytorchGeometric_GG(smiles_df, Nrows, Nconstituents):
    """
    For each LNP sample in the dataframes:
      1. Generate a graph (nodes and edge_index) for each constituent using from_smiles.
      2. Merge the constituent graphs by concatenating node features and edge indices.
      3. Attach global features (composition, RNA_type, lipid_to_RNA, dosage) to the merged graph.
    """
    data_list = []

    NODE_FEAT_DIM = 9  # expected dimension of node features
    DUMMY_NODE_FEATURE = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float)  # dummy node for invalid SMILES

    for i in range(Nrows):
        node_list = []
        edge_index_list = []
        composition = []
        
        # Retrieve global metadata for the sample.
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]

        cum_nodes = 0  # running node index offset
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2 * j]
            amount = float(smiles_df.iloc[i, 2 * j + 1])
            composition.append(amount)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                graph = from_smiles(smiles)
                node_list.append(graph.x.float())
                edge_index_list.append(graph.edge_index + cum_nodes)
                cum_nodes += graph.x.size(0)
            else:
                # For an invalid SMILES, substitute with one dummy node.
                node_list.append(DUMMY_NODE_FEATURE.clone())
                cum_nodes += 1

        # Merge node features.
        if node_list:
            x = torch.cat(node_list, dim=0)
        else:
            x = torch.empty((0, NODE_FEAT_DIM), dtype=torch.float)
        # Merge edge indices.
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        merged_data = Data(x=x, edge_index=edge_index)
        
        merged_data.global_features = torch.cat([torch.tensor(composition, dtype=torch.float),torch.tensor([float(RNA_type), float(lipid_to_RNA), float(dosage)], dtype=torch.float)], dim=0).unsqueeze(0)

        data_list.append(merged_data)
        
    return data_list


# # DeepChem Featurizers

# In[ ]:


def DeepChem_DMPNN_GG(smiles_df, Nrows, Nconstituents):

    featurizer = dc.feat.DMPNNFeaturizer()
    
    data_list = []

    NODE_FEAT_DIM = 133  # expected dimension of node features
    DUMMY_NODE_FEATURE = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float)  # dummy node for invalid SMILES

    for i in range(Nrows):
        node_list = []
        edge_index_list = []
        composition = []
        
        # Retrieve global metadata for the sample.
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]

        cum_nodes = 0  # running node index offset
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2 * j]
            amount = float(smiles_df.iloc[i, 2 * j + 1])
            composition.append(amount)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                graph = featurizer.featurize(smiles)
                node_list.append(torch.tensor(graph[0].node_features, dtype=torch.float))
                edge_index_list.append(torch.tensor(graph[0].edge_index, dtype=torch.long) + cum_nodes)
                cum_nodes += torch.tensor(graph[0].node_features, dtype=torch.float).size(0)
            else:
                # For an invalid SMILES, substitute with one dummy node.
                node_list.append(DUMMY_NODE_FEATURE.clone())
                cum_nodes += 1

        # Merge node features.
        if node_list:
            x = torch.cat(node_list, dim=0)
        else:
            x = torch.empty((0, NODE_FEAT_DIM), dtype=torch.float)
        # Merge edge indices.
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        merged_data = Data(x=x, edge_index=edge_index)
        
        merged_data.global_features = torch.cat([torch.tensor(composition, dtype=torch.float),torch.tensor([float(RNA_type), float(lipid_to_RNA), float(dosage)], dtype=torch.float)], dim=0).unsqueeze(0)

        data_list.append(merged_data)
        
    return data_list




def DeepChem_MolGraphConv_GG(smiles_df, Nrows, Nconstituents):

    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True, use_partial_charge=True)
    
    data_list = []

    NODE_FEAT_DIM = 31  # expected dimension of node features
    DUMMY_NODE_FEATURE = torch.zeros((1, NODE_FEAT_DIM), dtype=torch.float)  # dummy node for invalid SMILES

    for i in range(Nrows):
        node_list = []
        edge_index_list = []
        composition = []
        
        # Retrieve global metadata for the sample.
        RNA_type = smiles_df.iloc[i, 2 * Nconstituents]
        lipid_to_RNA = smiles_df.iloc[i, 2 * Nconstituents + 1]
        dosage = smiles_df.iloc[i, 2 * Nconstituents + 2]

        cum_nodes = 0  # running node index offset
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2 * j]
            amount = float(smiles_df.iloc[i, 2 * j + 1])
            composition.append(amount)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                graph = featurizer.featurize(smiles)
                node_list.append(torch.tensor(graph[0].node_features, dtype=torch.float))
                edge_index_list.append(torch.tensor(graph[0].edge_index, dtype=torch.long) + cum_nodes)
                cum_nodes += torch.tensor(graph[0].node_features, dtype=torch.float).size(0)
            else:
                # For an invalid SMILES, substitute with one dummy node.
                node_list.append(DUMMY_NODE_FEATURE.clone())
                cum_nodes += 1

        # Merge node features.
        if node_list:
            x = torch.cat(node_list, dim=0)
        else:
            x = torch.empty((0, NODE_FEAT_DIM), dtype=torch.float)
        # Merge edge indices.
        if edge_index_list:
            edge_index = torch.cat(edge_index_list, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        merged_data = Data(x=x, edge_index=edge_index)
        
        merged_data.global_features = torch.cat([torch.tensor(composition, dtype=torch.float),torch.tensor([float(RNA_type), float(lipid_to_RNA), float(dosage)], dtype=torch.float)], dim=0).unsqueeze(0)

        data_list.append(merged_data)
        
    return data_list



# In[2]:


# !jupyter nbconvert --to script All_Featurizers.ipynb


# In[ ]:




