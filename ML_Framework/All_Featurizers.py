#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# # RDKit Descriptors Featurizer

# In[3]:


def RDKit_Descriptors_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    ipc_index= [42]   # index of descriptor ipc to be removed as it has extremely large value
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        mol = Chem.MolFromSmiles(smiles)
        descriptors_dict = Descriptors.CalcMolDescriptors(mol)
        constituent_descriptors = [value for value in descriptors_dict.values()]        
        descriptors_sum = [0] * len(constituent_descriptors)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            mol = Chem.MolFromSmiles(smiles)
            # Calculate all molecular descriptors as a dictionary
            if mol is not None:
                descriptors_dict = Descriptors.CalcMolDescriptors(mol)
                constituent_descriptors = [value for value in descriptors_dict.values()]
                weighted_descriptors = [desc * amount for desc in constituent_descriptors]
                weighted_descriptors= np.array(weighted_descriptors)
                descriptors_sum= descriptors_sum + weighted_descriptors
            
        descriptors_sum= np.delete(descriptors_sum, ipc_index)
        X.append(descriptors_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# # Descriptastorus Featurizers

# In[4]:


def Descriptastorus_MorganCounts_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.MorganCounts() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y


def Descriptastorus_ChiralMorganCounts_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.ChiralMorganCounts() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y



def Descriptastorus_FeatureCounts_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.FeatureMorganCounts() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y



def Descriptastorus_AtomPairCounts_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.AtomPairCounts() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y



def Descriptastorus_RDKitFPBits_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.RDKitFPBits() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y



def Descriptastorus_RDKitFPUnbranched_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.RDKitFPUnbranched() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y



def Descriptastorus_RDKit2D_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdDescriptors.RDKit2D() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y



def Descriptastorus_RDKit2DNormalized_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from descriptastorus.descriptors import rdNormalizedDescriptors
    from descriptastorus.descriptors import rdDescriptors
    from rdkit import Chem

    generator = rdNormalizedDescriptors.RDKit2DNormalized() 
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        target= target_df.iloc[i, 0]
        results = generator.process(smiles)
        processed, constituent_features = results[0], results[1:]  
        features_sum = [0] * len(constituent_features)  # Initialize sum of features with zeros
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            results = generator.process(smiles)
            processed, constituent_features = results[0], results[1:]
            weighted_features = [desc * amount for desc in constituent_features]
            weighted_features= np.array(weighted_features)
            features_sum= features_sum + weighted_features
            if processed is None:
                logging.warning("Unable to process smiles %s", smiles)
            
        X.append(features_sum)
        y.append(target)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    return X, y


# # RDKit Fingerprints Featurizer

# In[5]:


def RDKit_FPGenerator_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Avalon
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fpgen = AllChem.GetRDKitFPGenerator(fpSize=2048)

    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        mol = Chem.MolFromSmiles(smiles)
        constituent_fingerprint = fpgen.GetFingerprint(mol)        
        fingerprint_sum = [0] * len(constituent_fingerprint)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            mol = Chem.MolFromSmiles(smiles)
            # Calculate all molecular descriptors as a dictionary
            if mol is not None:
                constituent_fingerprint = fpgen.GetFingerprint(mol)
                # Convert the fingerprint to a NumPy array and append to X
                # constituent_fingerprint_np = np.array(list(constituent_fingerprint.ToBitString())).astype(int)
                
                weighted_fingerprint = [desc * amount for desc in constituent_fingerprint]
                weighted_fingerprint = np.array(weighted_fingerprint)
                fingerprint_sum = fingerprint_sum + weighted_fingerprint
            
        X.append(fingerprint_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y



def RDKit_AtomPairGenerator_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Avalon
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fpgen = AllChem.GetAtomPairGenerator(fpSize=2048)

    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        mol = Chem.MolFromSmiles(smiles)
        constituent_fingerprint = fpgen.GetFingerprint(mol)        
        fingerprint_sum = [0] * len(constituent_fingerprint)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            mol = Chem.MolFromSmiles(smiles)
            # Calculate all molecular descriptors as a dictionary
            if mol is not None:
                constituent_fingerprint = fpgen.GetFingerprint(mol)
                # Convert the fingerprint to a NumPy array and append to X
                # constituent_fingerprint_np = np.array(list(constituent_fingerprint.ToBitString())).astype(int)
                
                weighted_fingerprint = [desc * amount for desc in constituent_fingerprint]
                weighted_fingerprint = np.array(weighted_fingerprint)
                fingerprint_sum = fingerprint_sum + weighted_fingerprint
            
        X.append(fingerprint_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y



def RDKit_TopologicalTorsionGenerator_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Avalon
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fpgen = AllChem.GetTopologicalTorsionGenerator(fpSize=2048)

    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        mol = Chem.MolFromSmiles(smiles)
        constituent_fingerprint = fpgen.GetFingerprint(mol)        
        fingerprint_sum = [0] * len(constituent_fingerprint)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            mol = Chem.MolFromSmiles(smiles)
            # Calculate all molecular descriptors as a dictionary
            if mol is not None:
                constituent_fingerprint = fpgen.GetFingerprint(mol)
                # Convert the fingerprint to a NumPy array and append to X
                # constituent_fingerprint_np = np.array(list(constituent_fingerprint.ToBitString())).astype(int)
                
                weighted_fingerprint = [desc * amount for desc in constituent_fingerprint]
                weighted_fingerprint = np.array(weighted_fingerprint)
                fingerprint_sum = fingerprint_sum + weighted_fingerprint
            
        X.append(fingerprint_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# # DGL Graph Featurizer

# In[6]:


def DGL_Mol2Bigraph_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    import dgllife
    from dgllife import utils
    from dgllife.utils import mol_to_bigraph, CanonicalBondFeaturizer, CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer

    # Define a function to generate GNN-based graph representations
    def generate_graph_representation(mol):
        num_atoms = mol.GetNumAtoms()
        g = mol_to_bigraph(mol,node_featurizer=CanonicalAtomFeaturizer(),edge_featurizer=CanonicalBondFeaturizer())  
        # g = mol_to_bigraph(molecule,node_featurizer=AttentiveFPAtomFeaturizer(),edge_featurizer=AttentiveFPBondFeaturizer())  
        return g

    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        mol = Chem.MolFromSmiles(smiles)
        constituent_graph = generate_graph_representation(mol)
        node_features = constituent_graph.ndata['h']
        node_features = node_features.numpy()
        edge_features= constituent_graph.edata['e']
        edge_features= edge_features.numpy()
        graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            mol = Chem.MolFromSmiles(smiles)
            # Calculate all molecular descriptors as a dictionary
            if mol is not None:
                constituent_graph = generate_graph_representation(mol)
                node_features = constituent_graph.ndata['h']
                node_features = node_features.numpy()
                edge_features= constituent_graph.edata['e']
                edge_features= edge_features.numpy()
                graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
                weighted_features = [desc * amount for desc in graph_features]
                weighted_features= np.array(weighted_features)
                graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y



def DGL_Mol2CompleteGraph_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    import dgllife
    from dgllife import utils
    from dgllife.utils import mol_to_complete_graph

    def featurize_atoms(mol):
        feats = []
        for atom in mol.GetAtoms():
            feats.append(atom.GetAtomicNum())
        return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    def featurize_edges(mol, add_self_loop=False):
        feats = []
        num_atoms = mol.GetNumAtoms()
        atoms = list(mol.GetAtoms())
        distance_matrix = Chem.GetDistanceMatrix(mol)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j or add_self_loop:
                    feats.append(float(distance_matrix[i, j]))
        return {'dist': torch.tensor(feats).reshape(-1, 1).float()}
    
    # Define a function to generate GNN-based graph representations
    def generate_graph_representation(mol):
        num_atoms = mol.GetNumAtoms()
        g = mol_to_complete_graph(mol,node_featurizer=featurize_atoms,edge_featurizer=featurize_edges)  
        return g

    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        mol = Chem.MolFromSmiles(smiles)
        constituent_graph = generate_graph_representation(mol)
        node_features = constituent_graph.ndata['atomic']
        node_features = node_features.numpy()
        edge_features= constituent_graph.edata['dist']
        edge_features= edge_features.numpy()
        graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            mol = Chem.MolFromSmiles(smiles)
            # Calculate all molecular descriptors as a dictionary
            if mol is not None:
                constituent_graph = generate_graph_representation(mol)
                node_features = constituent_graph.ndata['atomic']
                node_features = node_features.numpy()
                edge_features= constituent_graph.edata['dist']
                edge_features= edge_features.numpy()
                graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
                weighted_features = [desc * amount for desc in graph_features]
                weighted_features= np.array(weighted_features)
                graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# # Pytorch geometric Featurizer

# In[7]:


def PytorchGeometric_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    from rdkit import Chem
    from rdkit.Chem.rdmolops import GetAdjacencyMatrix
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import from_smiles
    from torch.utils.data import DataLoader


    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        constituent_graph = from_smiles(smiles)
        node_features = constituent_graph.x
        node_features = node_features.numpy()
        edge_features= constituent_graph.edge_attr
        edge_features= edge_features.numpy()
        graph_features = np.hstack((np.sum(node_features,axis=0),np.sum(edge_features,axis=0)))
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]

            constituent_graph = from_smiles(smiles)
            node_features = constituent_graph.x
            node_features = node_features.numpy()
            edge_features= constituent_graph.edge_attr
            edge_features= edge_features.numpy()
            graph_features = np.hstack((np.sum(node_features,axis=0),np.sum(edge_features,axis=0)))
            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# # DeepChem Featurizers

# In[29]:


def DeepChem_ConvMol_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers

    featurizer= dc.feat.ConvMolFeaturizer()

    # Initialize empty lists for features and target values
    X = []
    y = []

    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        constituent_graph = featurizer.featurize(smiles)
        for conv_mol in constituent_graph:
            graph_features = conv_mol.get_atom_features()
            graph_features = np.array(graph_features)
            graph_features= np.sum(graph_features, axis=0)
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            constituent_graph = featurizer.featurize(smiles)
            for conv_mol in constituent_graph:
                graph_features = conv_mol.get_atom_features()
                graph_features = np.array(graph_features)
                graph_features= np.sum(graph_features, axis=0)

            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def DeepChem_Weave_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers

    featurizer= dc.feat.WeaveFeaturizer()

    # Initialize empty lists for features and target values
    X = []
    y = []

    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        constituent_graph = featurizer.featurize(smiles)
        for conv_mol in constituent_graph:
            graph_features = conv_mol.get_atom_features()
            graph_features = np.array(graph_features)
            graph_features= np.sum(graph_features, axis=0)
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            constituent_graph = featurizer.featurize(smiles)
            for conv_mol in constituent_graph:
                graph_features = conv_mol.get_atom_features()
                graph_features = np.array(graph_features)
                graph_features= np.sum(graph_features, axis=0)

            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def DeepChem_CircularFP_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers

    featurizer= dc.feat.CircularFingerprint()

    # Initialize empty lists for features and target values
    X = []
    y = []

    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        graph_features = featurizer.featurize(smiles)
        graph_features = np.array(graph_features[0])
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            graph_features = featurizer.featurize(smiles)
            graph_features = np.array(graph_features[0])
            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y



def DeepChem_MACCS_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers

    featurizer= dc.feat.MACCSKeysFingerprint()

    # Initialize empty lists for features and target values
    X = []
    y = []

    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        graph_features = featurizer.featurize(smiles)
        graph_features = np.array(graph_features[0])
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]
            graph_features = featurizer.featurize(smiles)
            graph_features = np.array(graph_features[0])
            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y



def DeepChem_MolGraphConv_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers

    featurizer= dc.feat.MolGraphConvFeaturizer(use_edges=True, use_partial_charge=True)
    
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        constituent_graph = featurizer.featurize(smiles)
        node_features = constituent_graph[0].node_features
        edge_features= constituent_graph[0].edge_features
        graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]

            constituent_graph = featurizer.featurize(smiles)
            node_features = constituent_graph[0].node_features
            edge_features= constituent_graph[0].edge_features
            graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def DeepChem_DMPNN_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers

    featurizer= dc.feat.DMPNNFeaturizer()
    
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        constituent_graph = featurizer.featurize(smiles)
        node_features = constituent_graph[0].node_features
        edge_features= constituent_graph[0].edge_features
        graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]

            constituent_graph = featurizer.featurize(smiles)
            node_features = constituent_graph[0].node_features
            edge_features= constituent_graph[0].edge_features
            graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def DeepChem_PAGTN_Featurizer(smiles_df, target_df, Nrows, Nconstituents):
    import rdkit
    from rdkit import Chem
    import deepchem as dc
    from deepchem.feat import molecule_featurizers, PagtnMolGraphFeaturizer

    featurizer= dc.feat.PagtnMolGraphFeaturizer()
    
    # Initialize empty lists for features and target values
    X = []
    y = []
    
    # Iterate through rows and columns of both dataframes
    for i in range(Nrows):
        smiles= smiles_df.iloc[0,0]
        constituent_graph = featurizer.featurize(smiles)
        node_features = constituent_graph[0].node_features
        edge_features= constituent_graph[0].edge_features
        graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
        graph_features_sum = [0] * len(graph_features)  # Initialize sum of descriptors with zeros
        target= target_df.iloc[i, 0]
        for j in range(Nconstituents):
            smiles = smiles_df.iloc[i, 2*j]
            amount = smiles_df.iloc[i, 2*j+1]

            constituent_graph = featurizer.featurize(smiles)
            node_features = constituent_graph[0].node_features
            edge_features= constituent_graph[0].edge_features
            graph_features = np.hstack((np.mean(node_features, axis=0), np.mean(edge_features, axis=0)))
            weighted_features = [desc * amount for desc in graph_features]
            weighted_features= np.array(weighted_features)
            graph_features_sum = graph_features_sum + weighted_features
            
        X.append(graph_features_sum)
        y.append(target)
    
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# In[3]:


# !jupyter nbconvert --to script All_Featurizers.ipynb

