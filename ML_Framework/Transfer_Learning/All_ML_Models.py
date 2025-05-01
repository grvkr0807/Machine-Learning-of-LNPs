#!/usr/bin/env python
# coding: utf-8

# # ML Models Non Graph (NG)

# In[ ]:


# Import necessary libraries
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




# Random Forest Classification
def ML_Model_RF(X_train, X_test, y_train, y_test):
    
    # Create and train a Random Forest Regression model
    model = RandomForestClassifier(n_estimators= 100, max_depth= 10, min_samples_leaf= 4, min_samples_split= 10, random_state= 42, class_weight= 'balanced')
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    
    return model, y_pred


# SVM Classification
def ML_Model_SVM_Linear(X_train, X_test, y_train, y_test):
    
    # Create and train a Random Forest Regression model
    model = SVC(kernel='linear', random_state= 42, C= 1.0, max_iter=200, class_weight= 'balanced')
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    
    return model, y_pred


def ML_Model_SVM_RBF(X_train, X_test, y_train, y_test):
    
    # Create and train a Random Forest Regression model
    model = SVC(kernel='rbf', random_state= 42, gamma= 'scale', C= 0.1, max_iter=200, class_weight= 'balanced')
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    
    return model, y_pred



# Gradient Boosting Classification
def ML_Model_GB(X_train, X_test, y_train, y_test):
    
    # Create and train a Random Forest Regression model
    model = GradientBoostingClassifier(n_estimators= 100, max_depth=3, random_state= 42, learning_rate= 0.01)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    
    return model, y_pred



# Adaptive Boosting Classification
def ML_Model_AdaB(X_train, X_test, y_train, y_test):
    
    # Create and train a Random Forest Regression model
    model = AdaBoostClassifier(n_estimators= 100, random_state= 42, learning_rate= 0.01)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    
    return model, y_pred


# Extra Trees Classification
def ML_Model_ExtraTrees(X_train, X_test, y_train, y_test):
    
    # Create and train a Random Forest Regression model
    model = ExtraTreesClassifier(n_estimators= 50, max_depth=None, min_samples_split= 10, min_samples_leaf= 4, random_state= 42)
    model.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    
    return model, y_pred




# # ML Models Graph Based (GG)

# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import DataLoader

def ML_Model_GCN(X_train, X_test, y_train, y_test):
    """
    Trains a GCN model on LNP graphs. The model uses:
      - Node features (data.x) and edge connectivity (data.edge_index) via GCNConv layers.
      - Global pooling (global_mean_pool) to produce a graph-level node embedding.
      - Extra global features (composition, RNA_type, lipid_to_RNA, dosage) attached to each Data object.
      - The pooled node embedding and the global features are concatenated and fed into a fully connected layer.
    
    The targets (y_train, y_test) are merged into the Data objects via their .y attribute.
    """
    # Merge targets with graph data
    def merge_targets_with_data(data_list, target_tensor):
        for i, data in enumerate(data_list):
            data.y = target_tensor[i]
        return data_list

    X_train = merge_targets_with_data(X_train, y_train)
    X_test  = merge_targets_with_data(X_test, y_test)

    num_features = X_train[0].x.shape[1]
    num_classes = len(torch.unique(y_train))
    hidden_dim = 32
    epochs = 200
    learning_rate= 0.001
    global_dim = X_train[0].global_features.shape[1]

    class GCN(torch.nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, global_dim):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = torch.nn.Linear(hidden_dim + global_dim, num_classes)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            node_pool = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
            global_extra = data.global_features  # Should have shape [num_graphs, global_dim]
            graph_repr = torch.cat([node_pool, global_extra], dim=1)
            out = self.fc(graph_repr)
            return F.log_softmax(out, dim=1)
    
    model = GCN(num_features, hidden_dim, num_classes, global_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=32)

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
    
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
    return model, y_pred





def ML_Model_GAT(X_train, X_test, y_train, y_test):
    """
    Trains a GAT model on LNP graphs. The model uses:
      - Node features (data.x) and edge connectivity (data.edge_index) via GATConv layers.
      - Global pooling (global_mean_pool) to produce a graph-level embedding.
      - Extra global features (composition, RNA_type, lipid_to_RNA, dosage) attached to each Data object.
      - The pooled node embedding and the global features are concatenated and fed into a fully connected layer.
    
    The targets are merged into the Data objects via the .y attribute.
    """
    # Merge targets with graph data
    def merge_targets_with_data(data_list, target_tensor):
        for i, data in enumerate(data_list):
            data.y = target_tensor[i]
        return data_list

    X_train = merge_targets_with_data(X_train, y_train)
    X_test  = merge_targets_with_data(X_test, y_test)

    num_features = X_train[0].x.shape[1]
    num_classes = len(torch.unique(y_train))
    epochs = 200
    hidden_dim = 64
    num_heads = 4
    learning_rate= 0.01
    global_dim = X_train[0].global_features.shape[1]

    class GAT(torch.nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, num_heads, global_dim):
            super(GAT, self).__init__()
            self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
            self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
            self.fc = torch.nn.Linear(hidden_dim + global_dim, num_classes)
        
        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            node_pool = global_mean_pool(x, batch)
            global_extra = data.global_features
            graph_repr = torch.cat([node_pool, global_extra], dim=1)
            out = self.fc(graph_repr)
            return F.log_softmax(out, dim=1)
    
    model = GAT(num_features, hidden_dim, num_classes, num_heads, global_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
    train_loader = DataLoader(X_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(X_test, batch_size=32)
    
    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y.view(-1))
            loss.backward()
            optimizer.step()
    
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data in test_loader:
            out = model(data)
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
    return model, y_pred


# In[3]:


# !jupyter nbconvert --to script All_ML_Models.ipynb


# In[ ]:




