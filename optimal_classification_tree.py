import math
import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
from graphviz import Digraph

from ucimlrepo import fetch_ucirepo as fetc

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

class oct_tts:
    '''
    A class to handle the import and preprocessing of UCI datasets for optimal classification tree.
    '''
    def __init__(self, data_id, k_folds, n_observations=None):
        '''
        data_id: str, the ID of the dataset to be used
        k_folds: int, number of folds for cross-validation
        n_observations: int, number of observations in the dataset
        '''
        self.data_id = data_id
        self.k_folds = k_folds
        self.n_observations = n_observations
        self.df = self.import_data_uci(self.data_id)
        self.cvs = self.cross_valid_set(self.df, self.k_folds, self.n_observations)
        self.df_train = []
        self.df_cal = []
        self.df_test = []
        for i  in range(self.k_folds):
            self.df_train.append(self.cvs[0][i])
            self.df_cal.append(self.cvs[1][i])
            self.df_test.append(self.cvs[2][i])
    
    def is_numeric_like(self,series):
        try:
            pd.to_numeric(series.dropna().astype(str), errors='raise')
            return True
        except:
            return False
        
    def import_data_uci(self,data_id):
        '''
        Import data from UCI repository and preprocess it.
        data_id: str, the ID of the dataset to be used
        Returns:
        df: pd.DataFrame, preprocessed DataFrame with features and target variable
        '''
        dt = fetc(id = data_id)
        X = dt.data.features 
        y = dt.data.targets 
        df = pd.DataFrame(X, columns=dt.data.feature_names)
        df = df.dropna()
        target_name = 'target'
        df[target_name] = y
        df = df.reset_index(drop=True)
            
        #Change non numeric columns to numeric
        column_names = df.drop([target_name],axis=1).columns.tolist()
        
        for col in column_names:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            if self.is_numeric_like(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        for col in column_names[:len(column_names)-1]:
            df[col] = StandardScaler().fit_transform(df[[col]])
            df[col] = MinMaxScaler().fit_transform(df[[col]])
        
        #Convert target column to numeric
        le = LabelEncoder()
        df[target_name] = le.fit_transform(df[target_name])

        return df
    
    def cross_valid_set(self,df,k,observations=None):
        train_frac, cal_frac, test_frac = 0.5, 0.25, 0.25
        train_rel = train_frac / (train_frac + cal_frac)
        s_train, s_cal, s_test = [], [], []
        
        if observations is not None:
            df = df.sample(n=observations, random_state=42)
            df = df.reset_index(drop=True)
        
        def clean(dt):
            dt_clean = dt.copy()
            dt_clean.reset_index(drop=True, inplace=True)
            dt_clean.columns = df.columns.tolist()
            dt_clean['target'] = dt_clean['target'].astype(int)
            return dt_clean
            
        for i in range(k):
            df_shuf = df.sample(frac=1, random_state=42 + i).reset_index(drop=True)
            df_train_cal, df_test_split = train_test_split(df_shuf, test_size=test_frac, random_state=42 + i)
            df_train_split, df_cal_split = train_test_split(df_train_cal, train_size=train_rel, random_state=42 + i)
    
            s_train.append(clean(df_train_split))
            s_cal.append(clean(df_cal_split))
            s_test.append(clean(df_test_split))
            
        return s_train, s_cal, s_test
        
class bal_OCT:
    def __init__(self,train,test,D,timelim=None,warmstart_a=None,warmstart_b=None, cut=None, heuristics=None, MIPfocus=None, presolve=None):
        '''
        Predermined the parameters
        '''
        self.train = train.reset_index(drop=True)  # training data
        self.X = self.train.drop(['target'],axis=1)  # features
        self.y = self.train['target'].astype(int)
        
        self.test = test.reset_index(drop=True)  # test data
        self.X_test = self.test.drop(['target'],axis=1)
        self.y_test = self.test['target'].astype(int)
        
        self.n = self.X.shape[0]  # number of samples
        self.p = self.X.shape[1]  # number of features
        self.K = len(np.unique(self.y))  # number of classes
        self.D = D  # tree depth
        self.T = 2**(D+1)-1  # number of nodes
        self.L_hat = self.y.value_counts().max() / self.n
        self.N_min = math.floor(self.n * 0.05)
        
        self.epsilon = self._epsilon()
        self.epsilon_max = max(self.epsilon)
        self.Y_matrix = self._Y_matrix()
        
        self.left_ancestors = self._ancestors()[0]  # left ancestors of each node
        self.right_ancestors = self._ancestors()[1]
        
        self.TB = list(range(1, 2**D))   # branch (internal) nodes
        self.TL = list(range(2**D, 2**(D + 1)))  # leaf nodes
        
        '''
        Gurobi Model
        '''
        self.model = gp.Model('OCT')
        self.add_variables()  # add variables to the model
        
        #WarmStart
        self.clf = DecisionTreeClassifier(criterion='gini', max_depth=self.D, max_leaf_nodes=4, random_state=42)
        self.clf.fit(self.X, self.y)
        if warmstart_a is None and warmstart_b is None:
            self.feature_indices = self.warmstart_cart(self.clf)[0]
            self.threshold_indices = self.warmstart_cart(self.clf)[1]
            self.warmstart(self.feature_indices, self.threshold_indices)
        else:
            self.feature_indices = warmstart_a
            self.threshold_indices = warmstart_b
            self.warmstart(self.feature_indices, self.threshold_indices)
                 
        # Constraints
        self.add_constraints(self.model, self.a, self.b, self.d, self.z, self.l, self.Nk, self.N, self.ck, self.L, self.X, self.Y_matrix, self.left_ancestors, self.right_ancestors)
        
        # Objective function
        self.solvetime = timelim
        self.cut = cut
        self.heuristics = heuristics
        self.MIPfocus = MIPfocus
        self.presolve = presolve
        self.model.update()
        
        self.model.setObjective(self.L.sum('*') / self.L_hat, GRB.MINIMIZE)
        
        if timelim is not None:
            self.model.Params.TimeLimit = self.solvetime  # set time limit for optimization
        if cut is not None:
            self.model.Params.Cuts = self.cut  # set time limit for optimization
        if heuristics is not None:
            self.model.Params.Heuristics = self.heuristics 
        if MIPfocus is not None:
            self.model.Params.MIPFocus = self.MIPfocus 
        if presolve is not None:
            self.model.Params.Presolve = self.presolve  # set time limit for optimization
        
        self.model.optimize()  # optimize the model
        
        '''
        Splitting Criteria
        '''
        self.a_matrix = np.zeros((self.p, len(self.TB)))  # a matrix
        for i in range(len(self.TB)):
            for j in range(self.p):
                self.a_matrix[j, i] = self.a[j, self.TB[i]].X
                
        self.b_matrix = np.zeros(len(self.TB), dtype=float)  # b matrix
        for i in range(len(self.TB)):
            self.b_matrix[i] = self.b[self.TB[i]].X

        '''
        Calculate OCT and CART Accuracy
        '''
        # CART Accuracy
        self.cart_tree = DecisionTreeClassifier(criterion='gini', max_depth=self.D,random_state=42)
        self.cart_tree.fit(self.X, self.y)
        self.cart_accuracy_train= self.cart_accuracy(self.cart_tree)[0]
        self.cart_accuracy_test = self.cart_accuracy(self.cart_tree)[1]
        
        # OCT Accuracy
        self.accuracy_train = self.compute_accuracy(self.X, self.y)
        self.accuracy_test = self.compute_accuracy(self.X_test, self.y_test) 
            
    def _epsilon(self):
        epsilon = []
        for j in range(self.p):
            x_j = self.X.iloc[:, j].tolist()
            x_j.sort()
            e = []
            for i in range(self.n - 1):
                if x_j[i + 1] != x_j[i]:
                    e.append(x_j[i + 1] - x_j[i])
                else:
                    e.append(0.00001)
            epsilon.append(min(e))
        return epsilon

    def _Y_matrix(self):
        Y = np.full((self.n, self.K), -1, dtype=int)
        for i in range(self.n):
            Y[i, self.y[i]] = 1
        return Y
    
    def _ancestors(self):
        left_ancestors = []
        right_ancestors = []
        for t in range(1, self.T + 1):
            la_t =[]
            ra_t =[]
            tau=t
            while tau>1:
                pt = tau//2
                if tau % 2 == 0:
                    la_t.append(pt)
                else:
                    ra_t.append(pt)
                tau = pt
            la_t.sort() 
            ra_t.sort()
            left_ancestors.append(la_t)
            right_ancestors.append(ra_t)
        return left_ancestors, right_ancestors
    
    def add_variables(self):
        self.a = self.model.addVars(self.p, self.TB, vtype=GRB.BINARY, name="a_t")  # shape: p × |TB|
        self.b = self.model.addVars(self.TB, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="b_t")  # shape: |TB|
        self.d = self.model.addVars(self.TB, vtype=GRB.BINARY, name="d_t")  # shape: |TB|
        self.z = self.model.addVars(self.n, self.TL, vtype=GRB.BINARY, name="z")  # shape: n × |TL|
        self.l = self.model.addVars(self.TL, vtype=GRB.BINARY, name="l_t")  # shape: |TL|
        self.Nk = self.model.addVars(self.K, self.TL, vtype=GRB.INTEGER, name="N_kt")  # shape: K × |TL|
        self.N = self.model.addVars(self.TL, vtype=GRB.INTEGER, name="N_t")  # shape: |TL|
        self.ck = self.model.addVars(self.K, self.TL, vtype=GRB.BINARY, name="c_kt")  # shape: K × |TL|
        self.L = self.model.addVars(self.TL, name="L_t")  # shape: |TL|
    
    def warmstart(self, a, b):
        ''' 
        a = warmstart_a, b = warmstart_b
        warmstart_a: feature indices for each branch node
        warmstart_b: threshold values for each branch node
        '''
        for t in self.TB:
            self.b[t].Start = b[t-1]
        for t,j in enumerate(a):
            if j != -2:
                self.a[int(j), self.TB[t]].Start = 1 
    
    def warmstart_cart(self,cart):
        '''
        Warm start the model with initial values from a CART model.
        cart: A trained DecisionTreeClassifier object.
        Returns:
        - feature_indices: List of feature indices used in the CART model.
        - threshold_indices: List of threshold values used in the CART model.
        '''
        feature = cart.tree_.feature
        threshold = cart.tree_.threshold
        feature_indices = []
        threshold_indices = []
        if len(feature) == 2**(self.D+1)-1:
            feature_indices = [f for f in feature if f != -2]
            threshold_indices = [f for f in threshold if f != -2]
        else:
            feature_indices = feature.tolist()[:2**self.D-1]
            threshold_indices = threshold.tolist()[:2**self.D-1]
        if len(feature_indices) < 2**self.D-1:
            for i in range(len(feature_indices), 2**self.D-1):
                feature_indices.append(-2)
                threshold_indices.append(0)
        for i in range(len(feature_indices)):
            if threshold_indices[i] == -2:
                threshold_indices[i] = 0
        feature_indices = [int(f) if f != -2 else -2 for f in feature_indices]
        threshold_indices = [float(t) if t != -2 else 0 for t in threshold_indices]
        return feature_indices, threshold_indices
            
    def add_constraints(self, model, a, b, d, z, l, Nk, N, ck, L, X_train, Y, left_ancestors, right_ancestors):
        for t in self.TB:
            model.addConstr(a.sum("*", t) == d[t], name="sum_constraint_of_ajt")
            model.addConstr(b[t] <= d[t], name="bt_constraint_dt")
            model.addConstr(d[t] == 1, name="dt_constraint_d(t)")
        for t in self.TB[1:]:
            model.addConstr(d[t] <= d[t // 2], name="dt_constraint_dp(t)")
        for i in range(self.n):
            model.addConstr(z.sum(i, "*") == 1, name="sum_of_zi(t)_constraint_1")
        for t in self.TL:
            model.addConstr(z.sum("*", t) >= self.N_min * l[t], name="sum_of_zt_constraint_Nmin_lt")
            for i in range(self.n):
                model.addConstr(z[i, t] <= l[t])
            for k in range(self.K):
                model.addConstr(Nk[k, t] == 0.5 * gp.quicksum(z[i, t] * (Y[i, k] + 1) for i in range(self.n)), name=f"Nkt_{k}_{t}")
            model.addConstr(N[t] == z.sum("*", t), name=f"Nt_{t}")
            model.addConstr(l[t] == ck.sum("*", t), name=f"sum_ck_{t}")
            model.addConstr(l[t] == 1, name="dt_constraint_l(t)")
        for t in self.TL:
            l_ancestors = left_ancestors[t - 1]
            if l_ancestors:
                for la in l_ancestors:
                    for i in range(self.n):
                        xi = X_train.iloc[i]
                        model.addConstr(
                            gp.quicksum(a[j, la] * (xi[j] + self.epsilon[j]) for j in range(self.p)) <= b[la] + (1 + self.epsilon_max) * (1 - z[i, t]),
                            name=f"split_l_{la}_{i}_{t}"
                        )
            r_ancestors = right_ancestors[t - 1]
            if r_ancestors:
                for r in r_ancestors:
                    for i in range(self.n):
                        xi = X_train.iloc[i]
                        model.addConstr(
                            gp.quicksum(a[j, r] * xi[j] for j in range(self.p)) >= b[r] - (1 - z[i, t]),
                            name=f"split_r_{r}_{i}_{t}"
                       )
        for t in self.TL:
            model.addConstr(L[t] >= 0, name="Lt_constraint3")
            for k in range(self.K):
                model.addConstr(L[t] >= N[t] - Nk[k, t] - self.n * (1 - ck[k, t]), name="Lt_constraint1")
                model.addConstr(L[t] <= N[t] - Nk[k, t] + self.n * ck[k, t], name="Lt_constraint2") 

    def compute_leaf_indices(self, X_eval):
        ''' 
        To determine the indices of the points that are in each leaf. 
        '''
        leaf_indices = [[] for _ in range(2**self.D)]
        for i in range(X_eval.shape[0]):
            j = 1
            x_i = X_eval.iloc[i].to_numpy()
            while j < 2**self.D:
                if np.dot(self.a_matrix[:, j - 1], x_i) < self.b_matrix[j - 1]:
                    j = 2 * j
                else:
                    j = 2 * j + 1
            leaf_indices[j - 2**self.D].append(i)
        return leaf_indices
    
    def compute_leaf_classes(self,X_eval,y_eval):
        ''' 
        To determine the class label of each leaf. 
        '''
        leaf_indices = self.compute_leaf_indices(X_eval)
        leaf_classes = []
        for leaf in leaf_indices:
            if leaf:
                labels_in_leaf = y_eval.iloc[leaf]
                predicted_class = labels_in_leaf.mode()[0]
                leaf_classes.append(predicted_class)
            else:
                leaf_classes.append(None)
        return leaf_classes
    
    def compute_accuracy(self, X_eval, y_eval):
        D = self.D
        leaf_indices = self.compute_leaf_indices(X_eval) # Determines the indices of points from X_eval that are in each leaf. 
        leaf_classes = self.compute_leaf_classes(self.X,self.y) #Determines the class label based on train set. 
        total_misclassified = 0
        for i in range(2**D):
            indices = leaf_indices[i]
            if indices:
                actual_labels = y_eval.iloc[indices]
                predicted_label = leaf_classes[i]
                misclassified = (actual_labels != predicted_label).sum()
                total_misclassified += misclassified
        accuracy = (X_eval.shape[0] - total_misclassified) / X_eval.shape[0]
        return accuracy
    
    def cart_accuracy(self,cart):
        '''
        To compute the accuracy of the CART model on the training and test sets.
        Returns:
        - cart_accuracy_train: float, the accuracy of the CART model on the training set
        - cart_accuracy_test: float, the accuracy of the CART model on the test set
        '''
        cart.fit(self.X, self.y)
        y_pred_gini = cart.predict(self.X_test)
        y_pred_gini_train = cart.predict(self.X)
        return accuracy_score(self.y, y_pred_gini_train),accuracy_score(self.y_test, y_pred_gini)
    
    ''' 
    Plot Tree Structure
    '''
    def _generate_tree_structure(self):
        T = self.T
        TB = self.TB
        tree_structure = {}
        for t in TB:
            left = 2 * t
            right = 2 * t + 1
            if right <= T:
                tree_structure[t] = [left, right]
        return tree_structure
    
    def _extract_feature_names_from_a(self):
        feature_names = {}
        feature_columns = self.X.columns
        for t_idx, t in enumerate(self.TB):
            for j in range(self.a_matrix.shape[0]):
                if self.a_matrix[j, t_idx] == 1:
                    feature_names[t] = feature_columns[j]
                    break
        return feature_names
    
    def _extract_threshold_values_from_b(self):
        threshold_values = {t: self.b_matrix[t_idx] for t_idx, t in enumerate(self.TB)}
        return threshold_values

    def _build_leaf_labels(self):
        labels = self.compute_leaf_classes(self.X,self.y)
        return {t: labels[i] for i, t in enumerate(self.TL)}

    def plot_tree_structure(self):  
        tree_structure = self._generate_tree_structure()
        feature_names = self._extract_feature_names_from_a()
        threshold_values = self._extract_threshold_values_from_b()
        leaf_labels = self._build_leaf_labels()
        dot = Digraph()
        for node_id, children in tree_structure.items():
            feature = feature_names.get(node_id, f"Feature {node_id}")
            threshold = threshold_values.get(node_id, "?")
            dot.node(str(node_id), f"{feature} ≤ {threshold:.2f}", shape='ellipse')
            for idx, child in enumerate(children):
                edge_label = "Yes" if idx == 0 else "No"
                dot.edge(str(node_id), str(child), label=edge_label)
        for leaf_id, class_label in leaf_labels.items():
            dot.node(str(leaf_id), f"Class: {class_label}", shape='box', style='filled', fillcolor='lightgrey')
        dot.render("OCT_tree_structure", format='png', cleanup=True)
        dot.view()        
            
class OCT_Main:
    def __init__(self,train,test,D,alpha,timelim=None,warmstart_a=None,warmstart_b=None, cut=None, heuristics=None, MIPfocus=None, presolve=None):
        '''
        Predermined Parameters
        '''
        self.train = train.reset_index(drop=True)  # training data
        self.X = self.train.drop(['target'],axis=1)
        self.y = self.train['target'].astype(int)
        
        self.test = test.reset_index(drop=True)  # test data
        self.X_test = self.test.drop(['target'],axis=1)
        self.y_test = self.test['target'].astype(int)
        
        self.n = self.X.shape[0]  # number of observations
        self.p = self.X.shape[1]  # number of features
        self.K = len(np.unique(self.y))  # number of classes
        self.D = D  # tree depth
        self.T = 2**(D+1)-1  # number of nodes
        self.L_hat = self.y.value_counts().max() / self.n
        self.N_min = math.floor(self.n * 0.05)
        self.alpha = alpha
        
        self.epsilon = self._epsilon()
        self.epsilon_max = max(self.epsilon)
        self.Y_matrix = self._Y_matrix()
        
        self.left_ancestors = self._ancestors()[0]
        self.right_ancestors = self._ancestors()[1]
        
        self.TB = list(range(1, 2**D))
        self.TL = list(range(2**D, 2**(D + 1)))
        
        '''
        OCT Model
        '''
        self.model = gp.Model('OCT')
        self.add_variables()
        
        # WarmStart
        self.clf = DecisionTreeClassifier(criterion='gini', max_depth=self.D, random_state=42)
        self.clf.fit(self.X, self.y)
        if warmstart_a is None and warmstart_b is None:
            self.feature_indices = self.warmstart_cart(self.clf)[0]
            self.threshold_indices = self.warmstart_cart(self.clf)[1]
            self.warmstart(self.feature_indices, self.threshold_indices)
        else:
            self.feature_indices = warmstart_a
            self.threshold_indices = warmstart_b
            self.warmstart(self.feature_indices, self.threshold_indices)
            
        # Constraints
        self.add_constraints(self.model, self.a, self.b, self.d, self.z, self.l, self.Nk, self.N, self.ck, self.L, self.X, self.Y_matrix, self.left_ancestors, self.right_ancestors)
        
        # Objective function
        self.solvetime = timelim
        self.cut = cut
        self.heuristics = heuristics
        self.MIPfocus = MIPfocus
        self.presolve = presolve
        self.model.update()
        
        self.model.setObjective(self.L.sum('*') / self.L_hat + self.alpha * gp.quicksum(self.d[t] for t in self.TB), GRB.MINIMIZE)
        
        if timelim is not None:
            self.model.Params.TimeLimit = self.solvetime
        if cut is not None:
            self.model.Params.Cuts = self.cut
        if heuristics is not None:
            self.model.Params.Heuristics = self.heuristics 
        if MIPfocus is not None:
            self.model.Params.MIPFocus = self.MIPfocus 
        if presolve is not None:
            self.model.Params.Presolve = self.presolve
        
        self.model.optimize()
        
        '''
        Splitting Criteria
        '''
        self.a_matrix = np.zeros((self.p, len(self.TB)))  # a matrix
        for i in range(len(self.TB)):
            for j in range(self.p):
                self.a_matrix[j, i] = self.a[j, self.TB[i]].X
                
        self.b_matrix = np.zeros(len(self.TB), dtype=float)  # b matrix
        for i in range(len(self.TB)):
            self.b_matrix[i] = self.b[self.TB[i]].X
            
        '''
        Calculate OCT and CART Accuracy
        '''
        # CART Accuracy
        self.cart_tree = DecisionTreeClassifier(criterion='gini', max_depth=self.D, random_state=42)
        self.cart_tree.fit(self.X, self.y)
        self.cart_accuracy_train= self.cart_accuracy(self.cart_tree)[0]
        self.cart_accuracy_test = self.cart_accuracy(self.cart_tree)[1]
        
        # OCT Accuracy
        self.accuracy_train = self.compute_accuracy(self.X, self.y)
        self.accuracy_test = self.compute_accuracy(self.X_test, self.y_test)  
            
    def _epsilon(self):
        '''
        Obtain the epsilon vector with the method mentioned in the paper.
        '''
        epsilon = []
        for j in range(self.p):
            x_j = self.X.iloc[:, j].tolist()
            x_j.sort()
            e = []
            for i in range(self.n - 1):
                if x_j[i + 1] != x_j[i]:
                    e.append(x_j[i + 1] - x_j[i])
                else:
                    e.append(0.00001)
            epsilon.append(min(e))
        return epsilon

    def _Y_matrix(self):
        '''
        Generate the Y matrix as described in the paper.
        '''
        Y = np.full((self.n, self.K), -1, dtype=int)
        for i in range(self.n):
            Y[i, self.y[i]] = 1
        return Y
    
    def _ancestors(self):
        '''
        Generate the left and right ancestors for each node in the tree.
        '''
        left_ancestors = []
        right_ancestors = []
        for t in range(1, self.T + 1):
            la_t =[]
            ra_t =[]
            tau=t
            while tau>1:
                pt = tau//2
                if tau % 2 == 0:
                    la_t.append(pt)
                else:
                    ra_t.append(pt)
                tau = pt
            la_t.sort() 
            ra_t.sort()
            left_ancestors.append(la_t)
            right_ancestors.append(ra_t)
        return left_ancestors, right_ancestors
    
    def add_variables(self):
        self.a = self.model.addVars(self.p, self.TB, vtype=GRB.BINARY, name="a_t")  # shape: p × |TB|
        self.b = self.model.addVars(self.TB, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="b_t")  # shape: |TB|
        self.d = self.model.addVars(self.TB, vtype=GRB.BINARY, name="d_t")  # shape: |TB|
        self.z = self.model.addVars(self.n, self.TL, vtype=GRB.BINARY, name="z")  # shape: n × |TL|
        self.l = self.model.addVars(self.TL, vtype=GRB.BINARY, name="l_t")  # shape: |TL|
        self.Nk = self.model.addVars(self.K, self.TL, vtype=GRB.INTEGER, name="N_kt")  # shape: K × |TL|
        self.N = self.model.addVars(self.TL, vtype=GRB.INTEGER, name="N_t")  # shape: |TL|
        self.ck = self.model.addVars(self.K, self.TL, vtype=GRB.BINARY, name="c_kt")  # shape: K × |TL|
        self.L = self.model.addVars(self.TL, name="L_t")  # shape: |TL|
    
    def warmstart(self, a, b):
        ''' 
        a = warmstart_a, b = warmstart_b
        warmstart_a: feature indices for each branch node
        warmstart_b: threshold values for each branch node
        '''
        for t in self.TB:
            self.b[t].Start = b[t-1]
        for j in range(self.p):
            for t in self.TB:
                self.a[j, t].Start = 0
        for t,j in enumerate(a):
            if j != -2:
                self.a[int(j), self.TB[t]].Start = 1 
    
    def warmstart_cart(self,cart):
        '''
        Warm start the model with initial values from a CART model.
        cart: A trained DecisionTreeClassifier object.
        Returns:
        - feature_indices: List of feature indices used in the CART model.
        - threshold_indices: List of threshold values used in the CART model.
        '''
        feature = cart.tree_.feature
        threshold = cart.tree_.threshold
        feature_indices = []
        threshold_indices = []
        if len(feature) == 2**(self.D+1)-1:
            feature_indices = [f for f in feature if f != -2]
            threshold_indices = [f for f in threshold if f != -2]
        else:
            feature_indices = feature.tolist()[:2**self.D-1]
            threshold_indices = threshold.tolist()[:2**self.D-1]
        if len(feature_indices) < 2**self.D-1:
            for i in range(len(feature_indices), 2**self.D-1):
                feature_indices.append(-2)
                threshold_indices.append(0)
        for i in range(len(feature_indices)):
            if threshold_indices[i] == -2:
                threshold_indices[i] = 0
        feature_indices = [int(f) if f != -2 else -2 for f in feature_indices]
        threshold_indices = [float(t) if t != -2 else 0 for t in threshold_indices]
        return feature_indices, threshold_indices
            
    def add_constraints(self, model, a, b, d, z, l, Nk, N, ck, L, X_train, Y, left_ancestors, right_ancestors):
        for t in self.TB:
            model.addConstr(a.sum("*", t) == d[t], name="sum_constraint_of_ajt")
            model.addConstr(b[t] <= d[t], name="bt_constraint_dt")
        for t in self.TB[1:]:
            model.addConstr(d[t] <= d[t // 2], name="dt_constraint_dp(t)")
        for i in range(self.n):
            model.addConstr(z.sum(i, "*") == 1, name="sum_of_zi(t)_constraint_1")
        for t in self.TL:
            model.addConstr(z.sum("*", t) >= self.N_min * l[t], name="sum_of_zt_constraint_Nmin_lt")
            for i in range(self.n):
                model.addConstr(z[i, t] <= l[t])
            for k in range(self.K):
                model.addConstr(Nk[k, t] == 0.5 * gp.quicksum(z[i, t] * (Y[i, k] + 1) for i in range(self.n)), name=f"Nkt_{k}_{t}")
            model.addConstr(N[t] == z.sum("*", t), name=f"Nt_{t}")
            model.addConstr(l[t] == ck.sum("*", t), name=f"sum_ck_{t}")
        for t in self.TL:
            l_ancestors = left_ancestors[t - 1]
            if l_ancestors:
                for la in l_ancestors:
                    for i in range(self.n):
                        xi = X_train.iloc[i]
                        model.addConstr(
                            self.epsilon_max + gp.quicksum(a[j, la] * xi[j] for j in range(self.p)) <= b[la] + (1 + self.epsilon_max) * (1 - z[i, t]),
                            name=f"split_l_{la}_{i}_{t}"
                        )
            r_ancestors = right_ancestors[t - 1]
            if r_ancestors:
                for r in r_ancestors:
                    for i in range(self.n):
                        xi = X_train.iloc[i]
                        model.addConstr(
                            gp.quicksum(a[j, r] * xi[j] for j in range(self.p)) >= b[r] - (1 - z[i, t]),
                            name=f"split_r_{r}_{i}_{t}"
                       )
        for t in self.TL:
            model.addConstr(L[t] >= 0, name="Lt_constraint3")
            for k in range(self.K):
                model.addConstr(L[t] >= N[t] - Nk[k, t] - self.n * (1 - ck[k, t]), name="Lt_constraint1")
                model.addConstr(L[t] <= N[t] - Nk[k, t] + self.n * ck[k, t], name="Lt_constraint2") 

    def _generate_feature_indices(self):
        '''
        Generate feature indices based on the a_matrix.
        Returns:
        - feature_indices: List of feature indices for each branch node.
        '''
        feature_indices = []
        for j,t in enumerate(self.a_matrix.T):
            if np.any(t == 1):
                feature_indices.append(j)
            else:
                feature_indices.append(-2)
        return feature_indices
    
    def compute_leaf_indices(self, X_eval):
        ''' 
        To determine the indices of the points that are in each leaf. 
        '''
        leaf_indices = [[] for _ in range(2**self.D)]
        for i in range(X_eval.shape[0]):
            j = 1
            x_i = X_eval.iloc[i].to_numpy()
            while j < 2**self.D:
                if np.dot(self.a_matrix[:, j - 1], x_i) < self.b_matrix[j - 1]:
                    j = 2 * j
                else:
                    j = 2 * j + 1
            leaf_indices[j - 2**self.D].append(i)
        return leaf_indices
    
    def compute_leaf_classes(self,X_eval,y_eval):
        ''' 
        To determine the class label of each leaf. 
        '''
        leaf_indices = self.compute_leaf_indices(X_eval)
        leaf_classes = []
        for leaf in leaf_indices:
            if leaf:
                labels_in_leaf = y_eval.iloc[leaf]
                predicted_class = labels_in_leaf.mode()[0]
                leaf_classes.append(predicted_class)
            else:
                leaf_classes.append(None)
        return leaf_classes
    
    def compute_accuracy(self, X_eval, y_eval):
        '''
        To compute the accuracy of the OCT model on the evaluation set.
        X_eval: pd.DataFrame, evaluation features
        y_eval: pd.Series, evaluation target variable
        Returns:
        - accuracy: float, the accuracy of the OCT model on the evaluation set
        '''
        D = self.D
        leaf_indices = self.compute_leaf_indices(X_eval)
        leaf_classes = self.compute_leaf_classes(self.X,self.y)
        total_misclassified = 0
        for i in range(2**D):
            indices = leaf_indices[i]
            if indices:
                actual_labels = y_eval.iloc[indices]
                predicted_label = leaf_classes[i]
                misclassified = (actual_labels != predicted_label).sum()
                total_misclassified += misclassified
        accuracy = (X_eval.shape[0] - total_misclassified) / X_eval.shape[0]
        return accuracy
    
    def cart_accuracy(self,cart):
        '''
        To compute the accuracy of the CART model on the training and test sets.
        Returns:
        - cart_accuracy_train: float, the accuracy of the CART model on the training set
        - cart_accuracy_test: float, the accuracy of the CART model on the test set
        '''
        cart.fit(self.X, self.y)
        y_pred_gini = cart.predict(self.X_test)
        y_pred_gini_train = cart.predict(self.X)
        return accuracy_score(self.y, y_pred_gini_train),accuracy_score(self.y_test, y_pred_gini)
    
    ''' 
    Plot Tree Structure
    '''
    def _generate_tree_structure(self):
        T = self.T
        TB = self.TB
        tree_structure = {}
        for t in TB:
            left = 2 * t
            right = 2 * t + 1
            if right <= T:
                tree_structure[t] = [left, right]
        return tree_structure 
    def _extract_feature_names_from_a(self):
        feature_names = {}
        feature_columns = self.X.columns
        for t_idx, t in enumerate(self.TB):
            for j in range(self.a_matrix.shape[0]):
                if self.a_matrix[j, t_idx] == 1:
                    feature_names[t] = feature_columns[j]
                    break
        return feature_names
    def _extract_threshold_values_from_b(self):
        threshold_values = {t: self.b_matrix[t_idx] for t_idx, t in enumerate(self.TB)}
        return threshold_values
    def _build_leaf_labels(self):
        labels = self.compute_leaf_classes(self.X,self.y)
        return {t: labels[i] for i, t in enumerate(self.TL)}
    def plot_tree_structure(self):  
        tree_structure = self._generate_tree_structure()
        feature_names = self._extract_feature_names_from_a()
        threshold_values = self._extract_threshold_values_from_b()
        leaf_labels = self._build_leaf_labels()
        dot = Digraph()
        for node_id, children in tree_structure.items():
            feature = feature_names.get(node_id, f"Feature {node_id}")
            threshold = threshold_values.get(node_id, "?")
            dot.node(str(node_id), f"{feature} < {threshold:.2f}", shape='ellipse')
            for idx, child in enumerate(children):
                edge_label = "Yes" if idx == 0 else "No"
                dot.edge(str(node_id), str(child), label=edge_label)
        for leaf_id, class_label in leaf_labels.items():
            dot.node(str(leaf_id), f"Class: {class_label}", shape='box', style='filled', fillcolor='lightgrey')
        dot.render("OCT_tree_structure", format='png', cleanup=True)
        dot.view()


class OCT_AT_Sub:
    def __init__(self,train,test,D,C,timelim=None,warmstart_a=None,warmstart_b=None): 
        self.train = train.reset_index(drop=True)
        self.X = self.train.drop(['target'],axis=1)
        self.y = self.train['target'].astype(int)
        
        self.test = test.reset_index(drop=True)
        self.X_test = self.test.drop(['target'],axis=1)
        self.y_test = self.test['target'].astype(int)
        
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.K = len(np.unique(self.y))
        self.D = D
        self.T = 2**(D+1)-1
        self.L_hat = self.y.value_counts().max() / self.n
        self.N_min = math.floor(self.n * 0.05)
        
        self.epsilon = self._epsilon()
        self.epsilon_max = max(self.epsilon)
        self.Y_matrix = self._Y_matrix()
        
        self.left_ancestors = self._ancestors()[0]
        self.right_ancestors = self._ancestors()[1]
        
        self.TB = list(range(1, 2**D))
        self.TL = list(range(2**D, 2**(D + 1)))
        
        self.C = C
        self.model = gp.Model('OCT')
        self.add_variables()
        
        #WarmStart
        self.clf = DecisionTreeClassifier(criterion='gini', max_depth=D, max_leaf_nodes= self.C+1, random_state=42)
        self.clf.fit(self.X, self.y)
        if warmstart_a is None and warmstart_b is None:
            self.feature_indices = self.warmstart_cart(self.clf)[0]
            self.threshold_indices = self.warmstart_cart(self.clf)[1]
            self.warmstart(self.feature_indices, self.threshold_indices)
        else:
            self.feature_indices = warmstart_a
            self.threshold_indices = warmstart_b
            self.warmstart(self.feature_indices, self.threshold_indices)
        
        # Constraints
        self.add_constraints(self.model, self.a, self.b, self.d, self.z, self.l, self.Nk, self.N, self.ck, self.L, self.X, self.Y_matrix, self.left_ancestors, self.right_ancestors)
        
        # Objective function
        self.solvetime = timelim
        self.model.update()
        self.model.setObjective(self.L.sum('*') / self.L_hat, GRB.MINIMIZE)
        if timelim is not None:
            self.model.Params.TimeLimit = self.solvetime
        self.model.optimize()
        
        #Splitting Criteria
        self.a_matrix = np.zeros((self.p, len(self.TB)))
        for i in range(len(self.TB)):
            for j in range(self.p):
                self.a_matrix[j, i] = self.a[j, self.TB[i]].X  
        self.b_matrix = np.zeros(len(self.TB), dtype=float)
        for i in range(len(self.TB)):
            self.b_matrix[i] = self.b[self.TB[i]].X
        
        #Calculate CART accuracy:
        self.cart_accuracy_train= self.cart_accuracy()[0]
        self.cart_accuracy_test = self.cart_accuracy()[1]
        self.accuracy_train = self.compute_accuracy(self.X, self.y)
        self.accuracy_test = self.compute_accuracy(self.X_test, self.y_test)
    
    def _epsilon(self):
        epsilon = []
        for j in range(self.p):
            x_j = self.X.iloc[:, j].tolist()
            x_j.sort()
            e = []
            for i in range(self.n - 1):
                if x_j[i + 1] != x_j[i]:
                    e.append(x_j[i + 1] - x_j[i])
                else:
                    e.append(0.00001)
            epsilon.append(min(e))
        return epsilon
    
    def _Y_matrix(self):
        Y = np.full((self.n, self.K), -1, dtype=int)
        for i in range(self.n):
            Y[i, self.y[i]] = 1
        return Y
    
    def _ancestors(self):
        left_ancestors = []
        right_ancestors = []
        for t in range(1, self.T + 1):
            la_t =[]
            ra_t =[]
            tau=t
            while tau>1:
                pt = tau//2
                if tau % 2 == 0:
                    la_t.append(pt)
                else:
                    ra_t.append(pt)
                tau = pt
            la_t.sort() 
            ra_t.sort()
            left_ancestors.append(la_t)
            right_ancestors.append(ra_t)
        return left_ancestors, right_ancestors  
    
    def add_variables(self):
        self.a = self.model.addVars(self.p, self.TB, vtype=GRB.BINARY, name="a_t")
        self.b = self.model.addVars(self.TB, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="b_t")
        self.d = self.model.addVars(self.TB, vtype=GRB.BINARY, name="d_t")
        self.z = self.model.addVars(self.n, self.TL, vtype=GRB.BINARY, name="z")
        self.l = self.model.addVars(self.TL, vtype=GRB.BINARY, name="l_t")
        self.Nk = self.model.addVars(self.K, self.TL, vtype=GRB.INTEGER, name="N_kt")
        self.N = self.model.addVars(self.TL, vtype=GRB.INTEGER, name="N_t")
        self.ck = self.model.addVars(self.K, self.TL, vtype=GRB.BINARY, name="c_kt")
        self.L = self.model.addVars(self.TL, name="L_t")
    
    def warmstart(self, a, b):
        for t in self.TB:
            self.b[t].Start = b[t-1]
        for j in range(self.p):
            for t in self.TB:
                self.a[j, t].Start = 0
        for t,j in enumerate(a):
            if j != -2:
                self.a[int(j), self.TB[t]].Start = 1           
    
    def warmstart_cart(self,cart):
        feature = cart.tree_.feature
        threshold = cart.tree_.threshold
        feature_indices = []
        threshold_indices = []
        for i in range(len(feature)):
            if feature[i] != -2:
                feature_indices.append(int(feature[i]))
                threshold_indices.append(float(threshold[i]))
            else:
                if i <= 2**self.D-2:
                    feature_indices.append(-2)
                    threshold_indices.append(0)
        if len(feature_indices) < 2**self.D-1:
            for i in range(len(feature_indices), 2**self.D-1):
                feature_indices.append(-2)
                threshold_indices.append(0)
        return feature_indices, threshold_indices
    
    def add_constraints(self, model, a, b, d, z, l, Nk, N, ck, L, X_train, Y, left_ancestors, right_ancestors):
        for t in self.TB:
            model.addConstr(a.sum("*", t) == d[t], name="sum_constraint_of_ajt")
            model.addConstr(b[t] <= d[t], name="bt_constraint_dt")
        model.addConstr(gp.quicksum(d[t] for t in self.TB) == self.C)
        for t in self.TB[1:]:
            model.addConstr(d[t] <= d[t // 2], name="dt_constraint_dp(t)")
        for i in range(self.n):
            model.addConstr(z.sum(i, "*") == 1, name="sum_of_zi(t)_constraint_1")
        for t in self.TL:
            model.addConstr(z.sum("*", t) >= self.N_min * l[t], name="sum_of_zt_constraint_Nmin_lt")
            for i in range(self.n):
                model.addConstr(z[i, t] <= l[t])
            for k in range(self.K):
                model.addConstr(Nk[k, t] == 0.5 * gp.quicksum(z[i, t] * (Y[i, k] + 1) for i in range(self.n)), name=f"Nkt_{k}_{t}")
            model.addConstr(N[t] == z.sum("*", t), name=f"Nt_{t}")
            model.addConstr(l[t] == ck.sum("*", t), name=f"sum_ck_{t}")
        for t in self.TL:
            l_ancestors = left_ancestors[t - 1]
            if l_ancestors:
                for la in l_ancestors:
                    for i in range(self.n):
                        xi = X_train.iloc[i]
                        model.addConstr(
                            self.epsilon_max + gp.quicksum(a[j, la] * xi[j] for j in range(self.p)) <= b[la] + (1 + self.epsilon_max) * (1 - z[i, t]),
                            name=f"split_l_{la}_{i}_{t}"
                        )
            r_ancestors = right_ancestors[t - 1]
            if r_ancestors:
                for r in r_ancestors:
                    for i in range(self.n):
                        xi = X_train.iloc[i]
                        model.addConstr(
                            gp.quicksum(a[j, r] * xi[j] for j in range(self.p)) >= b[r] - (1 - z[i, t]),
                            name=f"split_r_{r}_{i}_{t}"
                       )
        for t in self.TL:
            model.addConstr(L[t] >= 0, name="Lt_constraint3")
            for k in range(self.K):
                model.addConstr(L[t] >= N[t] - Nk[k, t] - self.n * (1 - ck[k, t]), name="Lt_constraint1")
                model.addConstr(L[t] <= N[t] - Nk[k, t] + self.n * ck[k, t], name="Lt_constraint2") 
    
    def _generate_feature_indices(self):
        feature_indices = []
        for j,t in enumerate(self.a_matrix.T):
            if np.any(t == 1):
                feature_indices.append(j)
            else:
                feature_indices.append(-2)
        return feature_indices
    
    def compute_leaf_indices(self, X_eval):
        leaf_indices = [[] for _ in range(2**self.D)]
        for i in range(X_eval.shape[0]):
            j = 1
            x_i = X_eval.iloc[i].to_numpy()
            while j < 2**self.D:
                if np.dot(self.a_matrix[:, j - 1], x_i) < self.b_matrix[j - 1]:
                    j = 2 * j
                else:
                    j = 2 * j + 1
            leaf_indices[j - 2**self.D].append(i)
        return leaf_indices   
    
    def compute_leaf_classes(self,X_eval,y_eval):
        leaf_indices = self.compute_leaf_indices(X_eval)
        leaf_classes = []
        for leaf in leaf_indices:
            if leaf:
                labels_in_leaf = y_eval.iloc[leaf]
                predicted_class = labels_in_leaf.mode()[0]
                leaf_classes.append(predicted_class)
            else:
                leaf_classes.append(None)
        return leaf_classes   
    
    def compute_accuracy(self, X_eval, y_eval):
        D = self.D
        leaf_indices = self.compute_leaf_indices(X_eval)
        leaf_classes = self.compute_leaf_classes(self.X,self.y)
        total_misclassified = 0
        for i in range(2**D):
            indices = leaf_indices[i]
            if indices:
                actual_labels = y_eval.iloc[indices]
                predicted_label = leaf_classes[i]
                misclassified = (actual_labels != predicted_label).sum()
                total_misclassified += misclassified
        accuracy = (X_eval.shape[0] - total_misclassified) / X_eval.shape[0]
        return accuracy  
    
    def cart_accuracy(self):
        self.clf.fit(self.X, self.y)
        y_pred_gini = self.clf.predict(self.X_test)
        y_pred_gini_train = self.clf.predict(self.X)
        return accuracy_score(self.y, y_pred_gini_train),accuracy_score(self.y_test, y_pred_gini)

        
        
def OCT(data_id,D,k_folds,n=None,tuning_problem=None,alpha=None):
    '''
    To train OCT models, calibrate hyperparameters and return the final results.
    Input:
    - data_id: the data id of the dataset from UCIMLREPO
    - D: maximum depth of the tree
    - k_folds: number of folds for cross validation
    - tuning problem: 0 = calibrate alpha through cross validation, 1 = calibrate C through cross validation, None = no tuning
    - alpha: list of alpha values for tuning, if tuning_problem is 0
    Returns:
    - C_opt: optimal value of C if tuning_problem is 1
    - OCT_train_accuracy: float, the accuracy of the OCT model on the training set
    - OCT_test_accuracy: float, the accuracy of the OCT model on the test set
    - cart_accuracy_train: float, the accuracy of the CART model on the training set
    - cart_accuracy_test: float, the accuracy of the CART model on the test set
    '''
    dt = oct_tts(data_id, k_folds,n)
    df_train = dt.df_train
    df_cal = dt.df_cal
    df_test = dt.df_test
    
    if tuning_problem == 1:
        c = list(range(1,2**D))
        cal_acc = []
        for C in c:
            for i in range(k_folds):
                acc = []
                model = OCT_AT_Sub(df_train[i],df_cal[i],D,C)
                acc.append(model.accuracy_test)
            cal_acc.append(np.mean(acc))
        C_opt = c[np.argmax(cal_acc)]

        acc_in = 0
        acc_out = 0
        cart_acc_in = 0
        cart_acc_out = 0
        for i in range(k_folds):
            train = pd.concat([df_train[i], df_cal[i]], axis=0).reset_index(drop=True)
            test = df_test[i]
            model = OCT_AT_Sub(train, test, D, C_opt)
            acc_in += model.accuracy_train
            acc_out += model.accuracy_test
            cart_acc_in += model.cart_accuracy_train
            cart_acc_out += model.cart_accuracy_test
        accuracy_in = acc_in / k_folds
        accuracy_out = acc_out / k_folds
        cart_accuracy_in = cart_acc_in / k_folds
        cart_accuracy_out = cart_acc_out / k_folds
        results = {
            'C_opt': C_opt, 
            'OCT_train_accuracy': accuracy_in, 
            'OCT_test_accuracy': accuracy_out,
            'CART_train_accuracy': cart_accuracy_in,
            'CART_test_accuracy': cart_accuracy_out}
        return results
    
    elif tuning_problem == 0:
        acc_out = []
        for a in alpha:
            acc = []
            for i in range(k_folds):
                train = pd.concat([df_train[i], df_cal[i]], axis=0).reset_index(drop=True)
                model = OCT_Main(train, df_test[i], D, a)
                acc.append(model.accuracy_test)
            acc_out.append(float(np.mean(acc)))
        return acc_out
    
    elif tuning_problem is None:
        acc_in = 0
        acc_out = 0
        cart_acc_in = 0
        cart_acc_out = 0
        for i in range(k_folds):
            train = pd.concat([df_train[i], df_cal[i]], axis=0).reset_index(drop=True)
            model = bal_OCT(train, df_test[i], D)
            acc_in += model.accuracy_train
            acc_out += model.accuracy_test
            cart_acc_in += model.cart_accuracy_train
            cart_acc_out += model.cart_accuracy_test
        results = {
            'OCT_train_accuracy': acc_in / k_folds, 
            'OCT_test_accuracy': acc_out / k_folds,
            'CART_train_accuracy': cart_acc_in / k_folds,
            'CART_test_accuracy': cart_acc_out / k_folds}
        return results