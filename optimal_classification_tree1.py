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
        

class oct:
    def __init__(self,train,D,alpha,timelim=None,warmstart_a=None,warmstart_b=None, cut=None, heuristics=None, MIPfocus=None, presolve=None):
        '''
        Predermined the parameters
        '''
        self.train = train.reset_index(drop=True)  # training data
        self.X = self.train.drop(['target'],axis=1)  # features
        self.y = self.train['target'].astype(int)
        
        self.n = self.X.shape[0]  # number of samples
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
        self.feature_indices = warmstart_a
        self.threshold_indices = warmstart_b
        if self.feature_indices is not None and self.threshold_indices is not None:
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
        self.a_matrix = np.zeros((self.p, len(self.TB)), dtype=int)  # a matrix
        for i in range(len(self.TB)):
            for j in range(self.p):
                self.a_matrix[j, i] = self.a[j, self.TB[i]].X
                
        self.b_matrix = np.zeros(len(self.TB), dtype=float)  # b matrix
        for i in range(len(self.TB)):
            self.b_matrix[i] = self.b[self.TB[i]].X
            
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
        available_nodes = min(len(self.TB), len(a))
        for i in range(available_nodes):
            t = self.TB[i]
            self.b[t].Start = b[i]
            for j in range(self.p):
                if j == int(a[i]):
                    self.a[j, t].Start = 1
            
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
    
    def plot_classified_points(self, X_eval, y_eval):
        if self.p != 2:
            return ("Dataframe dimension exceeds 2D graph dimension.")
        plt.figure(figsize=(8, 6))
        for i in range(self.n):
            x = X_eval.iloc[i, 0]
            y = X_eval.iloc[i, 1]
            label = y_eval.iloc[i]
            color = 'red' if label == 1 else 'blue'
            plt.plot(x, y, marker='o', color=color)
        feature_1 = self.X.columns[0]
        feature_2 = self.X.columns[1]
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)
        plt.title("True (red/blue) vs. Leaf-Predicted (yellow/green) Labels")
        
        plt.grid(True)
        plt.show()

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


class oct_h:
    def __init__(self,train,D,alpha,timelim=None,warmstart_a=None,warmstart_b=None, cut=None, heuristics=None, MIPfocus=None, presolve=None):
        '''
        Predermined the parameters
        '''
        self.train = train.reset_index(drop=True)  # training data
        self.X = self.train.drop(['target'],axis=1)  # features
        self.y = self.train['target'].astype(int)  # target variable
        
        self.n = self.X.shape[0]  # number of samples
        self.p = self.X.shape[1]  # number of features
        self.K = len(np.unique(self.y))  # number of classes
        self.D = D  # tree depth
        self.T = 2**(D+1)-1  # number of nodes
        self.L_hat = self.y.value_counts().max() / self.n
        self.N_min = math.floor(self.n * 0.05)
        self.alpha = alpha
        self.mu = 0.005  # parameter for the objective function
        
        self.Y_matrix = self._Y_matrix()
        
        self.left_ancestors = self._ancestors()[0]  # left ancestors of each node
        self.right_ancestors = self._ancestors()[1]
        
        self.TB = list(range(1, 2**D))   # branch (internal) nodes
        self.TL = list(range(2**D, 2**(D + 1)))  # leaf nodes
      
        '''
        Gurobi Model
        '''
        self.model = gp.Model('OCT-H')
        self.add_variables()
        
        #WarmStart
        self.warmstart_a = warmstart_a
        self.warmstart_b = warmstart_b
        if self.warmstart_a is not None and self.warmstart_b is not None:
            self.warmstart(self.warmstart_a, self.warmstart_b)
        
        # Constraints
        self.add_constraints(self.model, self.a, self.a_hat, self.s, self.b, self.d,self.z, self.l, self.Nk, self.N, self.ck, self.L, self.X, self.Y_matrix, self.left_ancestors, self.right_ancestors,self.mu)
        
        # Objective function
        self.solvetime = timelim
        self.cut = cut
        self.heuristics = heuristics
        self.MIPfocus = MIPfocus
        self.presolve = presolve
        self.model.setObjective(self.L.sum('*') / self.L_hat + 0.9*gp.quicksum(gp.quicksum(self.s[i,t] for i in range(self.p)) for t in self.TB), GRB.MINIMIZE)
        if timelim is not None:
            self.model.Params.timelimit = self.solvetime
        if cut is not None:
            self.model.Params.Cuts = self.cut 
        if heuristics is not None:
            self.model.Params.Heuristics = self.heuristics 
        if MIPfocus is not None:
            self.model.Params.MIPFocus = self.MIPfocus 
        if presolve is not None:
            self.model.Params.Presolve = self.presolve
        self.model.setParam("MIPGap", 0.5)
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
        self.a = self.model.addVars(self.p, self.TB, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="a_t")
        self.a_hat = self.model.addVars(self.p, self.TB, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="a_hat")
        self.b = self.model.addVars(self.TB, vtype=GRB.CONTINUOUS, lb=-1, ub=1, name="b_t")  # shape: |TB|
        self.d = self.model.addVars(self.TB, vtype=GRB.BINARY, name="d_t")  # shape: |TB|
        self.s = self.model.addVars(self.p, self.TB, vtype=GRB.BINARY, name="s_t") #dim px|TB|
        self.z = self.model.addVars(self.n, self.TL, vtype=GRB.BINARY, name="z")  # shape: n × |TL|
        self.l = self.model.addVars(self.TL, vtype=GRB.BINARY, name="l_t")  # shape: |TL|
        self.Nk = self.model.addVars(self.K, self.TL, vtype=GRB.INTEGER, name="N_kt")  # shape: K × |TL|
        self.N = self.model.addVars(self.TL, vtype=GRB.INTEGER, name="N_t")  # shape: |TL|
        self.ck = self.model.addVars(self.K, self.TL, vtype=GRB.BINARY, name="c_kt")  # shape: K × |TL|
        self.L = self.model.addVars(self.TL, name="L_t")  # shape: |TL|
        
    def warmstart(self, a, b):
        for t_idx, t in enumerate(self.TB):
            self.b[t].Start = b[t_idx]
            for j in range(self.p):
                self.a[j, t].Start = a[j,t_idx]
                
    def add_constraints(self, model, a, a_hat, s, b, d, z, l, Nk, N, ck, L, X, Y, left_ancestors, right_ancestors, mu):
        for t in self.TB:
            model.addConstr(s.sum("*", t) >= d[t], name="sum_constraint_of_sjt")
            model.addConstr(b[t] <= d[t], name="bt_constraint1_dt")
            model.addConstr(-d[t] <= b[t], name="bt_constraint2_dt")
            model.addConstr(d[t] == 1, name="dt_constraint_d(t)")
            for i in range(self.p):
                model.addConstr(s[i, t] <= d[t], name="s_constraint1_dt")
                model.addConstr(-s[i, t] <= a[i, t], name="a_constraint1_-st")
                model.addConstr(a[i, t] <= s[i, t], name="a_constraint2_st")
            model.addConstr(a_hat.sum("*", t) <= d[t], name="sum_constraint_of_a_hat")
            for i in range(self.p):
                model.addConstr(a_hat[i, t] >= -a[i, t], name="a_hat_constraint_-a")
                model.addConstr(a_hat[i, t] >= a[i, t], name="a_hat_constraint_a")
        for t in self.TB[1:]:
            model.addConstr(d[t] <= d[t // 2], name="dt_constraint_dp(t)")
        for i in range(self.n):
            model.addConstr(z.sum(i, "*") == 1, name="sum_of_zi(t)_constraint_1")
        for t in self.TL:
            model.addConstr(z.sum("*", t) >= self.N_min * l[t], name="sum_of_zt_constraint_Nmin_lt")
            for i in range(self.n):
                model.addConstr(z[i, t] <= l[t])
            for k in range(self.K):
                model.addConstr(Nk[k, t] == 0.5 * gp.quicksum(z[i, t] * (Y[i, k] + 1) for i in range(self.n)))
            model.addConstr(N[t] == z.sum("*", t))
            model.addConstr(l[t] == ck.sum("*", t), name="sum_of_ckt_eq_lt")
            model.addConstr(l[t] == 1, name="dt_constraint_l(t)")
        for t in self.TL:
            l_ancestors = left_ancestors[t - 1]
            if l_ancestors:
                for la in l_ancestors:
                    for i in range(self.n):
                        xi = X.iloc[i]
                        model.addConstr(
                            gp.quicksum(a[j, la] * xi[j] for j in range(self.p)) + mu <= b[la] + (2 + mu) * (1 - z[i, t]),
                            name=f"split_l_{la}_{i}_{t}"
                            )
            r_ancestors = right_ancestors[t - 1]
            if r_ancestors:
                for ra in r_ancestors:
                    for i in range(self.n):
                        xi = X.iloc[i]
                        model.addConstr(
                            gp.quicksum(a[j, ra] * xi[j] for j in range(self.p)) >= b[ra] - 2 * (1 - z[i, t]),
                            name=f"split_r_{ra}_{i}_{t}"
                        )
        for t in self.TL:
            model.addConstr(L[t] >= 0, name="Lt_constraint3")
            for k in range(self.K):
                model.addConstr(L[t] >= N[t] - Nk[k, t] - self.n * (1 - ck[k, t]), name="Lt_constraint1")
                model.addConstr(L[t] <= N[t] - Nk[k, t] + self.n * ck[k, t], name="Lt_constraint2")

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

    def compute_leaf_classes(self, X_eval,y_eval):
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
    
    def build_leaf_labels(self, X_eval, y_eval):
        leaf_classes = self.compute_leaf_classes(X_eval, y_eval)
        return {t: label for t, label in zip(self.TL, leaf_classes)}
    
    def plot_tree_structure(self, X_eval, y_eval):
        tree_structure = {t: [2*t, 2*t + 1] for t in self.TB if 2*t in self.TB + self.TL}
        leaf_labels = self.build_leaf_labels(X_eval, y_eval)
        dot = Digraph()
        for node_id in self.TB:
            t_idx = self.TB.index(node_id)
            a_vec = self.a_matrix[:, t_idx]
            b_val = self.b_matrix[t_idx]
            terms = [f"{a_vec[j]:+.2f}{self.X.columns[j]}" for j in range(self.p) if abs(a_vec[j]) > 1e-6]
            lhs_expr = " ".join(terms).replace(" +", " + ").replace(" -", " - ")
            node_label = f"{lhs_expr} < {b_val:.2f}" if lhs_expr else f"< {b_val:.2f}"
            dot.node(str(node_id), node_label)
        for leaf in self.TL:
            class_label = leaf_labels.get(leaf, "?")
            dot.node(str(leaf), f"Class: {class_label}", shape='box', style='filled', fillcolor='lightgrey')
        for node_id, children in tree_structure.items():
            dot.edge(str(node_id), str(children[0]), label="Yes")
            dot.edge(str(node_id), str(children[1]), label="No")
        dot.render("octh_tree_structure", format="png", cleanup=True)
        dot.view()