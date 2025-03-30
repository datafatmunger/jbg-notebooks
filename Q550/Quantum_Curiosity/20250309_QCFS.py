#!/usr/bin/env python
# coding: utf-8

# #### 1. Importing all the libraries

# In[1]:


import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection

import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml

from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

import pennylane as qml

import time


# #### 2. Define the Functions

# In[2]:


# Function that create the episode data - sample randomaly
def get_data(episode_size,policy,mode):
    global dataset
    if mode=='train':
        if policy==0:
             dataset=data.sample(n=episode_size)
        else:
            dataset=data
    else:
        dataset = pd.read_csv(location + '/' + file +'_test_int.csv', index_col=0)
    return dataset


# In[3]:


# Function that separate the episode data into features and label
def data_separate (dataset):
    global X
    global y    
    X = dataset.iloc[:,0:dataset.shape[1]-1]  # all rows, all the features and no labels
    y = dataset.iloc[:, -1]  # all rows, label only
    return X,y


# In[4]:


# Function that split the episode data into train and test
def data_split(X,y):
    global X_train_main
    global X_test_main   
    global y_train
    global y_test  
    from sklearn.model_selection import train_test_split
    X_train_main, X_test_main, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    return X_train_main, X_test_main, y_train, y_test


# In[5]:


# Function that chooses exploration or explotation method
def exploration_explotation(epsilon):
    global exploration 
    if np.random.rand() < epsilon:  
        exploration=1
    else:
        exploration=0    
    return exploration


# In[6]:


# Function that returns all available actions in the state given as an argument: 
def available_actions(number_of_columns,columns,initial_state,current_state,trashold, exploration):
    global exclude
    global all_columns
#    exclude=[]
    all_columns=np.arange(number_of_columns+1)
    # remove columns that have been already selected
    exclude=columns.copy()
    # remove the initial_state and the current_state
    exclude.extend([initial_state, current_state])
    available_act = list(set(all_columns)-set(exclude))
    # remove actions that have negetiv Q value
    if exploration==0:
        index = np.where(Q[current_state,available_act] > trashold)[1]
        available_act= [available_act[i] for i in index.tolist()]
    return available_act


# In[7]:


def sample_next_action(current_state, Q, available_act, exploration):
    global available_act_q_value
    available_act_q_value = [float(q) for q in np.array(Q[current_state, available_act]).reshape(-1)]
    
    if exploration == 1: 
        # Random selection
        next_action = int(np.random.choice(available_act, 1).item())
    else: 
        # Greedy selection according to max value
        maxQ = max(available_act_q_value)
        count = available_act_q_value.count(maxQ)
        
        if count > 1:
            max_columns = [i for i in range(len(available_act_q_value)) if available_act_q_value[i] == maxQ]
            i = int(np.random.choice(max_columns, 1).item())
        else:
            i = available_act_q_value.index(maxQ)
        
        next_action = available_act[i]  
    
    return next_action


# In[8]:


# function that update a list with all selected columns in the episode
def update_columns(action, columns):
    update_columns=columns
    update_columns.append(action)
    return update_columns


# In[9]:


# function that update the X_train and X_test according to the current episode columns list 
def update_X_train_X_test(columns,X_train_main, X_test_main):
    X_train=X_train_main.iloc[:,columns]
    X_test=X_test_main.iloc[:,columns]
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    return X_train, X_test


# In[10]:


# Function that run the learner and get the error to the current episode columns list
def Learner(X_train, X_test,y_train, y_test):
    global learner
    global y_pred
    if learner_model == 'DT':
        learner = tree.DecisionTreeClassifier()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)
    elif learner_model == 'KNN':
        learner = KNeighborsClassifier(metric='hamming',n_neighbors=5)
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)        
    elif learner_model == 'SVM':
        learner = SVC()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)        
    elif learner_model == 'NB':
        learner = MultinomialNB()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)
    elif learner_model == 'AB':
        learner = AdaBoostClassifier()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)  
    elif learner_model == 'GB':
        learner = GradientBoostingClassifier()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)  
    elif learner_model == 'VQC':
        learner = QuantumLearner()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)  
    elif learner_model == 'ANN':
        learner = ClassicalLearner()
        learner = learner.fit(X_train, y_train)
        y_pred = learner.predict(X_test)  
    accuracy=metrics.accuracy_score(y_test, y_pred)
    error=1-accuracy
    return error


# In[11]:


def q_update(current_state, action, learning_rate, reward):
    # next_state = current action
    max_index = np.where(Q[action,] == np.max(Q[action,]))[0]  # Use [0] instead of [1] for 1D arrays
    
    if max_index.shape[0] > 1:
        # Resolve tie by selecting one randomly
        max_index = int(np.random.choice(max_index, size=1).item())
    else:
        max_index = int(max_index[0])  # Convert the first element to a scalar

    max_value = Q[action, max_index]

    # Update the Q matrix
    if Q[current_state, action] == 1:
        Q[current_state, action] = learning_rate * reward
    else:
        Q[current_state, action] = Q[current_state, action] + learning_rate * (
            reward + (discount_factor * max_value) - Q[current_state, action]
        )


# ### Experiment mangment

# #### 3. Define the parameters 

# In[17]:


## for run time ##
N_features=5
N_data=1
## for run time ##

#Experiment: 
experiment='test'
number_of_experiment=1

# Dataset parameters #
location = 'Datasets/adult'
outputlocation='Datasets'
file='adult' #adult #diabetic_data #no_show
#np.random.seed(3)

# Q learning parameter # 
learning_rate=0.005
discount_factor = 0.01 #0
epsilon = 0.1

# Learner and episode parameters #
learner_model = 'VQC' #DT #KNN #SVM
episode_size=10
internal_trashold=0
external_trashold=0
filename= file +'_int.csv'

#Experiments folder management: 
#if not os.path.exists('/Experiments'):
#    os.makedirs('/Experiments') 
if not os.path.exists('Experiments/'+ str(experiment)):
    os.makedirs('Experiments/'+ str(experiment))
else:
    shutil.rmtree('Experiments/'+ str(experiment))          #removes all the subdirectories!
    os.makedirs('Experiments/'+ str(experiment))
#writer = pd.ExcelWriter('Experiments/'+ str(experiment) + '/df.xlsx') 



text_file = open('Experiments/'+ str(experiment) +'/parameters.txt', "w")
text_file.write('experiment: ' + str(experiment)+ '\n')
text_file.write('number of experiments: ' + str(number_of_experiment)+ '\n')
text_file.write('file: ' + str(file)+ '\n')
text_file.write('learner model: ' + str(learner_model)+ '\n')
text_file.write('episode size: ' + str(episode_size)+ '\n')
#text_file.write('numbers of epocs: ' + str(epocs)+ '\n')
text_file.write('internal trashold: ' + str(internal_trashold)+ '\n')
text_file.write('external trashold: ' + str(external_trashold)+ '\n')
 
text_file.close()


# In[18]:


# Classical Learner based on a simple ANN
class ClassicalLearner(nn.Module):
    def __init__(self, num_layers=2, hidden_size=5):
        super().__init__()
        self.layers = None
        self.sigmoid = nn.Sigmoid()

    def initialize_layers(self, input_size, num_layers, hidden_size=5):
        layers = [input_size] + [hidden_size] * (num_layers - 1) + [1]
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1], dtype=torch.float64) for i in range(len(layers) - 1)])
    
    def forward(self, x):
        x = x.to(torch.float64)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.sigmoid(self.layers[-1](x))
    
    def fit(self, X_train, y_train, num_it=50, lr=0.01):
        input_size = X_train.shape[1]
        self.initialize_layers(input_size, num_layers=2)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        y_train = torch.tensor(y_train.values, dtype=torch.float64).reshape(-1, 1)
        
        for epoch in range(num_it):
            optimizer.zero_grad()
            y_pred = self.forward(torch.tensor(X_train.values, dtype=torch.float64)).reshape(-1, 1)
            loss = nn.BCELoss()(y_pred, y_train)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X_test):
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float64)
            y_pred = self.forward(X_test_tensor).reshape(-1, 1)
            return (y_pred.numpy().flatten() > 0.5).astype(int)


# In[19]:


# Quantum Learner based on Benedetti et al.
class QuantumLearner:
    def __init__(self, num_layers=2):
        self.num_layers = num_layers
        self.weights = None
        self.bias = None
        self.num_qubits = None
        self.dev = None
        self.opt = qml.optimize.AdamOptimizer(0.05)

    def _initialize_circuit(self, num_features):
        # Update the number of qubits to match the current feature count.
        self.num_qubits = num_features
        #self.dev = qml.device('qiskit.aer', wires=list(range(self.num_qubits)))
        self.dev = qml.device("default.qubit", wires=list(range(self.num_qubits)))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            self.feature_encoding(x)
            for W in weights:
                self.variational_layer(W)
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = circuit

    def feature_encoding(self, x):
        for i in range(self.num_qubits):
            qml.RY(np.pi * x[i], wires=i)
        for i in range(self.num_qubits - 1):
            qml.CZ(wires=[i, i + 1])

    def variational_layer(self, W):
        for i in range(self.num_qubits):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        if self.num_qubits > 1:
            qml.CNOT(wires=[self.num_qubits - 1, 0])

    def variational_classifier(self, weights, bias, x):
        return self.circuit(weights, x) + bias

    def cost(self, weights, bias, X, Y):
        # X is expected to be a NumPy array.
        predictions = qml.numpy.array([self.variational_classifier(weights, bias, x) for x in X])
        return qml.numpy.mean((qml.numpy.array(Y) - predictions) ** 2)

    def fit(self, X_train, y_train, num_it=10, batch_size=48, warm_start=False):
        """
        Train the QuantumLearner with debug logging to track performance bottlenecks.
        """
        print("🚀 [QuantumLearner] Starting fit function...")
    
        # Convert inputs to NumPy arrays if needed.
        if hasattr(X_train, "values"):
            X_train = X_train.values.astype(np.float64)
        else:
            X_train = np.array(X_train, dtype=np.float64)
        if hasattr(y_train, "values"):
            y_train = y_train.values
        else:
            y_train = np.array(y_train)
    
        print(f"📊 Dataset size: X_train={X_train.shape}, y_train={y_train.shape}")
    
        current_features = X_train.shape[1]
    
        # If not warm starting or if the number of features has changed, reinitialize the circuit.
        if not warm_start or (self.num_qubits is None) or (self.num_qubits != current_features):
            print("🔄 Reinitializing circuit...")
            self._initialize_circuit(current_features)
            np.random.seed(0)
            self.weights = qml.numpy.tensor(
                0.01 * np.random.randn(self.num_layers, self.num_qubits, 3),
                requires_grad=True,
            )
            self.bias = qml.numpy.tensor(0.0, requires_grad=True)
            print(f"✅ Initialized circuit with {self.num_qubits} qubits")
    
        batch_size = min(batch_size, len(X_train))
        print(f"📌 Using batch size: {batch_size}")
    
        # Track total training time
        start_time = time.time()
    
        for it in range(num_it):
            #print(f"⏳ Epoch {it+1}/{num_it}")
    
            batch_start = time.time()
            batch_index = np.random.choice(len(X_train), batch_size, replace=False)
            X_batch = X_train[batch_index]
            Y_batch = y_train[batch_index]
            batch_time = time.time() - batch_start
            #print(f"🛠 Batch selection completed in {batch_time:.2f} seconds")
    
            # Time the optimization step
            opt_start = time.time()
            self.weights, self.bias = self.opt.step(
                lambda w, b: self.cost(w, b, X_batch, Y_batch), self.weights, self.bias
            )
            opt_time = time.time() - opt_start
            #print(f"✅ Optimization step completed in {opt_time:.2f} seconds")
    
        total_time = time.time() - start_time
        print(f"🚀 Training completed in {total_time:.2f} seconds")
    
        return self


    def predict(self, X_test):
        if hasattr(X_test, "values"):
            X_test = X_test.values.astype(np.float64)
        else:
            X_test = np.array(X_test, dtype=np.float64)
        return np.array([
            float(qml.numpy.sign(self.variational_classifier(self.weights, self.bias, x)))
            for x in X_test
        ])


# In[20]:


class QuantumFeatureSelection:
    def __init__(self, num_features, learning_rate=0.1):
        self.num_features = num_features  # Number of possible features
        self.learning_rate = learning_rate  # Learning rate
        #self.device = qml.device("lightning.gpu", wires=self.num_features)
        self.device = qml.device("default.qubit", wires=self.num_features)
        self.state = self.initialize_state()

    def initialize_state(self):
        """Initialize the quantum state as an equal superposition over all features."""
        @qml.qnode(self.device)
        def circuit():
            for qubit in range(self.num_features):
                qml.Hadamard(wires=qubit)  # Create equal superposition of all features
            return qml.probs(wires=range(self.num_features))  # Get probability distribution
        
        state_vector = np.array(circuit())  # Convert to NumPy array
        state_vector = state_vector[:self.num_features]  # Extract relevant part
        state_vector /= state_vector.sum()  # Normalize to ensure probability sum is 1
        return state_vector

    def measure(self, available_features):
        """Measure the quantum state and return a valid feature (action)."""
        probabilities = self.state  # Already a probability distribution
    
        if not available_features:
            raise ValueError("No available features for selection.")
    
        # Only consider the probabilities of available features
        valid_probs = np.array([probabilities[i] for i in available_features], dtype=np.float64)
    
        print(f"🔎 Available features: {available_features}")
        print(f"🧐 Raw valid_probs before normalization: {valid_probs}")
    
        # Normalize the probabilities
        prob_sum = valid_probs.sum()
        if prob_sum == 0:
            print("⚠️ Warning: All probabilities are zero, using uniform distribution.")
            valid_probs = np.ones(len(available_features), dtype=np.float64) / len(available_features)
        else:
            valid_probs /= prob_sum
    
        print(f"✅ Valid probabilities after normalization: {valid_probs}")
    
        # Ensure arrays have matching lengths
        if len(valid_probs) != len(available_features):
            raise ValueError(f"Size mismatch: available_features={len(available_features)}, valid_probs={len(valid_probs)}")
    
        action = np.random.choice(available_features, p=valid_probs)
        return action

    def unitary_update(self, action, reward):
        """Update the quantum state using a reinforcement learning-inspired unitary operation."""
        if len(self.state.shape) > 1:
            self.state = self.state.flatten()
    
        # Ensure the state is float64
        self.state = self.state.astype(np.float64)
    
        U = np.eye(self.num_features, dtype=np.float64)  # Identity matrix as float64
    
        if reward > 0:
            U[action, action] += self.learning_rate  # Reinforce good actions
        else:
            U[action, action] -= self.learning_rate  # Reduce weight for bad actions
    
        # Apply transformation in probability space
        new_state = U @ self.state  # Matrix-vector multiplication
        new_state = np.abs(new_state)  # Ensure non-negative values
    
        # Avoid division by zero in normalization
        state_sum = np.sum(new_state, dtype=np.float64)
        if state_sum == 0:
            print("Warning: Normalization issue in unitary_update. Resetting state.")
            new_state = np.full(self.num_features, 1 / self.num_features, dtype=np.float64)
        else:
            new_state /= state_sum  # Normalize to maintain probabilities
    
        self.state = new_state.astype(np.float64)  # Ensure final state is float64




# #### 4. Run all experiments

# In[21]:


for e in range (number_of_experiment):
    if not os.path.exists('Experiments/'+ str(experiment)+ '/'+ str(e)):
        os.makedirs('Experiments/'+ str(experiment)+ '/'+ str(e))
    else:
        shutil.rmtree('Experiments/'+ str(experiment)+ '/'+ str(e))          #removes all the subdirectories!
        os.makedirs('Experiments/'+ str(experiment)+ '/'+ str(e))
    print ('Experiments ' + str(e) + ' start')
##########################Experiment setup##########################
    # Read the data
    data = pd.read_csv(location + '/' + filename, index_col=0)
    
##### for run time - start #####
    import timeit
    start = timeit.default_timer()
    size= int(N_data* len(data.index))
    data = data.sample(n=size)
    data=data.iloc[:,-N_features-1:]
##### for run time - end #####
    
    #Set the number of iterations:
    interations=10*len(data.index)/episode_size
    # Set the number of columns exclude the class column
    number_of_columns=data.shape[1]-1 
    print ("number of columns: "+ str(number_of_columns) +" (exclude class column)" ) 
    # Set the number of episodes 
    # episodes_number=epocs*len(data.index)/episode_size
    episodes_number=interations
    print ("Number of episodes: "+ str(episodes_number) ) 
    # Initialize matrix Q as a 1 values matrix:
    #Q = np.matrix(np.ones([number_of_columns+1,number_of_columns+1])) # we will use the last dummy columns as initial state s
    Q = np.matrix(np.ones([number_of_columns+1,number_of_columns+1])) # we will use the last dummy columns as initial state s
    # Set initial_state to be the last dummy column we have created
    initial_state=number_of_columns
    # define data frame to save episode policies results
    df = pd.DataFrame(columns=('episode','episode_columns','policy_columns','policy_accuracy_train','policy_accuracy_test'))
    print ("initial state number: "+ str(initial_state) + " (the last dummy column we have created)") 

    ##########################  episode  ##########################  
    for i in range (int(episodes_number)):
    ########## Begining of episode  ############
        # Initiate lists for available_act, episode_columns and and the policy mode & episode_error
        episode_available_act=list(np.arange(number_of_columns))
        episode_columns=[]
        policy=0
        episode_error=0
        # Initiate the error to 0.5
        episode_last_error=0.5
        # Initiate current_state to be initial_state
        episode_current_state=initial_state
        # Create the episode data 
        episode= get_data(episode_size, policy=0, mode='train')
        # Separate the episode data into features and label
        X_episode,y_episode=data_separate(episode)
        # Split the data into train and test 
        X_train_main_episode, X_test_main_episode, y_train_episode, y_test_episode = data_split(X_episode,y_episode)
        if i<episodes_number*0.25:
            epsilon=0.9
            learning_rate=0.09
        elif i<episodes_number*0.5:
            epsilon=0.5
            learning_rate=0.05
        elif i<episodes_number*0.75:
            epsilon=0.3
            learning_rate=0.01
        else:
            epsilon=0.1
            learning_rate=0.005
        ########## Q learning start ############

        # Initialize Quantum RL feature selection
        quantum_rl = QuantumFeatureSelection(num_features=number_of_columns)
        
        # While there are features left to select
        # while len(episode_available_act) > 0:
        #     # Select next feature using quantum measurement
        #     episode_action = quantum_rl.measure(episode_available_act)
        
        #     # Update the selected feature list
        #     episode_columns = update_columns(episode_action, episode_columns)
        
        #     # Prepare training dataset with selected features
        #     X_train_episode, X_test_episode = update_X_train_X_test(episode_columns, X_train_main_episode, X_test_main_episode)
        
        #     # Evaluate the model accuracy with the selected features
        #     episode_error = Learner(X_train_episode, X_test_episode, y_train_episode, y_test_episode)
        
        #     # Compute reward based on improvement
        #     episode_reward = episode_last_error - episode_error
        
        #     # Quantum RL update step
        #     quantum_rl.unitary_update(episode_action, episode_reward)
        
        #     # Move to next state (selected feature)
        #     episode_current_state = episode_action
        #     episode_last_error = episode_error


        while len(episode_available_act) > 0:
            print(f"\n🟢 Available features: {episode_available_act}")
        
            # Determine exploration vs. exploitation
            exploration = exploration_explotation(epsilon)  # 🔥 Fix: Define exploration
        
            # Update the available actions list
            episode_available_act = available_actions(number_of_columns, episode_columns, initial_state, episode_current_state, internal_trashold, exploration)
            
            # If no available actions remain, terminate the loop
            if len(episode_available_act) == 0:
                print("❌ No available actions left. Terminating episode.")
                break
        
            # Select next feature using quantum measurement
            episode_action = quantum_rl.measure(episode_available_act)
            print(f"🎯 Selected feature: {episode_action}")
        
            # Update the selected feature list
            episode_columns = update_columns(episode_action, episode_columns)
            print(f"📌 Updated selected features: {episode_columns}")
        
            # Prepare training dataset with selected features
            X_train_episode, X_test_episode = update_X_train_X_test(
                episode_columns, X_train_main_episode, X_test_main_episode
            )
            print(f"📊 Training set shape: {X_train_episode.shape}, Test set shape: {X_test_episode.shape}")

            # Evaluate the model accuracy with the selected features
            episode_error = Learner(X_train_episode, X_test_episode, y_train_episode, y_test_episode)
            print(f"📉 Model error after selection: {episode_error}")
        
            # Compute reward based on improvement
            episode_reward = episode_last_error - episode_error
            print(f"🏆 Computed Reward: {episode_reward}")
        
            # Quantum RL update step
            print(f"Quantum state before update: {quantum_rl.state}")
            quantum_rl.unitary_update(episode_action, episode_reward)
            print(f"🔄 Updated Quantum State: {quantum_rl.state}")
        
            # Move to next state (selected feature)
            episode_current_state = episode_action
            episode_last_error = episode_error

        print("Q learning End.")
        ########## Q learning End ############

        #Save Q matrix: 
        if (i%100 ==0):
            Q_save=pd.DataFrame(Q)
            Q_save.to_csv('Experiments/'+ str(experiment)+ '/'+ str(e)+ '/Q.'+ str(i+1) + '.csv') 
            
        print("Calculating policy...")
        # Calculate policy 
        policy_available_actions=list(np.arange(number_of_columns))
        policy_columns=[]
        policy_current_state=initial_state
        while len(policy_available_actions)>0:
            # Get available actions in the current state
            policy_available_actions = available_actions(number_of_columns,policy_columns,initial_state,policy_current_state, external_trashold, exploration=0)
            # # Sample next action to be performed
            if len(policy_available_actions)>0:
                policy_select_action = sample_next_action(policy_current_state, Q, policy_available_actions, exploration=0)
                # Update the episode_columns
                policy_columns=update_columns(policy_select_action,policy_columns)
                policy_current_state=policy_select_action

        print("Calculating policy_accuracy...")
        # Calculate policy_accuracy    
        if len(policy_columns)>0:
            ##for training dataset##
            policy_data=get_data(episode_size,policy=1,mode='train')
            X_policy,y_policy=data_separate(policy_data)
            X_train_main_policy, X_test_main_policy, y_train_policy, y_test_policy = data_split(X,y)
            X_train_policy, X_test_policy =update_X_train_X_test(policy_columns, X_train_main_policy, X_test_main_policy)
            

            # Print dataset shape before training to check if it's too large
            print(f"📊 Policy dataset shape: Train={X_train_policy.shape}, Test={X_test_policy.shape}")
            
            # Time the execution of the Learner function
            print("⏳ Running Learner...", X_train_policy, X_test_policy, y_train_policy, y_test_policy)
            start_time = time.time()
            policy_error = Learner(X_train_policy, X_test_policy, y_train_policy, y_test_policy)
            end_time = time.time()
            
            # Print execution time
            print(f"✅ Learner execution time: {end_time - start_time:.2f} seconds")

            
            
            
            policy_accuracy_train=1-policy_error
            ##for testing dataset##
            policy_data=get_data(episode_size,policy=1,mode='test')
            X_policy,y_policy=data_separate(policy_data)
            X_train_main_policy, X_test_main_policy, y_train_policy, y_test_policy = data_split(X,y)
            X_train_policy, X_test_policy =update_X_train_X_test(policy_columns, X_train_main_policy, X_test_main_policy)
            policy_error=Learner(X_train_policy, X_test_policy,y_train_policy, y_test_policy)
            policy_accuracy_test=1-policy_error 
        else:
            policy_accuracy_train=0 
            policy_accuracy_test=0
        #df=df.append({'episode':str(i+1), 'episode_columns':str(episode_columns),'policy_columns':str(policy_columns),'policy_accuracy_train':policy_accuracy_train,'policy_accuracy_test':policy_accuracy_test}, ignore_index=True)
        #new_row = pd.DataFrame([{'episode': str(i+1),
        #                  'episode_columns': str(episode_columns),
        #                  'policy_columns': str(policy_columns),
        #                  'policy_accuracy_train': policy_accuracy_train,
        #                  'policy_accuracy_test': policy_accuracy_test}])
        #df = pd.concat([df, new_row], ignore_index=True)
        df.loc[len(df)] = {
            'episode': str(i+1),
            'episode_columns': str(episode_columns),
            'policy_columns': str(policy_columns),
            'policy_accuracy_train': policy_accuracy_train,
            'policy_accuracy_test': policy_accuracy_test
        }

        #Prints
        print ("episode "+ str(i+1) +" start") 
        print ("episode columns: "+ str(episode_columns) + " epsilon: " + str(epsilon) + " learning rate: " + str(learning_rate) + " error: " +str(episode_error))
        print ("episode policy:" + str(policy_columns) + " train accuracy: " + str(policy_accuracy_train)  + " test accuracy: " +str(policy_accuracy_test)) 
        print ("episode "+ str(i+1) +" end") 
    ########## End of episode  ############
    #df.to_excel(writer, 'Experiment' + str(e))
    df.to_excel(writer, sheet_name='Experiment' + str(e))
    df_plot=df[['episode','policy_accuracy_train','policy_accuracy_test']]
    plot=df_plot.plot()
    fig = plot.get_figure()
    fig.savefig('Experiments/'+ str(experiment) + '/plot_experiment_' + str(e) +'.png')
    
#writer.save()
with pd.ExcelWriter('Experiments/'+ str(experiment) + '/df.xlsx') as writer:
    df.to_excel(writer, sheet_name='Experiment' + str(e))

## for run time ##
stop = timeit.default_timer()
print (stop - start)
## for run time ##


# In[ ]:


import numpy as np

def test_quantum_feature_selection():
    num_features = 4
    learning_rate = 0.1
    qfs = QuantumFeatureSelection(num_features, learning_rate)

    # Test initialization
    assert qfs.state.shape == (num_features,), "State shape is incorrect."
    assert np.isclose(qfs.state.sum(), 1.0), "State is not normalized."

    # Test measurement
    available_features = [0, 1, 2, 3]
    for _ in range(10):
        action = qfs.measure(available_features)
        assert action in available_features, f"Invalid action selected: {action}"

    # Test unitary update
    initial_state = qfs.state.copy()
    qfs.unitary_update(action, reward=1)
    assert np.isclose(qfs.state.sum(), 1.0), "State is not normalized after update."
    assert not np.allclose(qfs.state, initial_state), "State did not change after update."

    # Edge case: No available features
    try:
        qfs.measure([])
        assert False, "Expected ValueError when measuring with no available features."
    except ValueError:
        pass  # Expected behavior

    # Edge case: Unitary update does not lead to zero state
    qfs.unitary_update(action, reward=-1)
    assert np.isclose(qfs.state.sum(), 1.0), "State is not normalized after negative reward update."

    print("All tests passed!")

# Run the test
test_quantum_feature_selection()


# In[48]:


def test_quantum_learner():
    """Test the QuantumLearner class with basic training and prediction."""
    np.random.seed(42)

    # Sample training data (binary classification, {-1, 1} labels)
    X_train = np.array([
        [0.1, 0.2], 
        [0.2, 0.3], 
        [0.3, 0.4], 
        [0.4, 0.5]
    ])
    y_train = np.array([1, -1, 1, -1])  # Binary labels

    # Initialize QuantumLearner
    ql = QuantumLearner(num_layers=2)

    # Ensure the circuit initializes correctly
    ql._initialize_circuit(num_features=2)
    assert ql.num_qubits == 2, "Circuit initialization failed: Incorrect number of qubits."

    # Test fitting the model
    ql.fit(X_train, y_train, num_it=5)  # Small number of iterations for quick test

    # Ensure the weights and bias are updated
    assert ql.weights is not None, "Weights were not initialized."
    assert ql.bias is not None, "Bias was not initialized."

    # Test prediction on new samples
    X_test = np.array([
        [0.15, 0.25],
        [0.35, 0.45]
    ])
    predictions = ql.predict(X_test)
    
    # Ensure output shape matches input shape
    assert predictions.shape == (X_test.shape[0],), "Prediction output shape mismatch."

    print("✅ All tests passed successfully!")

# Run the test
test_quantum_learner()


# In[49]:


def test_quantum_learner_multiple_episodes():
    """Test the QuantumLearner class with multiple episodes to observe improvement."""
    np.random.seed(42)

    # Sample training data (binary classification, {-1, 1} labels)
    X_train = np.array([
        [0.1, 0.2], 
        [0.2, 0.3], 
        [0.3, 0.4], 
        [0.4, 0.5]
    ])
    y_train = np.array([1, -1, 1, -1])  # Binary labels

    # Initialize QuantumLearner
    ql = QuantumLearner(num_layers=2)

    # Ensure the circuit initializes correctly
    ql._initialize_circuit(num_features=2)
    assert ql.num_qubits == 2, "Circuit initialization failed: Incorrect number of qubits."

    # Parameters for multiple episodes
    num_episodes = 10
    epsilon = 0.9  # Initial exploration rate
    epsilon_decay = 0.1  # Decay rate for epsilon
    learning_rate = 0.1
    rewards = []

    for episode in range(num_episodes):
        print(f"\n🔄 Episode {episode + 1} start")

        # Simulate feature selection process
        selected_features = []
        available_features = list(range(X_train.shape[1]))
        last_error = 0.5  # Initialize error

        while available_features:
            # Exploration vs. exploitation
            explore = np.random.rand() < epsilon
            if explore:
                action = np.random.choice(available_features)  # Randomly select a feature
                print(f"🧭 Exploration: Selected feature {action}")
            else:
                action = available_features[0]  # Exploit (select the first available feature)
                print(f"📈 Exploitation: Selected feature {action}")

            # Update selected features and remove from available features
            selected_features.append(action)
            available_features.remove(action)

            # Simulate training with selected features
            X_train_selected = X_train[:, selected_features]
            ql.fit(X_train_selected, y_train, num_it=5)  # Train the learner
            error = 1 - np.mean(ql.predict(X_train_selected) == y_train)  # Calculate error
            reward = last_error - error  # Calculate reward
            rewards.append(reward)
            last_error = error

            print(f"🎯 Features: {selected_features}, Error: {error:.4f}, Reward: {reward:.4f}")

        # Decay epsilon to reduce exploration over time
        epsilon = max(0.1, epsilon - epsilon_decay)
        print(f"🔽 Epsilon after decay: {epsilon:.4f}")

        print(f"✅ Episode {episode + 1} end")

    # Check if rewards improve over episodes
    assert len(rewards) > 0, "No rewards were calculated."
    assert rewards[-1] > rewards[0], "Rewards did not improve over episodes."

    print("\n✅ All tests passed successfully!")

# Run the test
test_quantum_learner_multiple_episodes()


# In[ ]:




