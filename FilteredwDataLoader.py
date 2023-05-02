import torch
from torch.autograd import Variable
import torch.utils.data as utils_data
from EarlyStop import EarlyStopping
import pandas as pd
from pandas import DataFrame as DF
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
import math as math
import scipy.special
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
import optuna
from Network import Net3
import shap

# define number of input features and output size for the model
input   = 28
output  = 1

"""This function plots the SmoothL1Loss per epoch for both training 
and validation dataset"""

def plot_loss(TE, VE):
    fig = plt.figure()  
    #plt.xscale('log')   
    plt.xlabel('epochs')  
    plt.plot(TE, label = "Transformed_Training Errors", color='blue')   
    plt.plot(VE, label = "Transformed_Validation Errors", color='green') 
    plt.legend()    
    plt.ylabel('SmoothL1Loss')  
    
    plt.show()

"""This function plots the quantile transformed validation predictions
from the model vs. quantile transfomed actual values. Quantile transformation
of validation dataset happens in FilteredInputs.py file"""

def plot_PredvsaActual(Predicted, Actual):
    fig = plt.figure()  
    #plt.xscale('log')   
    plt.xlabel('num_examples')     
    plt.plot(Predicted, label = "Transformed_Predicted", color='red')
    plt.plot(Actual, label= "Transformed_Target", color = 'green')    
    plt.legend()    
    plt.ylabel('Heat Unit')  
    
    plt.show()
   
"""This function gets the input of predicted outputs from validation set,
and actual labels for val set. These inputs are already quantile transformed.
This function bins the predicted and actual into 4 bins of heat as commented below.
Since the inputs are transformed therefore the bin edges are also transformed before
comparison"""

def rankbin (Y):

    ranked = []
    bin_edges = np.array([25,50,200])
    bin_edges = bin_edges.reshape(-1,1)

    #Transform the bin edges with quantile_transform
    qt = QuantileTransformer(n_quantiles=3, output_distribution='normal')

    trans_binedges = qt.fit_transform (bin_edges)
    
    for i in range (len(Y)):
        # Low 
        if Y[i] <= trans_binedges[0]:
            ranked.append(1)
        #transition
        elif Y[i] > trans_binedges[0] and Y[i] <= trans_binedges[1]:
            ranked.append(2)
        #high
        elif Y[i] > trans_binedges[1] and Y[i] <= trans_binedges[2]:
            ranked.append(3)
        #Very High
        elif Y[i] > trans_binedges[2]:
            ranked.append(4)
    
    return np.asarray(ranked)

"""This function reads the train and validation dataset which were produces by FilteredInputs.py.
It extracts the last column of the files to get all the features and saves the last column as labels. 
It then convers the features and labels for train and val to numpy arrays and returns them."""

def PrepareInputs():
    
    # Read the train and val csv
    df_train = pd.read_csv('../mp01files/filtered_data/filtered_data_zscore_train.csv', delimiter=',')  
    df_val = pd.read_csv('../mp01files/val.csv', delimiter=',')   
    df_test = pd.read_csv('../mp01files/Transformed_Test.csv', delimiter=',') 
   
    #Assign the last column to target      
    Target = df_train['hfqc_resid']  
    val_target = df_val['hfqc_resid']

    # drop last column
    df_train = df_train.drop(df_train.columns[-1], axis = 1) 
    df_val = df_val.drop(df_val.columns[-1], axis = 1)     

    # Convert labels to numpy and reshape them from 1d to 2d arrays.
    # size [622 x 1]
    target = Target.to_numpy()
    target = target.reshape(-1,1)
    valtarget = val_target.to_numpy()
    valtarget = valtarget.reshape(-1,1)      

    # Convert labels to numpy.
    # size train_in [1813 x 28]
    # 27 % training data loss due to outlier removal
    # size val_in [622 x 28]
    train_in = df_train.to_numpy() 
    val_in = df_val.to_numpy()
    test_in = df_test.to_numpy()
    
    return train_in, target, val_in, valtarget, test_in

"""This function makes an object of Net class which has the Neural Network. Net class is defined in Network.py.
This function also contains model architecture elements e.g. Loss function, optimizer etc. 
It recieves training data, val data and all hyper parameters."""

def Predict (Xtrain, Ytrain, Xval, Yval, Xtest, hidden, learning_rate, epochs, patience, batch_size):

    TE = [] 
    VE = []  
    plot = 0    
    
    # Define the model architecture
    model = Net3(input, hidden, output)
    model.double()
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    # Use DataLoader
    loaderT = DataLoader(list(zip(Xtrain,Ytrain)), shuffle=True, batch_size=batch_size)
    # Set highest batch size for validation because there is no advantage of using small batches for val
    loaderV = DataLoader(list(zip(Xval,Yval)), shuffle=True, batch_size=Yval.shape[0])
        
    # training
    for epoch in range(epochs):      
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for i, (X, Y) in enumerate(loaderT):                                            
            # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()
            # get output from the model, given the inputs
            outputs = model(X)        
            # get loss for the predicted output
            lossT = criterion(outputs, Y)               
            # get gradients w.r.t to parameters
            lossT.backward()
            # update parameters
            optimizer.step()
            train_loss += lossT.item()
        # append training loss per epoch into a list    
        train_loss /= len (loaderT)      
        TE.append((train_loss)/len(loaderT))                   

        # validation
        model.eval()
        with torch.no_grad(): # we don't need gradients in the testing phase
            for X, Y in loaderV: 

                predicted = model(X)                                 
                LossV = criterion (predicted, Y)
                val_loss += LossV.item()     
        val_loss /= len(loaderV)
        VE.append(val_loss/len(loaderV))
        
        #print (f'epoch {epoch} train_loss {round(train_loss,2)} val_loss {round(val_loss,2)}')
        
        # Prevent Overtaining by keeping an eye on validation loss changes
        early_stopping(val_loss, model)	
        if early_stopping.early_stop:
            if patience == patience:
                print(f'Early stopping at epoch {epoch}')
                break          

    # Bin predicted and labels and rank them    
    binY_hat = rankbin (predicted.numpy())
    #Use Y instead of Yval because shuffle is True for loaderV. Y is of the same size as Yval
    binY = rankbin (Y.numpy()) 

    # calculate accuracy
    correct = 0
    for i in range (len(binY)) :
        correct += (binY_hat[i] == binY[i])

    accuracy = (correct / len(binY)) * 100 
    
    L1_loss = 0
    # calculate the ordinal loss matrix
    L1_loss = np.abs (binY_hat - binY)

    if plot == 1:
    
        #Plot training and validation loss with fit lines
        plt.xlabel('epochs')  
        plt.ylabel('SmoothL1Loss')       
        x = np.arange(0, len(VE), 1)
        z = np.polyfit(x, VE, 1)
        p = np.poly1d(z)

        #add trendline to plot
        plt.scatter (x, VE, label ="Transformed_val_errors_wfitline", color="green")
        plt.plot(x, p(x))
        plt.legend()
        plt.show()    

        plt.xlabel('epochs')  
        plt.ylabel('SmoothL1Loss') 
        z = np.polyfit(x, TE, 1)
        p = np.poly1d(z)
        plt.scatter (x, TE, label ="Transformed_training_errors_wfitline", color="blue")
        plt.plot(x, p(x))
        plt.legend()
        plt.show()

        #Plot training and validation loss 
        plot_loss(TE, VE) 

        #Plot transformed prediction outputs vs. transformed labels for val set
        plot_PredvsaActual (binY_hat, binY)

    #Save the binned predictions and labels for val set in a csv
    with open ('../Results_new/ValPred.csv', 'w+') as f:
        print (f'predicted_bin, actual_bin', file=f)
        for i in range (predicted.shape[0]):
            print (f'{round(binY_hat[i].item(), 3)}, {round (binY[i].item(), 3)}', file=f)
    
    #print(f'accuracy = {accuracy}, mean_L1loss = {L1_loss.mean()}')

    #Test
    #convert Xtest to tensor
    X = torch.Tensor(Xtest)
    X = X.double()
    # Predict Test Dataset    
    model.eval()
    with torch.no_grad():               
        y_test_transformed = model(X)    

    #Check L1 score for test set
    #Read tranformed test labels
    test_labels = pd.read_csv('../mp01files/Transformed_Test_Labels.csv', delimiter=',')  
    test_labels = test_labels.to_numpy()   
    
    #Bin predicted and labels and rank them    
    binY_hat = rankbin (y_test_transformed.numpy())
    #Use Y instead of Yval because shuffle is True for loaderV. Y is of the same size as Yval
    binY = rankbin (test_labels) 

    #Save the binned predictions and labels for val set in a csv
    with open ('../Results_new/TestPred.csv', 'w+') as f:
        print (f'predicted_bin, actual_bin', file=f)
        for i in range (y_test_transformed.shape[0]):
            print (f'{round(binY_hat[i].item(), 3)}, {round (binY[i].item(), 3)}', file=f)

    # calculate accuracy
    correct = 0
    for i in range (len(binY)) :
        correct += (binY_hat[i] == binY[i])

    test_accuracy = (correct / len(binY)) * 100 
    
    test_L1_loss = 0
    # calculate the ordinal loss matrix
    test_L1_loss = np.abs (binY_hat - binY)    

    print (f'test_accuracy = {test_accuracy}, test_mean_ordinal_L1_loss = {test_L1_loss.mean()}')     
    
    #Predictions are quantil transformed.
    #Convert numpy to dataframe
    #test_df = pd.DataFrame (y_test_transformed.numpy())     
    #save test predictions
    #test_df.to_csv("../Results_new/TestPred.csv",index=False)        

    return L1_loss, accuracy   

"""This function is required for Optuna hyper param tuning. It calls the predict function 
for n_trials times with different hyper-param values"""

def objective(trial):

     #Define Hyperparams to optimize
    hidden = trial.suggest_int('hidden', 3, 20)
    epoch = trial.suggest_int("epochs", 100,5000)
    patience = trial.suggest_int("patience", 50,500)
    lr = trial.suggest_float('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 5,100)
    
    Xtrain, Ytrain, Xval, Yval, Xtest = PrepareInputs ()
    L1_loss, accuracy = Predict (Xtrain,Ytrain, Xval, Yval, Xtest, hidden, lr, epoch, patience, batch_size=batch_size)
      
    print(f'accuracy = {accuracy}, mean_L1loss = {L1_loss.mean()}')
    
    return L1_loss.mean()

""" Main function is split into two parts.
part 1 - Optuna is enabled and we search for best hyper params
part 2 - Hyper params are set and we call PrepareInputs() and Predict () function for a number of iterations
         Optuna takes the mean ordinal L1 loss as a parameter and finds the best hyper param values to minimize
         mean ordinal L1 loss
"""

def main():   
    optuna_en = 0

    if optuna_en == 1:        
        n_trials = 200    

        # Run the hyperparameter search using Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)   

        # Print the best hyperparameters and loss
        print('Best hyperparameters: {}'.format(study.best_params))
        print('Best mean_L1Loss: {:.6f}'.format(study.best_value))

    else :
        iter = 1
        accuracy = []
        loss  = []
        # Best Hyper params found from Optuna
        hidden = 15
        lr = 0.019236343193189905
        epoch = 934       
        patience = 141
        batch_size = 97

        for i in range(iter):
            Xtrain, Ytrain, Xval, Yval, Xtest = PrepareInputs ()
            l1loss, acc = (Predict (Xtrain, Ytrain, Xval, Yval, Xtest, hidden, lr, epoch, patience, batch_size))
            accuracy.append(acc)
            loss.append(l1loss)
        mean_accuracy = np.asarray(accuracy).mean()
        mean_ordinal_loss = np.asarray(loss).mean()

        print(f'val_accuracy = {mean_accuracy}, val_mean_ordinal_l1_loss = {mean_ordinal_loss}, iterations = {iter}')        

if __name__ == '__main__':
   main()

     
    