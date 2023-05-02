import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

"""This file takes the raw input of training data provided for this proj and 
performs the following EDA on it. 

Box Plots           - We want to look at all the 29 features in the data and see the 25th
                      and the 75th quartile of features, mean and outliers.

Plot distributions - We want to look at the distribution of the data to observe skews
                     in the dataset.
                     
Outlier removal    - Outliers can throw off the mean and standard deviation of features
                     There are several outlier removal technique. For this project I have 
                     Z Score outlier removal, however I have experimented with both Z score and
                     Inter Quartile Range method (IQR). Both Zscore and IQR are performed with 
                     feature wise mean and std.
                     
Data Transformation - After the outlier removal, the distribution of each feature was still skewed. 
                      Therefore I have chosen quantile transformation with normal disribution
                      
This file finally saves train and test data as csv files which will be used by FilteredwDataLoader.py

Data size reduction due to outlier removal 
Original 3112 x 29 (Last column is label)
Train 80%  - 2490 x 28
Val   20%  - 622 x 28
IQR Train size - 856 x 29
Z Score   size - 1813 x 29 (This is why I have chosen Z score method)
"""

"""This function performs inter quartile range (IQR) method for outlier removal
Since each feature has a different spread and distribution therefore I have chosen to perform IQR
separately for each column. I have borrowed this code from the following source.

Algo 
1. Calculate 75th quartile (Q3) and 25th quartile (Q1) of the feature. 
2. Calculate IQR by subtracting Q1 from Q3
3. Calculate the boundaries by slightly expanding iqr.
4. Remove all the data points that are above the upperbound and below the lower bound

source : https://www.geeksforgeeks.org/how-to-use-pandas-filter-with-iqr/
"""

def iqr (data, col):
    q3 = np.quantile (data[col], 0.75)
    q1 = np.quantile (data[col], 0.25)
    iqr = q3 - q1

    lowerbound = q1 - 1.5 * iqr
    upperbound = q3 + 1.5 * iqr

    outlier_free_list = [x for x in data[col] if (x > lowerbound)
                        & (x < upperbound)]
    filtered_data = data.loc[data[col].isin(outlier_free_list)]
    
    return filtered_data

"""This function performs zscore method for outlier removal
Since each feature has a different spread and distribution therefore I have chosen to perform zscore
separately for each column.

Algo 
1. Calculate mean and std for the particular feature that is passed in. 
2. Calculate upper and lower bounds. Note that here the bounds are larger than IQR which lets some of 
   the noisy, outliers be present in training data and produces lower mean ordinal loss compared to 
   IQR method or zscore without column wise mean and std.    
3. Remove all the data points that are above the upperbound and below the lower bound
"""

def zscore (data, col):    

    data_mean = data[col].mean()
    
    #Calculate std for all columns
    data_std = data[col].std()
    #print (data_std[0])    
    
    upperbound = data_mean + 3*data_std
    lowerbound = data_mean - 3*data_std

    outlier_free_list = [x for x in data[col] if (x > lowerbound)
                        & (x < upperbound)]
    filtered_data = data.loc[data[col].isin(outlier_free_list)]
    
    return filtered_data

"""
This function produces box plots for all the features. The reason I have put it in a function is because I wanted
to see the box plot of all the features before and after outlier removal and data transformation.
I have plotted both all columns in one figure and individual columns. All column box plot can be taken as a starting
point to focus on features with more outliers and skewed spread.
"""

def boxplots (allorindividual, df, test):

    if allorindividual == 1:
        # All Columns together
        df.plot(
        kind='box', 
        subplots=True, 
        sharey=False, 
        figsize=(10, 6)
        )

        # increase spacing between subplots
        plt.subplots_adjust(wspace=0.5) 
        plt.show()       
        
    else :
         # Column wise subplots
        fig, ax = plt.subplots(1, 10, figsize=(10, 6))
        fig, ax1 = plt.subplots(1, 10, figsize=(10, 6))
        fig, ax2 = plt.subplots(1, 10, figsize=(10, 6))
        # draw boxplots - for one column in each subplot
        df.boxplot('hf22_tgwt', ax=ax[0])
        df.boxplot('cond_sur', ax=ax[1])
        df.boxplot('cond_lcr', ax=ax[2])
        df.boxplot('cond_mcr', ax=ax[3])        
        df.boxplot('cond_uma', ax=ax[4])    
        df.boxplot('cond_man', ax=ax[5])
        df.boxplot('geop_mag', ax=ax[6])
        df.boxplot('geop_grv', ax=ax[7])
        df.boxplot('geop_dtb', ax=ax[8])
        df.boxplot('eqi200n5', ax=ax[9])        
        #------------------------------
        df.boxplot('eqd200n5', ax=ax1[0])
        df.boxplot('geod_shr', ax=ax1[1])
        df.boxplot('geod_dil', ax=ax1[2])  
        df.boxplot('geod_2nd', ax=ax1[3])
        df.boxplot('vent_ing', ax=ax1[4])
        df.boxplot('dTrend005',ax=ax1[5])
        df.boxplot('dTrend03', ax=ax1[6])
        df.boxplot('dTrend01', ax=ax1[7])
        df.boxplot('hfqc_tRise',ax=ax[8])
        df.boxplot('faultAl', ax=ax1[9])
        #-----------------------------
        df.boxplot('fault01', ax=ax2[0])  
        df.boxplot('fault02', ax=ax2[1])
        df.boxplot('fault03', ax=ax2[2])
        df.boxplot('fault04', ax=ax2[3])
        df.boxplot('fault05', ax=ax2[4])
        df.boxplot('fault06', ax=ax2[5])
        df.boxplot('fault07', ax=ax2[6])
        df.boxplot('fault08', ax=ax2[7])
        if test == 0:
            df.boxplot('hfqc_resid', ax=ax2[8])

        plt.subplots_adjust(wspace=0.5) 
        plt.show()

"""
This function plots histogram of all the columns in the dataframe
"""

def plot_dist (df):
    df.hist(figsize=(10,10))  
    plt.subplots_adjust(hspace=0.5)
    plt.show()

"""
This function performs log transform of features. I dropped the log transformation because
the distribution was not normal and the distribution still appeared skewed.
"""
   
def log_transform (df):

    df = df.applymap(np.log)
    df = df.replace([np.inf, -np.inf], np.nan)
    plot_dist (df)

    return df

"""
This function performs quantile transformation of features. This transformation makes the feature
distribution normal.
"""

def quantileTransform (df) :
    # create QuantileTransformer object
    qt = QuantileTransformer(output_distribution='normal')

    # apply quantile transform on dataframe
    df_quantile = pd.DataFrame(qt.fit_transform(df), columns=df.columns)    
    
    return df_quantile

"""
This function is the heart of this file. 
1. First split the training file into training and validation data sets. 
2. Perform quantile transformation on validation data set and save it in an excel sheet. 
3. Perform outlier removal and data transformation on training set and save it in an excel 
   sheet.
"""

def removeOutlier_transform(iqr_or_zscore, edabefore, edaafter): 

    allorindividual = 0
    skew = 0
    test = 0
        
    data = pd.read_csv('../mp01files/heatflow_resid_train.csv', delimiter=',')  

    #Look at the mean, std and quartiles of the features
    data.describe()

    #split train and val dataset right away and dont do anything to validation set    
    train, val = train_test_split(data, test_size=0.20)    

    #Transform the val data to get normal distribution but do not remove outliers
    valT = quantileTransform (val)  

    #plot val distribution
    if edaafter == 1:
        plot_dist (valT)  
    
    #save val dataset into a file
    valT.to_csv('../mp01files/val.csv', index=False)

    #print(train.shape, test.shape)
    
    df = train 

    #Now that we are done with training and validation data sets, lets plot the distribution
    #of test dataset.
    data_test = pd.read_csv('../mp01files/heatflow_resid_test.csv', delimiter=',') 
    data_test_labels = pd.read_csv('../mp01files/heatflow_resid_test_labels.csv', delimiter=',') 

    if edabefore == 1:
        test = 1
        plot_dist(data_test)
        boxplots (allorindividual=allorindividual,df=data_test, test=test)

    #transform test dataset, no outlier removal just like val set
    #Save the quantile object to inverse_transform in Predict() in FilteredwDataLoader.py
    TestT = quantileTransform (data_test) 
    Test_labels = quantileTransform (data_test_labels) 
    TestT.to_csv('../mp01files/Transformed_Test.csv', index=False)
    Test_labels.to_csv('../mp01files/Transformed_Test_Labels.csv', index=False)    
    
    if edaafter == 1:
        plot_dist (TestT)

    #Outlier removal and data transformation
    """with open ('DataDescribe.txt', 'w+') as f:
        print(f'{df.describe()}', file=f)"""
    
    if edabefore == 1:
        plot_dist(df)
        boxplots (allorindividual=allorindividual,df=df, test=test)
    
    # Inter Quartile Range outlier removal
    if iqr_or_zscore == 1: 
        for i in df.columns:
            if i == df.columns[0]:
                dataiqr = iqr (df, i)
            else :                
                dataiqr = iqr (dataiqr, i)
        new_data_filtered = dataiqr       
        
        dataT = quantileTransform (new_data_filtered)
        print(dataT.shape)

        dataT.to_csv('../mp01files/filtered_data/filtered_data_iqr_train.csv', index=False)

    # Z Score outlier removal technique
    elif iqr_or_zscore == 2:       
        
        for i in df.columns:
            if i == df.columns[0]:
                datazscore = zscore (df, i)   
            else :
                datazscore = zscore (datazscore, i) 
        new_data_filtered = datazscore
        
        dataT = quantileTransform (new_data_filtered)
        print(dataT.shape)
             

        dataT.to_csv('../mp01files/filtered_data/filtered_data_zscore_train.csv', index=False)  

    # Apply both of the above, zscore first and then IQR
    else :   
        #zscore outlier removal
        for i in df.columns:
            if i == df.columns[0]:
                datazscore = zscore (df, i)   
            else :
                datazscore = zscore (datazscore, i) 
        zscoreD = datazscore

        #IQR outlier rmeoval after zscore 
        for i in zscoreD.columns:
            if i == zscoreD.columns[0]:
                dataiqr = iqr (zscoreD, i)
            else :                
                dataiqr = iqr (dataiqr, i)
        
        new_data_filtered = dataiqr

        dataT = quantileTransform (new_data_filtered)
        print(dataT.shape)

        dataT.to_csv('../mp01files/filtered_data/filtered_data_both_train.csv', index=False)        

    if edaafter == 1:
        plot_dist(dataT)
        boxplots (allorindividual=0,df=dataT, test=test)   
    

def main():   
    
   iqr_or_zscore = 2
   edabefore = 0
   edaafter = 0
   removeOutlier_transform (iqr_or_zscore, edabefore, edaafter)       

if __name__ == '__main__':
   main()

     
    