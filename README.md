# GeothermalEnergyPrediction
This repository takes geothermal data as input and predicts geothermal favorability.

Details about the code files:
1. FilteredInputs.py - Reads Train.csv and splits it into Train and Val sets based on 80:20 train:val ratio. 
Removes outliers in train set and transforms the distribution to normal. Saves it in a csv.
Transforms val distribution to normal but does not remove outliers. Saves it in a csv.
Reads test.csv, transforms the distribution and save it in a csv. 

2. FilteredwDataLoader.py - This file reads the above saves csv, prepares the inputs, configures the neural network model and saves the val and test output in a csv. The val output is transformed from the model so I performed the same transformation on labels and rank bin edges. Then I calculated ordinal L1 loss and accuracy. 
Test outputs are quantile transformed. 

3. EarlyStop.py has code prevent overtraining

4. Network.py has neural network init and forward pass.

5. MP1_solved.pdf - Technical Report on this project. 

Couple of questions that I want to explore in future:
1. What kind of data transformation is good for this dataset? How do we pick different types of data transformation methods?
2. QuantileTransformer converted the distribution to normal but that completely omitted 1s and 4s. I might want to try StdScalar transformation from Sklearn. 
3. For outlier removal, may be we should observe box plots and remove outliers for certain features only where most of the data falls outside of the box. 
4. I should plot feature maps and try removing correlated features to see what does that do to the accuracy?
