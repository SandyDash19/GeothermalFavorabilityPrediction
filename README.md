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
