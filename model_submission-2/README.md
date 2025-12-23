
#Important note for the baseline model provided in the Google Colab:

#1. The example submission uses 'alternative' thresholds as placeholders for the model. 
Please refer to the .mat file for the flooding thresholds ('Seed Coastal Stations Thresholds.mat)' in the iHARP Github repository.

#2. Given 12 coastal stations, 9 stations are fixed for training and 3 for testing which aligns with how the model will be processed (during ingestion) and scored. 
Therefore the results for the evaluation metrics for the baseline model may vary on the local machine vs. in codabench (which has predetermined training and test sets for modelling out of distribution). This type of evaluation will be applied during the final phase on the hidden dataset.


