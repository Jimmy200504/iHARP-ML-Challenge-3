
# iHARP ML Challenge 2 - 'Predicting Coastal Flooding Events.'
## Challengle Overview
### Background and Motivation
The United States East Coast is particularly vulnerable to hazards such as storm surge-driven extreme flooding in coastal and low-lying areas of the region. The network of offshore and coastal buoys deployed and maintained by the National Data Buoy Center (NDBC) provides invaluable high-resolution data of tidal variability, meteorological data as well as coastal sea level rise. This data is vital for characterizing extreme sea level variation, which is driven by the complex interplay between remotely-forced large-scale atmospheric and ocean dynamics, as well as local meteorological and coastal ocean conditions, making forecasting and predicting anomalous sea level variability and coastal flooding extremely challenging using traditional statistical methods.

The NDBC buoy database provides high-resolution temporal data of tidal and local sea level variation at a plethora of coastal locations spanning over 70 years. The participants in the challenge will leverage novel AI/ML methods to predict coastal flooding events across the different spatial locations of the buoys. Developing a comprehensive extreme flood prediction framework from long-term time series information is vital to enhance the environmental resiliency of coastal communities to the effects of storm surge and coastal flooding in a changing climate.

## Dataset
Sea level observations from the National Data Buoy Center (NDBC) buoys along the US East Coast are essential for monitoring activities and have significant societal impacts. These buoys, equipped with advanced instruments, provide continuous real-time data on tidal variations, storm surges, and long-term sea level trends [Church and White, 2011]. This data is transmitted via satellite to shore-based stations, offering detailed records.

**Please note that: The dataset has seasonality and noise as well as auto-correlated and not stationary. Participants should consider these factors while developing the model.**
### Dataset Structure
Data Source: 'NEUSTG_19502020_12stations.mat'

For the proposed competition, the organizers will provide access to satellite observations from 1950 to 2020 for 12 coastal stations collected through the NDBC. Each station’s data is a time series taken at hourly intervals. Note: The coastal stations selected for this challenge have ATLEAST 80% hourly records from 1950 to 2020, hence number of records may vary slightly across coastal stations.

The format of the data files is .mat and participants can access the challenge dataset from the iHARP Github repository.

See Data Fields, below, or snapshot of the dataset.

Station_Name - Name of the Coas(tal Station
Latitude - Latitude of the Coastal Station
Longitude - Longitude of the Coastal Station
Start_Date - YYYY-MM-DD
End-Date - YYYY-MM-DD
Num_hourly_timestamps - Number of hours from Start_Date to End_Date

### Expected Input
Seed Coastal Stations List of 12 Coastal Stations for Training and Testing. These include:

TRAINING_STATIONS = ['Annapolis','Atlantic_City','Charleston','Washington','Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery']
(To access the dataset, see file 'Seed_Coastal_Stations.txt' in the iHARP Github repository)

Seed Historical Time Intervals List of 15 Seed Historical Time Intervals, where each time window is a Historical 7-Day Interval. For example:

Seed Time Window 1 (7-Day Historical Interval): 1/1/1991 to 1/7/1991
Seed Time Window 2 (7-Day Historical Interval): 1/2/1991 to 1/8/1991
(see file 'Seed_Historical_Time_Intervals.txt' in the iHARP Github repository)

Prediction Time Window For each given historical 7-day interval, participants will predict flooding events for the succeeding 14-day interval.

For example:

Seed Historical Time Interval 1: - Given a 7-Day Historical Interval of 1/1/1991 to 1/7/1991, the Prediction Time Window is 1/8/1991 to 1/21/1991

Seed Historical Time Interval 2: - Given a 7-Day Historical Interval of 1/2/1991 to 1/8/1991, the Prediction Time Window is 1/9/1991 to 1/22/1991

Flooding Thresholds List of 12 numeric continuous values as flooding thresholds for each coastal station, respectively. (see file 'Seed_Coastal_Stations_Thresholds.mat' in the iHARP Github repository)
## Evaluation
### Traing and Testing
#### Traing Set
* Participants will be given hourly data from 1950 to 2020 for 9 coastal stations. This will be used for model training whereby participants can train their model on any 7-day historical interval and make predictions for the 14-day prediction interval following each historical time window given.
* TRAINING_STATIONS = ['Annapolis','Atlantic_City','Charleston','Washington','Wilmington', 'Eastport', 'Portland', 'Sewells_Point', 'Sandy_Hook']
* Additional validation can be varied across select coastal stations (outof 9) to improve outcomes for modelling out-of-distribution across space.
#### Testing Set
* Participants will be given hourly data from 1950 to 2020 for 3 coastal stations, which will be used strictly for testing and evaluating out-of-distribution.
* TESTING_STATIONS = ['Lewes', 'Fernandina_Beach', 'The_Battery'] 
* To test their models, participants have been given specific 15 historical time intervals (7-days per interval) as seed time intervals. These seed intervals have been selected between 1950 to 2020 and are not specific to any particular coastal station from the total of 12 coastal stations provided.
* Participants also have the alternative to test their models using ANY user-defined 7-day historical time intervals to improve outcomes for modelling out-of-distribution across time.
### Evaluation Metrics
This competition allows you to submit your developed algorithm, which will be run on the development and the final test dataset through CodaBench.

**Expected Outputs**: For EACH 14-day prediction time interval (following each 7-day historical time interval/seed window), your algorithm needs to generate the following outputs:

**Evaluation Metrics**: Models will be evaluated based on:
1. F1-Score
2. Accuracy
3. Matthews Correlation Coefficient (MCC)

### Model Testing using Hidden Dataset
**Hidden Test Dataset:** Each participant's FINAL model will be evaluated on the hidden test dataset during the final phase. This test dataset comprises of FOUR (4) hidden coastal stations with hourly sea level measurements for the time period of 1950 to 2020. Each hidden station's file format is similar to those in the first training and test dataset that is accessible to the participants.

**Model Expectations**

The participant is expected to develop a global model that should predict across all stations. Therefore, the prediction should take into account 2520 entries (12 stations X 15 historical windows X 14 prediction windows), where 9 are for training and 3 for out-of-distribution testing.
Please note that the input to the model will be a 7-day window, but the participant is expected to predict for the next succeeding 14-day window period.
The dataset will be in hourly format, but the participant should consider developing the model on a daily interval rather than an hourly interval.
This will be treated as a binary prediction model. Please note that predictions will be done on a binary scale [1 -> Flooding, 0 -> Non-flooding]. We assume that a flooding day has at least ONE (1) hour of a flooding event. So given 24 hours in day D1, if any hour hx where x {1,… ,24} has a flooding event, then the day D1 is marked as flooding.
Evaluation should be based on per entry in the global model, hence 2520 entries derived from (12 stations X 15 historical windows X 14 prediction windows) where 9 are for training and 3 for out-of-distribution testing.
The output should consist of a Confusion Matrix indicating predicted flooding days.
The output should include the following evaluation metrics: Accuracy, F1-Score & Matthews Correlation Coefficient (MCC). Please note that predictions will be done on a binary scale [1 -> Flooding, 0 -> Non-flooding].

**During the final phase models will be further tested using a hidden dataset containing 840 entries (4 stations X 15 historical windows X 14 prediction windows). This will be conducted 'secretly' by the challenge organizers.**
### Evaluation Phases
There are TWO (2) phases for this challenge:

(A) DEVELOPMENT PHASE:

The provided dataset contains:

9 coastal stations for training and 3 coastal stations for testing, both with hourly sea level measurements from 1950 to 2020
15 pre-selected historical time intervals (7-day interval)
Upload your model:

Feedback will be provided on the development output until the end of the challenge. Therefore, up to five submissions are allowed per day (to allow participants to improve their model accordingly). See 'starting kit and sample submission' for expected submission files.

**Note: **Participants may submit "one score" during the development phase, which will be displayed on the leaderboard. This score can be removed and replaced with a newer or better score as preferred.

(B) FINAL PHASE:

This phase will start automatically at the end of the challenge. Models will also be tested using the hidden dataset during the final phase, which will be conducted 'secretly' by the challenge organizers.

Be sure to submit your preferred algorithm as a final submission before the end of the challenge, as this will be the model run on the hidden test dataset for final scores.

Each participant's last submission will be evaluated on the final test set and scores will be posted to the leaderboard.

Questions about this challenge can be placed by creating a NEW ISSUE on the iHARP-ML-Challenge-2 repository on GitHub.


# This is a repository for the Year 2 HDR ML Challenge themed 'Modelling Out Of Distribution.'

***Important note for the baseline model:***
1. The example submission uses 'alternative' thresholds as placeholders for the model.
Please refer to the .mat file for the flooding thresholds ('Seed Coastal Stations Thresholds.mat') in this repository.

2. Given 12 coastal stations, 9 stations are fixed for training and 3 for testing, which aligns with how the model will be processed (during ingestion) and scored.

Therefore, the results for the evaluation metrics of the baseline model may vary between the local machine and Codabench (which has predetermined training and test sets for out-of-distribution modelling). This type of evaluation will be applied during the final phase on the hidden dataset.

3. ***Refer to the zipped submission files to understand the expectations:***
- Example 1: model_submission.zip contains only the model.py, which is the baseline model in this case.
- Example 2: model_submission-2.zip contains ALL files as outlined under the "Expected Submission Files" on Codabench. These include model.py, model.pkl, requirements.txt, and README.md.
  
  ***Both types of submissions will be executed appropriately in Codabench***
