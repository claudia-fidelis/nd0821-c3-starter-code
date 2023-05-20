# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Developer: Cláudia Miriam Fidelis
Model date: 18/05/2023
Model type: Gradient Boosting from sklearn.ensemble

## Intended Use

Predict whether a person makes more then 50K a year using the Census Income (AKA Adult) Data Set.

## Training Data

The data was obtained from the Barry Becker from the 1994 Census database (https://archive.ics.uci.edu/ml/datasets/census+income).

The original data set has 32561 rows and 14 features, and a 80-20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.

We used GridSearchCV to perform a hyperparameter optimization and we gwt the following result: 
`{'learning_rate': 0.2, 'max_depth': 5, 'n_estimators': 50}`

|Feature|Importance|
|---|---|
|education|0,190156|
|fnlgt|0,186653|
|education-num|0,062478|
|age|0,056633|
|marital-status|0,039080|
|workclass|0,013561|
|hours-per-week|0,005584|
|relationship|0,002151|
|race|0,001746|
|capital-gain|0,001073|
|capital-loss|0,001011|
|native-country|0,000513|
|occupation|0,000000|
|sex|0,000000|


## Evaluation Data

## Metrics

Precision = 0.7896341463414634,
Recall = 0.6544535691724573,
FBeta = 0.7157167530224526

## Ethical Considerations

From the metrics above we see that bias is present in some of the features and is not consistent across metrics. Specially we see unfairness in the model regarding to the race Amer-Indian-Eskimo.

## Caveats and Recommendations

Model was trained with 1994 data and can be unsuited for use on recent data Model have higher performance on more educated individuals and shouldn't be used in a real life scenario due to potential bias.