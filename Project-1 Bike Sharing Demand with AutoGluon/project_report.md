
# Report: Predict Bike Sharing Demand with AutoGluon Solution

#### Uday Damerla

## Initial Training

### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

The Kaggle score increased when compared to the previous models.  
I observed that Kaggle refuses the submissions containing negative values so we need to replace all the negative values with 0.

### What was the top ranked model that performed?

The third model with hyper optimisation and added features was the top performed model with the Kaggle score of 0.49396.

## Exploratory data analysis and feature creation

### What did the exploratory analysis find and how did you add additional features?

Feature datetime was parsed as a datetime feature to obtain hour information from timestamp  
Independent features season and weather were converted to categorical variables.  
It was noticed that the RMSE scores improved significantly during cross-validation and these independent features were highly corelated to the target variable count.  
The features casual and registered are only present in the train dataset and absent in the test data; hence, these features were ignored/dropped during model training.

### How much better did your model preform after adding additional features and why do you think that is?

Before adding additional features the model Kaggle score was 1.79132 but after adding these additional features the Kaggle score was 0.66704.  
It was a significant improvement.The model performance improved after converting certain categorical variables with integer data types into their true categorical datatypes.

## Hyper parameter tuning

### How much better did your model preform after trying different hyper parameters?

Hyperparameter tuning was performed to enhance the model's performance compared to the initial submission. Although hyperparameter tuned models delivered competitive performances in comparison to the model with EDA and added features, the latter performed exceptionally better on the Kaggle (test) dataset. The biggest challenge while using AutoGluon with a prescribed range of hyperparameters.

### If you were given more time with this dataset, where do you think you would spend more time?

If given more time to work with the dataset, I would like to investigate additional potential outcomes when AutoGluon is run for an extended period with a high-quality preset and enhanced hyperparameter tuning.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

| model          | hop1         | hop2                 | hop3                | score   |
|----------------|--------------|----------------------|---------------------|---------|
| initial        | default      | default              | default             | 1.79132 |
| added_features | default      | default              | default             | 0.66704 |
| hop            | max_depth:16 | max_features:0.77379 | max_samples:0.91334 | 0.49396 |


### Create a line plot showing the top model score for the three (or more) training runs during the project.

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/e3d65f0229644126049fcde71fb72962241c3e535f36a587.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/0b2c8d8ab62b0da46c0615376a9b4a12776c5ff34abab1ee.png)  
## Summary

I was pleased with how AutoGluon performed with the data. It was easy to find efficient models and could focus more on the quality of the data and features, as well as on the optimization of the hyperparameters. The next steps are to explore the data more efficiently with graphs and statistical analysis, focus on a specific model that performs better, and optimize the hyperparameters of that model to win the competition.
