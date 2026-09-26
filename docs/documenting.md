## Using nflreadpy
This python package receives data from nflverse repos, which provides easy access to NFL data with caching, progress tracking, and modern Python conventions. Includes fast data loading with Polars DataFrames.

## Separating Player and D/ST Prediction Pipelines
The MVP will focus on QB, RB, WR, TE, and K.
D/ST will require a separate team-level prediction pipeline because fantasy defenses represent an entire team's defensive performance rather than an individual defensive player.
I'll postpone D/ST until after the MVP works so that I can first learn and validate the core ML workflow without adding this complexity to it.

## What is data leakage?
Data leakage is when information unavailable at prediction time accidentally being included in the model's inputs. For example: Trying to predict a player's week 5 fantasy points, but using only their stats from week 5.

# Early-season Predictions (Weeks 1-3)
With the current rolling features, they limit the historical information early in the season. Week 1 gives no current-season history, while weeks 2-3 are just based on those games. Need to determine how the model should handle insufficient current-season data without relying too heavily on potentially outdated previous season performance.

# RB - Opportunity
Opportunity is defined as carries + targets, which gives us a sense of RB usage. Targets are used over receptions because it shows more opportunities.

# Sleeper API
Sleeper provides league specific scoring_settings and roster_positions. My initial ML model will use a consistent baseline target, while future league integration can adapt projections/recommendations to each league's scoring rules

# Evaluation Metrics - How to measure how wrong the predictions are
MAE - Mean Absolute Error
RMSE - Root Mean Squared Error
MAE is good for fantasy football,
EX: MAE = 4.8
Meaning on average, the model's prediction is off by about 4.8 fantasy points.
RMSE is similar but punishes bad misses more heavily.

# V1 Baselines
Linear Regression
Validation MAE: 4.36
Validation RMSE: 5.99
Summary: Our Linear regression baseline misses WR/TE fantasy points by around 4.36 on average, and the higher RMSE suggests that it has some larger prediction inaccuracies
Note: Training performance tells us how well the model learned the training data, Validation performance tells us how useful that learning is on new data

Random Forest
Validation MAE: 4.569101092794647
Validation RMSE: 6.1381267566983535

XGBoost
Validation MAE: 4.596481784364022
Validation RMSE: 6.327636564601338

# Tuning XG Boost hyperparameters
n_estimators: How many boosting trees to build
learning_rate: How much each new tree is allowed to correct the previous model. Smaller balues learn more cautiously
max_depth: How complex each individual tree can get. Larger depth can capture more patterns but can also overfit.
subample: What fraction of training rows each tree sees. Using less than 1.0 can reduce overfitting
colsample_bytree: What fraction of features each tree sees. This can also reduce overfitting
More depth/more trees -> More model felxibility -> Can learn more useful patterns BUT can also overfit more easily

Tuning can make a complex model catch up to a simple baseline, but it doesn't automatically mean the complex model is the better choice.

AFter tuning XGBoost, it significantly improved the validation performance and brought it to roughly even with linear regression

# Season-based CV Grid Search - Hyperparameter tuning for XGBoost
We gave GridSearchCV a set of possible values and it trained a lot of XGBoost models using different combinations of those settings and then evaluated them across chronological folds.

The grid search selected
max_depth = 2
learning_rate = 0.03
n_estimators = 300
subsample = 0.8
colsample_bytree = 1.0
and produced the best 2024 WR?TE validation performance by far

Hyperparameters
→ chosen before/during tuning
→ control how the model learns

Model parameters
→ learned during fitting
→ represent what the model learned from the data

We test different hyperparameter values, choose the bes combination, then the model uses those settings to learn its model parameters from the training data

# Addition of matchup features
Prediction features are selected based on validation performance, while matchup metrics can remain available to the recommendation/explanation layer as grounded contextual evidence

# Final test metrics
WR
- Final Model: Linear Regression
- 2025 MAE: 4.383
- 2025 RMSE: 5.887

TE
- Final Model: Linear Regression
- 2025 MAE: 3.612
- 2025 RMSE: 5.072

RB
- Final Model: Linear Regression + matchup features
- 2025 MAE: 4.458
- 2025 RMSE: 6.285

QB
- Final Model: Random Forest
- 2025 MAE: 6.898
- 2025 RMSE: 8.456

K
- Final Model: Linear Regression
- 2025 MAE: 3.006
- 2025 RMSE: 3.772