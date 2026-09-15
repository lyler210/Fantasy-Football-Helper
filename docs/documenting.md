## Using nflreadpy
This python package receives data from nflverse repos, which provides easy access to NFL data with caching, progress tracking, and modern Python conventions. Includes fast data loading with Polars DataFrames.

## Why use PPR points
The app will initially target PPR leagues, so fantasy_points_ppr will be the prediction target.

Future versions should calculate points from each league's actual Sleeper scoring settings

## Using previous-game rolling statistics
Current-week statistics would not be known before the game and would cause data leakage. Using stats from previous weeks to represent the recent form.

## Separating Player and D/ST Prediction Pipelines
The MVP will focus on QB, RB, WR, TE, and K.
D/ST will require a separate team-level prediction pipeline because fantasy defenses represent an entire team's defensive performance rather than an individual defensive player.
I'll postpone D/ST until after the MVP works so that I can first learn and validate the core ML workflow without adding this complexity to it.

## What is data leakage?
Data leakage is when information unavailable at prediction time accidentally being included in the model's inputs. For example: Trying to predict a player's week 5 fantasy points, but using only their stats from week 5.

## Row layout for training
Each row will represent a player entering a specific game/week. Predictor features must contain only information that would have been available BEFORE the game, such as historical player performance, recent usage, season-to-date statistics, opponent information, and matchup context. The player's actual fantasy points from that game will serve as the training target.
Example: A row for a WR going into a Week 8 game would contain his average receiving yards from weeks 5-7, average target share from Weeks 5-7, and his week 8 opponent
ONE EXCEPTION: Including Week 8 fantasy points. For a training row, theres Features (X) = information known before Week 8, and Target(Y) = what actually happened in Week 8 which is 'fantasy_points_ppr'.
This is crucial because it is what the model actually learns from (its like an answer key).

## First ML approach - Supervised learning
I will first explore using supervised learning with a regression model as the initial ML approach. Past games will provide labeled training examples as features (X), and the player's actual fantasy points are the target (Y). The model will predict the expected fantasy points rather than directly classifying what players to Start vs Sit. Those recommendations will be determined by comparing player predictions within the user's roster.
Current hierarchy: ML -> Supervised Learning -> Regression -> Predict fantasy points

## Feature Engineering - Rolling Averages
1. Why not use the target week's stats?
A: This will lead to data leakage because those stats are not known before the game, causing the model to fail when it's actually used
2. Why choose last 3 game averages?
A: This will show the current performance of the player
3. Why chose last 5 game averages?
A: This will reveal how the player has generally been performing over a longer-term baseline. 
4. What will comparing these two numbers do?
A: Comparing the 3 game average to this will also allow the model to detect if there are any changes in recent performance.

## Season boundaries for rolling averages
For the initial model, the rolling averages will be reset at the start of each NFL season. The early-season predictions will only be based on whatever information is already available rather than trying to pull from previous season games. This avoids carrying over a potentially outdated context from an old coordinator, roster, etc. This also helps prevent numbers weeks 17/18 from affecting the model, where teams that are locked in for the playoffs are more likely to sit their starters to prevent injury.
To sort by player id were using group by which prevents players from mixing their data, and shift(1) to exclude the current game

# Historical Data Usage
I will start off by using player statistics from the 2019-2025 seasons to provide the model with a larger training dataset. Older data may become less representative of the current NFL, so the historical range can later be treated as a modeling decision and evaluated through validation performance.

# Selecting Trend Features
Initially using these trend features for WRs - targets, target share, receiving yards, and receiving air yards. A Trend is defined as 3-game rolling average - 5-game rolling average, where positive values will indicate recent performance/usage above the broader 5 game baseline and negative values indicate a decline

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

# Splitting up WR and TE
At this point I have finished finding all of the models that I want to use for each position before doing the final evaluation. But I want to separate the WR and TE dataset because their offensive roles differ depending on the team. An elite TE could see the same amount of opportunities as a WR2 or even a WR1, while other TEs have less opportunity.

# Models for positions
WR - Linear Regression
TE - Linear Regression
RB - Linear Regression
QB - Tuned/Grid-searched Random Forest
K - Linear Regression