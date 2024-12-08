# NBA Player Point Prediction

## Overview
This project aims to predict the points scored by NBA players based on various game-related statistics. By leveraging machine learning models, it analyzes historical data, identifies patterns, and predicts future performance. The primary goal is to create an accurate model that considers features such as field goals, rebounds, assists, and more.

---

# Introduction
Predicting player performance in professional sports has become an essential tool for strategic decision-making. Accurate predictions can guide coaches, analysts, and fantasy sports enthusiasts in assessing a player's potential and making informed decisions about team composition, game strategy, and overall performance.

In basketball, metrics like points scored, field goal percentage, assists, and rebounds correlate with future game outcomes. Understanding these relationships helps uncover trends and make informed predictions.  

The dataset used contains performance data from basketball players across multiple past games, including:
- **Key statistics**: Points scored (PTS), field goals made (FGM), assists, rebounds, turnovers, and more.
- **Game context**: Opponent (OPP), game date, and whether it was a home or away game.

**Objective**: Predict the number of points (PTS) a player will score in their next game based on historical performance data.  
Factors such as player form, opponent strength, and team dynamics are considered to answer: *How can we predict a player's points in the next game based on their past performance, opponent, and other relevant factors?*

By employing machine learning techniques like regression models and feature engineering, this project seeks to identify patterns indicative of future player performance.

---

## Dataset Overview
The dataset was sourced using [NBA-API](https://github.com/swar/nba_api), which fetches data from [NBA.com](https://www.nba.com). For this analysis, I am using LeBron James as an example.

### **Key Statistics**
- **Seasons covered**: 2018–2024
- **Number of records**: 372 game entries for a specific player  

---

## Introduction of Columns
Here are the columns included in the dataset:
1. **SEASON_ID**: The season in which the game was played.  
2. **WL**: Win/Loss outcome of the game. 
3. **MIN**: Minutes played by the player in the game.  
4. **FGM**: Field Goals Made.  
5. **FGA**: Field Goals Attempted.  
6. **FG_PCT**: Field Goal Percentage (FGM/FGA).  
7. **FG3M**: Three-Point Field Goals Made.  
8. **FG3A**: Three-Point Field Goals Attempted.  
9. **FG3_PCT**: Three-Point Field Goal Percentage (FG3M/FG3A).  
10. **FTM**: Free Throws Made.  
11. **FTA**: Free Throws Attempted.  
12. **FT_PCT**: Free Throw Percentage (FTM/FTA).  
13. **OREB**: Offensive Rebounds.  
14. **DREB**: Defensive Rebounds.  
15. **REB**: Total Rebounds (OREB + DREB).  
16. **AST**: Assists.  
17. **STL**: Steals.  
18. **TOV**: Turnovers.  
19. **PTS**: Points scored by the player in the game.  

---

## Why This Dataset Matters
This dataset provides a rich, real-world representation of basketball player performance across seasons.  
- **Key insights**: By predicting points scored based on historical metrics, we can help improve game strategies, player evaluations, and fantasy sports decisions.  
- **Practical applications**: Combining raw performance metrics with situational data (e.g., home/away games and opponent strength) offers a deeper understanding of player dynamics.  

Predicting player points is more than analyzing past performance. It requires understanding nuances like game location, opponent strength, and player form. This project combines these factors to predict future performance, offering actionable insights for coaches, analysts, and enthusiasts alike.

---

## Conclusion
This project harnesses the power of machine learning to enhance our understanding of player performance dynamics. Accurate point predictions can optimize game strategies, support fantasy sports decisions, and offer deeper insights into player and team performance.  

By analyzing historical data, we aim to provide practical and impactful solutions for individual and team-level performance analysis.

---

# Data Cleaning and Exploratory Data Analysis
## Data Cleaning

To ensure the dataset was ready for analysis and accurately reflected the data-generating process, I performed the following data cleaning steps. Each step and its rationale are described below:

### 1. **Identifying Home vs. Away Games**
   - **Step**: Added a new column `HOME` to indicate whether the game was played at home (`1`) or away (`0`) by checking if the string `"vs"` appeared in the `MATCHUP` column.
   - **Effect**: Provided a binary indicator for game location, which is a significant factor influencing player performance.

### 2. **Extracting Opponent Teams**
   - **Step**: Created a column `OPP` by extracting the opponent team's abbreviation from the `MATCHUP` column using string operations.
   - **Effect**: Allowed analysis of performance variations against different opponents.

### 3. **Standardizing Season Format**
   - **Step**: Converted the `SEASON_ID` into a readable numeric `SEASON` column by removing a prefix and casting the remaining string to an integer.
   - **Effect**: Ensured consistent season representation for grouping and analysis.

### 4. **Mapping Win/Loss Outcomes**
   - **Step**: Mapped the `WL` column values (`'W'` or `'L'`) to binary values (`1` for win and `0` for loss).
   - **Effect**: Facilitated easier statistical modeling and aggregation based on game outcomes.

### 5. **Handling Missing Data**
   - **Step**: Dropped rows with missing values in critical performance metrics (`PTS`, `FG_PCT`, `FT_PCT`, `AST`, `MIN`).
   - **Effect**: Improved data integrity by ensuring all records contained the essential features for analysis.

### 6. **Feature Engineering with Rolling Averages**
   - **Step**: Computed rolling averages for key performance metrics (`MIN`, `FG_PCT`, `FT_PCT`, `PTS`, `AST`) using a rolling window of 5 games within each season (`SEASON`). The rolling metrics included:
     - `MIN_Roll`: Average minutes played
     - `FG_PCT_Roll`: Field goal percentage
     - `FT_PCT_Roll`: Free throw percentage
     - `PTS_Rolling`: Points scored
     - `AST_Rolling`: Assists
   - **Effect**: Captured recent trends in performance, reflecting the dynamic nature of a player's form over time.

### 7. **Column Selection**
   - **Step**: Selected a subset of relevant columns for further analysis, focusing on game-related and rolling average features:
     - `SEASON`, `WL`, `MIN`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `REB`, `AST`, `STL`, `TOV`, `PTS`, `HOME`, `OPP`, `MIN_Roll`, `FG_PCT_Roll`, `FT_PCT_Roll`, `PTS_Rolling`, `AST_Rolling`
   - **Effect**: Streamlined the dataset to include features most relevant to the analysis.

### Cleaned DataFrame Head
The head of the cleaned DataFrame is shown below, highlighting the transformed and newly added columns:

|   SEASON |   HOME | OPP   |   PTS |   AST |   STL |   TOV |   FG_PCT |   MIN_Roll |   PTS_Rolling |   AST_Rolling |   FG_PCT_Roll |
|---------:|-------:|:------|------:|------:|------:|------:|---------:|-----------:|--------------:|--------------:|--------------:|
|     2018 |      1 | CHA   |    27 |     9 |     0 |     6 |    0.579 |    32      |       27      |        9      |      0.579    |
|     2018 |      1 | WAS   |    23 |    14 |     1 |     3 |    0.55  |    33      |       25      |       11.5    |      0.5645   |
|     2018 |      1 | SAC   |    29 |    11 |     2 |     4 |    0.409 |    33.6667 |       26.3333 |       11.3333 |      0.512667 |
|     2018 |      1 | BKN   |    25 |    14 |     1 |     8 |    0.32  |    34.5    |       26      |       12      |      0.4645   |
|     2018 |      0 | NYK   |    33 |     8 |     0 |     2 |    0.423 |    34.6    |       27.4    |       11.2    |      0.4562   |
|     2018 |      0 | TOR   |    29 |     6 |     1 |     4 |    0.522 |    34.6    |       27.8    |       10.6    |      0.4448   |
|     2018 |      0 | CHI   |    36 |     4 |     2 |     5 |    0.652 |    34.4    |       30.4    |        8.6    |      0.4652   |
|     2018 |      1 | BOS   |    30 |    12 |     0 |     3 |    0.565 |    33      |       30.6    |        8.8    |      0.4964   |
|     2018 |      1 | DEN   |    31 |     7 |     1 |     4 |    0.591 |    31.8    |       31.8    |        7.4    |      0.5506   |
|     2018 |      1 | LAC   |    27 |     6 |     1 |     2 |    0.5   |    33.2    |       30.6    |        7      |      0.566    |

---

## Univariate Analysis

### Distribution of Points Scored (PTS)

<iframe
  src="assets/distribution-pts.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The histogram of points scored (PTS) shows a distribution resembling a normal curve, with most values concentrated around the mean. This suggests that the player's scoring performance is consistent, with fewer games at the extremes of very low or very high points scored. The visualization provides insight into the player's typical scoring range and highlights their reliability as a scorer.

### Distribution of Minutes Played (MIN)

<iframe
  src="assets/distribution-min.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The histogram of minutes played (MIN) displays a distribution that appears to be centered around a moderate range, with most values clustered near the mean. This suggests that the player’s playing time is relatively consistent across games, with fewer instances of extremely low or high minutes played. The visualization offers insight into the player’s typical role in the team, indicating a steady involvement in games and contributing to an understanding of their workload and stamina.

---

## Bivariate Analysis

### Relationship Between Assists and Points Scored

<iframe
  src="assets/pts-ast.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The scatter plot visualizing the relationship between assists (AST) and points scored (PTS) shows a slightly negative trend, where higher assist numbers tend to correspond with slightly lower point totals. This suggests that as the player focuses more on facilitating plays and passing the ball, they score fewer points themselves. The visualization highlights the trade-off between playmaking and scoring, reflecting the player's role as more of a facilitator in the offense.

### Points Scored by Home/Away

<iframe
  src="assets/pts-HomeAway.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

The box plot comparing points scored (PTS) at home versus away games shows similar median values for both situations. However, when the player is at home, there are a couple of outliers where the points scored are notably higher than the rest of the games. This suggests that while the player's scoring performance is consistent whether playing at home or away, there are certain home games where their performance spikes significantly, possibly due to factors like crowd support or matchups.

---

## Interesting Aggregates

### Win Percentage by Home/Away

The table below shows the win percentage for the player in home and away games. The win percentage is calculated as the ratio of wins to the total number of games played in each setting (home/away).

| HOME/AWAY | Win Percentage (%) |
|-----------|---------------------|
| Away      | 54.44               |
| Home      | 61.46               |

This suggests that the player performs better in home games, with a higher win percentage compared to away games.

---

## Imputation

- Original data shape: (372, 30)
- Filtered data shape (after removing rows with NAs): (372, 26)

Since the number of rows remains the same before and after filtering out the missing values, imputation is not needed in this case.

---

# Framing a Prediction Problem
### Prediction Problem and Type

The problem at hand is a **regression** problem where we are predicting the number of **points scored (PTS)** by LeBron James in each game based on various features such as minutes played, field goal percentage, assists, etc. Since we are predicting a continuous numerical value (points scored), this is a **regression** task.

### Response Variable

The response variable, or the target we are predicting, is the **points scored (PTS)** in each game. We chose **PTS** as the target because it represents a key performance metric in basketball, and understanding how different features (like minutes played, shooting percentage, assists, etc.) influence scoring can provide valuable insights into a player's performance.

### Evaluation Metric

To evaluate our model, we are using **Mean Squared Error (MSE)**. MSE is chosen because it penalizes larger errors more heavily, which is useful when dealing with continuous variables like points scored. It is a standard evaluation metric for regression models, allowing us to assess how well the model’s predictions align with the actual values. In our case, lower MSE values indicate better model performance, as the predicted points are closer to the true points scored.


### Information Known at the Time of Prediction

At the time of prediction, we would know the following features for a given game (based on the input data):
- **Minutes Played (MIN)**: The total number of minutes LeBron James played in the game.
- **Shooting Percentages**: Includes field goal percentage (FG_PCT), free throw percentage (FT_PCT), and 3-point percentage (FG3_PCT).
- **Assists (AST)**: The number of assists LeBron James made in the game.
- **Home/Away Status (HOME)**: Whether the game was played at home or away.
- **Opponent (OPP)**: The team that LeBron James played against.

We would not have access to the **PTS** for the game we are predicting, but we do have access to all the other features listed above.

### Model Training

In this regression task, we train the model using the following features:
- **Numerical features**: Minutes played, shooting percentages, assists, etc.
- **Categorical features**: Opponent team (OPP), which is one-hot encoded.
  
By using these features, we aim to predict the points scored by LeBron James in a given game while ensuring that the features selected are known at the time of prediction and do not include future game information.

---

# Baseline Model
### Model Description

For this prediction problem, I am using a **Ridge Regression** model. Ridge regression is a regularized linear regression model that helps prevent overfitting by penalizing large coefficients. This is useful because our dataset may have some multicollinearity (when independent variables are highly correlated), and regularization helps improve the generalization of the model.

The model is built in a **Pipeline** that includes a **ColumnTransformer** to preprocess the data. Specifically, the preprocessing consists of:
1. **One-Hot Encoding** for the **OPP** (opponent) column to convert the categorical feature into numerical values.
2. **Standard Scaling** for the continuous numerical features to normalize them so that they have zero mean and unit variance. This step is important for models like Ridge Regression that are sensitive to the scale of the data.

The final model then fits a **Ridge Regression** model to the processed features.

### Features in the Model

The model includes the following features:

1. **Quantitative Features (Numerical)**:
    - **MIN**: Minutes played in the game.
    - **FGM**: Field goals made.
    - **FGA**: Field goals attempted.
    - **FG_PCT**: Field goal percentage.
    - **FG3M**: 3-point field goals made.
    - **FG3A**: 3-point field goals attempted.
    - **FG3_PCT**: 3-point field goal percentage.
    - **FTM**: Free throws made.
    - **FTA**: Free throws attempted.
    - **FT_PCT**: Free throw percentage.
    - **OREB**: Offensive rebounds.
    - **DREB**: Defensive rebounds.
    - **REB**: Total rebounds.
    - **AST**: Assists.
    - **STL**: Steals.
    - **TOV**: Turnovers.
    - **MIN_Roll**: Rolling average of minutes played over the last 5 games.
    - **FG_PCT_Roll**: Rolling average of field goal percentage over the last 5 games.
    - **FT_PCT_Roll**: Rolling average of free throw percentage over the last 5 games.
    - **PTS_Rolling**: Rolling average of points scored over the last 5 games.
    - **AST_Rolling**: Rolling average of assists over the last 5 games.

   **Total Quantitative Features: 21**

2. **Nominal Features (Categorical)**:
    - **OPP**: Opponent team, which is one-hot encoded during preprocessing.

   **Total Nominal Features: 1**

3. **Ordinal Features**:
    - **HOME**: Whether the game was played at home (1 for home, 0 for away). This feature is treated as nominal in the model since it represents categorical data with no inherent order.

   **Total Ordinal Features: 1**

### Model Performance

To evaluate the performance of the model, we used the **Mean Squared Error (MSE)**, which is a standard metric for regression tasks. The MSE was computed on both the training and testing datasets:

- **Training Mean Squared Error**: 0.0143
- **Test Mean Squared Error**: 0.0381
- **Cross-Validation MSE**: 0.0200

These results suggest that the model is performing reasonably well, with the training MSE being quite low, indicating that the model has learned well on the training data. However, the test MSE is higher, which could indicate that the model is not generalizing perfectly to unseen data. The cross-validation MSE provides a good balance, showing that the model performs moderately well across different splits of the data.

### Model Evaluation

While the model is performing decently, I believe there is room for improvement. Here are a few reasons why:

1. **Model Regularization**: While Ridge Regression is helpful in reducing overfitting, we could explore more sophisticated models such as **Random Forests**, **Gradient Boosting Machines (GBMs)**, or even **Neural Networks** to capture non-linear relationships that Ridge Regression might miss.
   
2. **Feature Engineering**: The rolling averages provide useful information, but we may benefit from creating additional features or interactions between existing features, such as player efficiency ratings or adjusted shooting percentages based on opponent defense.

3. **Hyperparameter Tuning**: The regularization strength (alpha) in Ridge Regression was not optimized. Hyperparameter tuning via techniques such as **Grid Search** or **Randomized Search** could help find a better model configuration.

4. **Evaluation Metrics**: We only used MSE as the evaluation metric. While it's a good standard, using other metrics like **R-squared**, **Root Mean Squared Error (RMSE)**, or even **mean absolute error (MAE)** could give us more insights into model performance.

In conclusion, the current model is a solid baseline, but there is still potential for improvement. Further experimentation with more advanced models and additional feature engineering could enhance the prediction accuracy.

---

# Final Model

In the final model, I utilized a combination of transformed and scaled features. The key transformations and additions include:

1. **Quantile Transformation** on the following numerical features: **MIN** (minutes played), **REB** (rebounds), **AST** (assists), **STL** (steals), **TOV** (turnovers), **FGA** (field goals attempted), and **FTA** (free throws attempted). The purpose of the **QuantileTransformer** was to reduce the influence of outliers and skewed distributions. These features are vital for predicting points (**PTS**), as they represent critical aspects of a player's performance. Applying quantile transformation to these variables ensures that extreme values do not disproportionately affect the model, allowing it to focus on general trends rather than individual outliers.

2. **Standard Scaling** was applied to all numerical features, ensuring that they were centered around zero with unit variance. This standardization is particularly beneficial for models like Random Forest, as it makes the features comparable and avoids giving undue weight to certain features based solely on their scale.

3. **One-Hot Encoding** was applied to the **OPP** (opponent) column. By converting this categorical variable into binary features, the model can properly handle the categorical nature of the opponent variable, enabling it to learn the relationships between the opponent type and the target variable (**PTS**).

These features were chosen because they capture key aspects of the data-generating process for basketball games. Minutes played, rebounds, assists, and other statistics are well-known indicators of player performance, and transforming these features ensures that the model can handle a variety of distributions in the data effectively.

### Modeling Algorithm and Hyperparameters

The **RandomForestRegressor** was chosen for this task because it is a powerful ensemble learning method capable of capturing complex, non-linear relationships in the data. Random forests are particularly useful when dealing with a mix of numerical and categorical features, as they do not require the data to be linearly separable and can naturally handle missing data and outliers.

#### Hyperparameter Tuning:
To improve the performance of the Random Forest model, **GridSearchCV** was used for hyperparameter optimization. The grid search tested several combinations of hyperparameters to identify the best model:

- **n_estimators**: Number of trees in the forest. Values tested were 100, 200, and 300.
- **max_depth**: Maximum depth of the trees. Tested values were `None`, 10, and 20.
- **min_samples_split**: Minimum number of samples required to split an internal node. Tested values were 2, 5, and 10.
- **min_samples_leaf**: Minimum number of samples required to be at a leaf node. Tested values were 1, 2, and 4.

The best combination of hyperparameters identified by GridSearchCV was:
- **n_estimators**: 300
- **max_depth**: 10
- **min_samples_split**: 2
- **min_samples_leaf**: 1

These hyperparameters were selected because they resulted in the lowest **Mean Squared Error (MSE)** during cross-validation, indicating that this combination of parameters best balanced model complexity and generalization.

### Model Performance

The performance of the final model was evaluated using **Mean Squared Error (MSE)**, which measures the average squared difference between the actual and predicted values. 

- **Training Mean Squared Error (Final Model)**: 0.4366
- **Test Mean Squared Error (Final Model)**: 2.2598
- **Cross-Validation MSE (Final Model)**: 2.9146

#### Baseline Model Performance:
The **Baseline Model** performance was as follows:
- **Training Mean Squared Error**: 0.0143
- **Test Mean Squared Error**: 0.0381
- **Cross-Validation MSE**: 0.0200

#### Performance Comparison:
- **Training MSE**: The **Final Model**'s training MSE is 0.4366, which is higher than the baseline model's 0.0143. While this indicates overfitting in the final model, it may suggest that the model has learned more complex patterns from the data.
- **Test MSE**: The **Final Model**'s test MSE of 2.2598 is higher than the baseline model's test MSE of 0.0381, indicating a performance drop on unseen data. This could be due to overfitting, which the baseline model avoids due to its simpler structure.
- **Cross-Validation MSE**: The **Final Model**'s cross-validation MSE of 2.9146 is substantially higher than the baseline model's 0.0200. This confirms that the final model might not generalize well across different subsets of data and overfits to the training data.

### Conclusion

The final model, incorporating **Quantile Transformation**, **Standard Scaling**, and **One-Hot Encoding**, along with **RandomForestRegressor** and optimized hyperparameters, does not show a clear improvement over the baseline model. In fact, it exhibits signs of overfitting, as seen from the higher **Test MSE** and **Cross-Validation MSE**. Despite the more complex feature engineering and hyperparameter optimization, the simpler baseline model appears to generalize better, with a lower test and cross-validation error. This suggests that for this particular problem, a simpler model might be more effective in predicting player points.


