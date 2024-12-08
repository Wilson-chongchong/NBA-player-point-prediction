# NBA Player Point Prediction

## Overview
This project aims to predict the points scored by NBA players based on various game-related statistics. It uses machine learning models to analyze historical data, identify patterns, and predict future performance. The primary goal is to create an accurate model that can predict player points based on features like field goals, rebounds, assists, and more.

Step 1: Introduction
In the world of professional sports, predicting player performance in upcoming games has become an essential part of strategic decision-making. Accurate predictions can guide coaches, analysts, and fantasy sports enthusiasts in assessing a player's potential and making informed decisions about team composition, game strategy, and overall performance. In basketball, for instance, understanding how previous performance metrics such as points scored, field goal percentage, assists, and rebounds correlate with future game outcomes is critical for identifying trends and making predictions.

The dataset we are working with contains performance data for basketball players from various past games. This data includes a variety of game statistics, such as points scored (PTS), field goals made (FGM), assists, rebounds, turnovers, and other relevant metrics for individual players. It also captures the opponent (OPP) faced by the player, the date of each game, and other features that may influence performance, such as game location (home vs. away), player injuries, or rest days.

Our primary focus is to predict the number of points (PTS) a player will score in their next game based on historical performance data. Predicting player performance, particularly points scored, is a complex task influenced by multiple factors, such as a player's current form, the strength of the opponent, and the dynamics of the team. By building a predictive model, we aim to explore how a player's past performance, game conditions, and opponent statistics interact to determine their potential points in future games.

The central question we are exploring is: How can we predict a player's points for the next game based on their past performance, opponent, and other relevant factors? To answer this, we will use machine learning techniques, including regression models and feature engineering, to build a model that can make these predictions. By leveraging these techniques, we aim to identify patterns in the data that are indicative of a player's future performance.

Dataset Overview:
I am utilizing [NBA-API](https://github.com/swar/nba_api) to fetch the data from [nba.com](https://www.nba.com)

Number of Rows: The dataset consists of 372 game entries for a specific player, recorded across multiple seasons (2018-2024).

Relevant Columns:

SEASON: The season in which the game was played.
WL: Win/Loss outcome of the game (1 for win, 0 for loss).
MIN: Minutes played by the player in the game.
FGM: Field Goals Made.
FGA: Field Goals Attempted.
FG_PCT: Field Goal Percentage (FGM/FGA).
FG3M: Three-Point Field Goals Made.
FG3A: Three-Point Field Goals Attempted.
FG3_PCT: Three-Point Field Goal Percentage (FG3M/FG3A).
FTM: Free Throws Made.
FTA: Free Throws Attempted.
FT_PCT: Free Throw Percentage (FTM/FTA).
OREB: Offensive Rebounds.
DREB: Defensive Rebounds.
REB: Total Rebounds (OREB + DREB).
AST: Assists.
STL: Steals.
TOV: Turnovers.
PTS: Points scored by the player in the game.
HOME: Whether the game was played at home (1 if yes, 0 if away).
OPP: The opponent the player faced in the game.

Why This Dataset Matters:
This dataset is important because it offers a rich, real-world representation of how basketball players perform across different games and seasons. It includes both the raw performance metrics (e.g., points, assists, field goal percentage) and situational data (e.g., home or away game, opponent) that can significantly influence player performance. By predicting future points scored based on these variables, we can gain valuable insights that help with game strategy, player evaluations, and fantasy sports decisions.

Predicting player points isn't just about understanding a player's past performance—it's about factoring in nuances like game location, opponent strength, and the dynamics of a player's recent form. With this model, we aim to predict how these factors contribute to future performance, offering actionable insights for coaches, analysts, and sports enthusiasts.

Conclusion:
Ultimately, this work seeks to harness the power of machine learning to improve understanding of player performance dynamics. The ability to predict player points can help optimize game strategies, evaluate players more accurately, and even support decisions in fantasy sports. By leveraging historical performance data, we aim to make predictions that offer concrete benefits for both individual performance analysis and broader team-level strategy.
