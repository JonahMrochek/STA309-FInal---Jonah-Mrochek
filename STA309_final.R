# Final Exam
# STA 309
# Jonah Mrochek


library(readr)
library(dplyr)
library(tidyverse)

# Data taken from publicly accessible Basketball Reference website
nba <- read_csv("NBA_24_25.txt")

glimpse(nba)
head(nba)

nba <- nba %>%
  select(-Awards, -`Player-additional`)

nba$EFF <- with(nba,
                (PTS + TRB + AST + STL + BLK) -
                  (FGA - FG) - 
                  (FTA - FT) -
                  TOV)

nba <- nba %>%
  drop_na()

nba$IsCav <- ifelse(nba$Team == "CLE", "Cavs", "League")

# The response variable I am exploring is which per-game statistics best predict
# player efficiency. All of the data comes from the 2024-2025 NBA season from
# "Basketball Reference." Each variable is a per-game statistic from each
# individual player, such as points, rebounds, assists, steals, and so on.

# The 5 predictors I will use to compare with the EFF (efficiency) rating are
# points, rebounds, assists, position (categorical), and free throw attempts

library(patchwork)

points_plot <- ggplot(nba, aes(x = PTS, y = EFF)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(
    title = "Points vs Player Efficiency",
    subtitle = "As Expected, When Points Incease, Efficiency does as Well"
  ) +
  theme_minimal()

rebounds_plot <- ggplot(nba, aes(x = TRB, y = EFF)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(
    title = "Total Rebounds vs Player Efficiency",
    subtitle = "As the Number of Rebounds Increase, so Does Efficiency"
  ) +
  theme_minimal()

assists_plot <- ggplot(nba, aes(x = AST, y = EFF)) +
  geom_point() +
  geom_smooth(se=FALSE) +
  labs(
    title = "Assists vs Player Efficiency",
    subtitle = "As the Number Assists Increases, Efficiency Also Increases"
  ) +
  theme_minimal()


Position_plot <- ggplot(nba, aes(x = Pos, y = EFF)) +
  geom_boxplot() +
  labs(title = "Efficiency by Player Position",
       subtitle = "Centers are generally the most efficient")

nba$FTA_bin <- cut(
  nba$FTA,
  breaks = 3,
  labels = c("Low FTA", "Medium FTA", "High FTA")
)

fta_eff_plot <- ggplot(nba, aes(x = FTA_bin, y = EFF)) +
  geom_boxplot(fill = "skyblue") +
  labs(
    title = "Efficiency by Free Throw Attempt Levels",
    subtitle = "The more free throws attempted, regardless of if they make it,
    results in higher efficiency",
    x = "Free Throw Attempt Level",
    y = "EFF"
  ) +
  theme_minimal()



nba_dashboard <- points_plot + rebounds_plot + assists_plot + fta_eff_plot + Position_plot +
  plot_annotation(title = "NBA Per Game Statistics Compared to Player Efficiency",
                  subtitle = "Data From 2024-2025 NBA Season")

ggsave(
  "nba_dashboard.png",
  nba_dashboard,
  width = 16, 
  height = 10, 
  dpi = 300
)




# Fitting models

library(caret)
library(rpart)
library(randomForest)

train_control <- trainControl(method = "cv", number = 5)

# Full model with all 5 predictors
model_lm_full <- train(
  EFF ~ PTS + TRB + AST + Pos + FTA,
  data = nba,
  method = "lm",
  trControl = train_control)

summary(model_lm_full)

# Smaller model featuring points, rebounds, and assists as variables
model_lm_small <- train(
  EFF ~ PTS + TRB + AST,
  data = nba,
  method = "lm",
  trControl = train_control
)

summary(model_lm_small)

model_tree <- train(
  EFF ~ PTS + TRB + AST + FTA + Pos,
  data = nba,
  method = "rpart",
  trControl = train_control,
  tuneLength = 10
)
summary(model_tree)

model_rf <- train(
  EFF ~ PTS + TRB + AST + FTA + Pos,
  data = nba,
  method = "rf",
  trControl = train_control,
  tuneLength = 5
)
summary(model_rf)

model_knn <- train(
  EFF ~ PTS + TRB + AST + FTA + Pos,
  data = nba,
  method = "knn",
  preProcess = c("center", "scale"),
  trControl = train_control,
  tuneLength = 10
)
summary(model_knn)


# Rationales for why I chose these models

# Full model:
# I fit a linear regression to all of the predictors that I am analyzing to get 
# a baseline linear performance. This can be easily interpreted, showing me which
# statistical values correlate most to player efficiency.

# Smaller model:
# I chose to fit points, rebounds, and assists to a smaller linear model to
# compare with the full model. This model tests whether this smaller set of 
# predictors is more efficient in predicting efficiency than the full model.

# Tree model:
# A tree model captures interactions between predictors, so I chose to include
# it in comparing my 5 chosen predictors. I also needed to include a tree model
# so it will help by comparing its effectiveness to other models.

# Random forest model:
# The random forest model collects different decision trees which could lead to
# a more accurate predictive model. I also used all predictors for this model to
# provide me with a good variety.

# k-NN model:
# This method makes predictions based on similarities, providing a different 
# approach in explaining my predictors. The k-NN model provides a contrast in 
# the other models, specifically the tree and linear models.



# Model Comparison

library(ggthemes)

results <- resamples(list(
  LM_Full = model_lm_full,
  LM_Subset = model_lm_small,
  Tree = model_tree,
  RF = model_rf,
  KNN = model_knn
))

summary(results)


RMSE_plot <- dotplot(results, metric = "RMSE")
MAE_plot <- dotplot(results, metric = "MAE")
Rsquared_plot <- dotplot(results, metric = "Rsquared")

results_plot <- dotplot(results)
results_plot

# Summary

# Based on this output, it is safe to say that the linear models perform the
# best when compared to rf, knn, and tree models. Both the linear subset model
# and full model had the lowest MAE values as well as RMSE, signifying a
# successful model. They also are on the higher end of R squared values, further
# proving their validity. R squared doesn't apply as much in this scenario as
# MAE and RMSE, but I still included it to provide another metric that helps 
# support the linear model. 

# This all goes to show me that when searching for which factors best influence
# NBA player efficiency in 2024-25, the linear model is the best predictor. Tree,
# Random Forest, and k-NN models may still work, but aren't nearly as accurate
# as the linear model.




