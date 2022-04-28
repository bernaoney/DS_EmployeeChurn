
load("data/employee_churn_data.RData")
library(tidyverse)
head(df)

df <- df %>% mutate(ID = 1:nrow(df),
                    left = fct_relevel(left, "yes")) %>%
  select(ID, department:left)

cores <- parallel::detectCores(logical = FALSE)
doParallel::registerDoParallel(cores = cores - 1)

library(tidymodels)
set.seed(1)
tt_split <- initial_split(df, 
                          prop = .8, 
                          strata = left)

train_set <- training(tt_split)
test_set <- testing(tt_split)

model_recipe <- recipe(left ~ ., data = train_set) %>%
  update_role(ID, new_role = "ID") %>%
  step_naomit(everything(), skip = TRUE) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_zv(all_numeric(), -all_outcomes()) %>%
  step_corr(all_predictors(), threshold = 0.7, method = "spearman")

summary(model_recipe)

prepped_data <- model_recipe %>% prep() %>% juice()


# cross-validation
set.seed(3)
cv_folds <- vfold_cv(train_set, v = 10, strata = left)


# logistic regression
logr_spec <- logistic_reg() %>% 
  set_engine(engine = "glm") %>%
  set_mode("classification")

logr_wflow <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(logr_spec)

logr_res <- logr_wflow %>% 
  fit_resamples(resamples = cv_folds,
                metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                         roc_auc, sens, spec),
    control = control_resamples(save_pred = TRUE))

logr_res %>% collect_metrics(summarize = TRUE)

logr_metrics <- logr_res %>% collect_metrics(summarise = TRUE) %>%
  mutate(model = "Logistic Regression")


# k nearest neighbors
knn_spec <- nearest_neighbor(neighbors = 7) %>%
  set_engine("kknn") %>% 
  set_mode("classification")

knn_wflow <- workflow() %>%
  add_recipe(model_recipe) %>% 
  add_model(knn_spec)

knn_res <- knn_wflow %>% 
  fit_resamples(resamples = cv_folds, 
                metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                                     roc_auc, sens, spec),
    control = control_resamples(save_pred = TRUE))

knn_res %>% collect_metrics(summarize = TRUE)

knn_metrics <- knn_res %>% collect_metrics(summarise = TRUE) %>%
  mutate(model = "KNN")


# random forest
rf_spec <- rand_forest() %>% 
  set_engine("ranger",
             num.trees = 1000,
             importance = "impurity",
             seed = 123) %>% 
  set_mode("classification")

rf_wflow <- workflow() %>%
  add_recipe(model_recipe) %>% 
  add_model(rf_spec)

rf_res <- rf_wflow %>% 
  fit_resamples(resamples = cv_folds,
                metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                                     roc_auc, sens, spec),
                control = control_resamples(save_pred = TRUE))

rf_res %>%  collect_metrics(summarize = TRUE)

rf_metrics <- rf_res %>% collect_metrics(summarise = TRUE) %>%
  mutate(model = "Random Forest")


# XGboost
xgb_spec <- boost_tree(trees = 1000, learn_rate = 0.01) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_wflow <- workflow() %>%
  add_recipe(model_recipe) %>% 
  add_model(xgb_spec)

xgb_res <- xgb_wflow %>% 
  fit_resamples(resamples = cv_folds,
                metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                                     roc_auc, sens, spec),
                control = control_resamples(save_pred = TRUE))

xgb_res %>% collect_metrics(summarize = TRUE)

xgb_metrics <- xgb_res %>% collect_metrics(summarise = T) %>%
  mutate(model = "XGBoost")


# neural network
library(keras)
nnet_spec <- mlp() %>%
  set_mode("classification") %>% 
  set_engine("keras", verbose = 1) 

nnet_wflow <- workflow() %>%
  add_recipe(model_recipe) %>% 
  add_model(nnet_spec)

nnet_res <- nnet_wflow %>% 
  fit_resamples(resamples = cv_folds,
                metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                                     roc_auc, sens, spec),
                control = control_resamples(save_pred = TRUE))

nnet_res %>% collect_metrics(summarise = TRUE) 

nnet_metrics <- nnet_res %>%
  collect_metrics(summarise = TRUE) %>%
  mutate(model = "Neural Net")

# XGboost wins in the train set
last_fit(xgb_wflow, 
         split = tt_split,
         metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                              roc_auc, sens, spec)) %>% collect_metrics()
# also try random forest in the test set
last_fit(rf_wflow, 
         split = tt_split,
         metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                              roc_auc, sens, spec)) %>% collect_metrics()


ML_op_auc <- bind_rows(logr_metrics, knn_metrics, rf_metrics, xgb_metrics, nnet_metrics) %>%
  select(model, .metric, mean, std_err) %>% 
  pivot_wider(names_from = .metric, values_from = c(mean, std_err)) %>%
  arrange(mean_roc_auc) %>% 
  mutate(model = fct_reorder(model, mean_roc_auc)) %>%
  ggplot(aes(model, mean_roc_auc, fill = model)) +
  geom_col() +
  coord_flip() +
  xlab("Mean ROC AUC -- Binary Classification") +
  ylab("Model") + 
  labs(subtitle = "Algorithms", 
       caption = "Best Model is XGBoost with .904 in Train Set & .910 in Test Set
       \nRandomForest .898 in Train Set & .900 in Test Set
       \nNeuralNet .853 in Train Set
       \nKNN .799 in Train Set
       \nLogistic Regression .715 in Train Set") +
  theme_bw() + theme(legend.position = "none")

xgb_pred_roc_auc <- xgb_res %>% collect_predictions() %>% 
  group_by(id) %>%
  roc_curve(left, `.pred_yes`) %>% 
  autoplot() + labs(subtitle = "ROC AUC in Train Set with XGBoost")

cm_xgb <- last_fit(xgb_wflow, 
         split = tt_split,
         metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                              roc_auc, sens, spec)) %>%
  collect_predictions() %>% 
  conf_mat(left, .pred_class) %>% 
  autoplot(type = "heatmap") +
  labs(subtitle = "Test set confusion matrix with XGBoost")

cm_rf <- last_fit(rf_wflow, 
         split = tt_split,
         metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                              roc_auc, sens, spec)) %>%
  collect_predictions() %>% 
  conf_mat(left, .pred_class) %>% 
  autoplot(type = "heatmap") +
  labs(subtitle = "Test set confusion matrix with Random Forest")

gridExtra::grid.arrange(ML_op_auc, xgb_pred_roc_auc,
                        cm_xgb, cm_rf,
                        ncol = 2,
                        top = "Model Evaluation Metrics")



library(vip)
last_fit(xgb_wflow, 
         split = tt_split,
         metrics = metric_set(recall, precision, f_meas, accuracy, kap,
                              roc_auc, sens, spec)) %>% 
  pluck(".workflow", 1) %>%   
  extract_fit_parsnip() %>% 
  vip(num_features = 9) + theme_bw() +
  labs(title = "Variable importance", 
       subtitle = "with XGBoost algorithm") 

# hyper parameter grid search
xgb_spec <- boost_tree(trees = 1000,
                       tree_depth = tune(), min_n = tune(),
                       loss_reduction = tune(),
                       sample_size = tune(), mtry = tune(),
                       learn_rate = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_grid <- grid_latin_hypercube(tree_depth(), min_n(),
                                 loss_reduction(),
                                 sample_size = sample_prop(),
                                 finalize(mtry(), train_set),
                                 learn_rate(), size = 30)

set.seed(1234)
xgb_res_hpt <- tune_grid(xgb_wflow, resamples = cv_folds, grid = xgb_grid,
                         control = control_grid(save_pred = TRUE))

xgb_res_hpt
