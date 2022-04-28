
load("data/employee_churn_data.RData")
library(tidyverse)
head(df)

df <- df %>% mutate(ID = 1:nrow(df)) %>%
  select(ID, department:left)

cat_var <- colnames(df[,c(2, 3, 6, 9)])
new_df_w_dumm <- fastDummies::dummy_cols(df, select_columns = cat_var,
                                         remove_first_dummy = TRUE,
                                         remove_selected_columns = TRUE)

new_df_w_dumm <- new_df_w_dumm %>% 
  mutate(left = as.numeric(left) -1)

set.seed(123)
train <- new_df_w_dumm %>% sample_frac(.8)
test  <- anti_join(new_df_w_dumm, train, by = 'ID')

x_train <- train %>% select(review:avg_hrs_month,
                            department_engineering:bonus_received)
x_test  <- test %>% select(review:avg_hrs_month,
                           department_engineering:bonus_received)
y_train <- train %>% select(left)
y_test <- test %>% select(left)

library(reticulate)
conda_list()
use_condaenv("r-reticulate")

library(keras)
y_train <- to_categorical(y_train, 2)
y_test  <- to_categorical(y_test,  2)

x_train <- as.matrix(x_train)
y_train <- as.matrix(y_train)
x_test <- as.matrix(x_test)
y_test <- as.matrix(y_test)

build_model <- function() {
  model <- keras_model_sequential()
  model %>% 
    layer_dense(units = 18,
                input_shape = dim(x_train)[2],
                kernel_regularizer = regularizer_l2(l = 0.001)) %>%
    layer_activation_relu() %>%
    layer_dense(units = 100,
                activation = 'relu',
                kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>% 
    layer_dropout(0.2) %>%
    layer_dense(units = 200,
                activation = 'relu',
                kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>% 
    layer_dropout(0.4) %>%
    layer_dense(units = 80,
                activation = 'relu',
                kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>%
    layer_dropout(0.5) %>%
    layer_dense(units = 40,
                activation = 'relu',
                kernel_regularizer = regularizer_l1_l2(l1 = 0.01, l2 = 0.01)) %>% 
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "adam",
    metrics = tensorflow::tf$keras$metrics$AUC()
  )
  model
}

model <- build_model()
model %>% summary()


print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

epochs <- 100

model_history <- model %>% fit(
  x_train,
  y_train,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(early_stop, print_dot_callback)
)

evaluate(model, x_test, y_test)

plot(model_history) + theme_bw() + xlab("") +
  labs(title = "Feed forward deep neural network",
       caption = "AUC | Binary classification in the test set = .70")
