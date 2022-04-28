
load("data/employee_churn_data.RData")
library(tidyverse)
df_numeric <- df %>% mutate_if(is.factor, as.numeric) %>% as.data.frame()

library(parallel)
cl <- makeCluster(7)

black_list_matrix = as.matrix(data.frame(from = rep("left", 9),
                                         to = colnames(df_numeric[,1:9])))

library(bnlearn)
set.seed(1)
bn_op <- tabu(df_numeric, blacklist = black_list_matrix)
library(qgraph)
qgraph(bn_op, vsize = 9, label.cex = 2)

set.seed(2)
boot_res <- boot.strength(data = df_numeric,
                          R = 10000,
                          algorithm = "tabu",
                          algorithm.args = list(blacklist = black_list_matrix),
                          cluster = cl)
avgnet_threshold <- averaged.network(boot_res, threshold = .99)
qgraph(avgnet_threshold, vsize = 9, label.cex = 2,
       title = "Bayesian Belief Network Explaining the Mechanism for Employee Churn")
title(sub = "Years of experience determines the average monthly hours of work,
             \nwhich in turn impact employees' review score.
             \nReview score has a causal association to employee churn.
             \nEmployees' satisfaction score ")
