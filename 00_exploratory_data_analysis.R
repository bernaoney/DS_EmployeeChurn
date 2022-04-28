
library(tidyverse)
df <- readr::read_csv('data/employee_churn_data.csv')
head(df)

df$department <- factor(df$department)
df$promoted <- factor(df$promoted, levels = c(0,1), labels = c("not_promoted",
                                                                "promoted"))
df$salary <- factor(df$salary, levels = c("low", "medium", "high"))
df$bonus <- factor(df$bonus, levels = c(0,1), labels = c("not_received",
                                                          "received"))
df$left <- factor(df$left, levels = c("no", "yes"))
head(df)

save(df, file = "data/employee_churn_data.RData")

# churn prop by department
df %>% ggplot(aes(x = department, fill = left)) +
  geom_bar(position = "fill") +
  labs(x = "Department", 
       y = "Proportion of Employees") +
  theme_bw()




# churn prop by department
# improved plot with %s
# prep df
d_plot <- df %>%
  group_by(department, left) %>%
  summarize(n = n()) %>% 
  mutate(pct = n/sum(n),
         lbl = scales::percent(pct),
         department = as_factor(department),
         left = as_factor(left)) 
# plot
d_plot %>% ggplot(aes(x = department,
                      y = pct,
                      fill = left)) +
  geom_bar(stat = "identity",
           position = "fill") + 
  scale_y_continuous(breaks = seq(0, 1, .1)) +
  geom_text(aes(label = lbl), 
            size = 3, 
            position = position_stack(vjust = 0.5)) +
  labs(x = "Department", 
       y = "% of Employees",
       title = "Employee Churn",
       subtitle = "by department") +
  theme_bw() + theme(axis.text.x = element_text(angle = 30))




# churn by salary group
df %>% ggplot(aes(x = salary, fill = left)) +
  geom_bar(position = "fill") +
  labs(x = "Salary Group", 
       y = "Proportion of Employees") +
  theme_bw()
chisq.test(table(df$salary, df$left))
# Pearson's Chi-squared test with Yates' continuity correction
# X-squared = 1.1484, df = 2, p-value = 0.5632




# churn by promotion
df %>% ggplot(aes(x = promoted, fill = left)) +
  geom_bar(position = "fill") +
  labs(x = "Salary Group", 
       y = "Proportion of Employees",
       title = "Employee Churn",
       subtitle = "by being promoted or not") +
  theme_bw()
chisq.test(table(df$promoted, df$left))
# Pearson's Chi-squared test with Yates' continuity correction
# X-squared = 12.436, df = 1, p-value = 0.0004212




# churn by bonus
df %>% ggplot(aes(x = bonus, fill = left)) +
  geom_bar(position = "fill") +
  labs(x = "Salary Group", 
       y = "Proportion of Employees") +
  theme_bw()
chisq.test(table(df$bonus, df$left))
# Pearson's Chi-squared test with Yates' continuity correction
# X-squared = 1.1973, df = 1, p-value = 0.2739




# see the relationship between 3 continuous variables
# by department & churn
df %>% ggplot(aes(x = review, y = satisfaction, color = avg_hrs_month)) +
  geom_jitter(alpha = .3) + scale_colour_gradientn(colours = terrain.colors(10)) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(x = "Composite score the employee received in their last evaluation", 
       y = "Employee satisfaction score",
       title = "The relationship between employee evaluation score & employee satisfaction",
       subtitle = "by department & employee churn",
       caption = "There seems to be a negative relationship between employee review scores & employee satisfaction:
        the higher review scores employees get the less satisfied they are,
       \nThe average hours the employee worked in a month also seem to be involved in this relationship
       \nVisual inspections suggest that the intercept and slope differences across those who left the company are noteworthy") +
  theme_bw() +
  facet_grid(left ~ department)




# see the relationship between 3 continuous variables
# by department, promotion & churn
df %>% 
  ggplot(aes(x = review, y = satisfaction)) +
  geom_jitter(aes(color = promoted), alpha = 0.6, show.legend = TRUE) +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  theme_bw() +
  facet_grid(left ~ department, scales = "free")




# adjust correlations for nested data
library(easystats)
df %>% select(review, satisfaction, avg_hrs_month, department) %>% correlation(multilevel = TRUE)
# see correlations with group by
df %>% select(review, satisfaction, avg_hrs_month, department, promoted, left) %>% 
  group_by(promoted, left) %>%
  correlation()
  



# function taken from
# https://www.pipinghotdata.com/posts/2021-10-11-estimating-correlations-adjusted-for-group-membership/
# compute correlation adjusted for nested data 
compute_adj_corr <- function(data, var_dep, var_ind, var_group){
  
  mixed_formula <- glue::glue("{var_dep} ~ {var_ind} + (1 | {var_group})")
  
  mixed_model <- lme4::lmer(mixed_formula, data)
  
  coef_sign <- mixed_model %>% 
    broom.mixed::tidy() %>% 
    filter(term == var_ind) %>% 
    pull(estimate) %>% 
    sign()
  
  r2_by_group <- performance::r2_nakagawa(mixed_model, by_group = TRUE)$R2[1]
  
  adj_corr <- coef_sign * sqrt(r2_by_group)
  
  return(adj_corr)
}

adj_corr_by_dept <- df %>% group_by(department) %>% nest() %>%
  mutate(corr = map_dbl(data,
                        compute_adj_corr,
                        var_dep = "satisfaction",
                        var_ind = "review",
                        var_group = "left")) %>%
  select(department, corr)
adj_corr_by_dept

