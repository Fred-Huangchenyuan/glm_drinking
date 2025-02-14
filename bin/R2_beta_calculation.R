#' This script is for calculating delta R2 and standardized effect size for the
#' optimized model
#' @author Huang Chenyuan, Xu Lingyu
#' @date 2024-02-14

library(glm2)
library(readxl)
library(rsq)
library(lm.beta)

set.seed(234)

# Read all files under the destination and combined into one tibble
xlsx_files <- list.files(pattern = "*.xlsx")
all_xlsx_data <- lapply(xlsx_files, read_xlsx)
combined <- all_xlsx_data[[1]]

for (i in c(2:length(all_xlsx_data))){
  combined <- rbind(combined, all_xlsx_data[[i]])
}

# Create categorical variable of water and non-water
combined$water <- 0
combined[combined$Osmolality == "water", "water"] <- 1

# Rename the fourth column - "df/f%"
names(combined)[4] <- "df_f_per"

# Set the number of iterations
num_iterations <- 100

# Initialize a vector to store the MSE values
rsq_water <- numeric(num_iterations)
rsq_Past5s <- numeric(num_iterations)
rsq_Lickrate <- numeric(num_iterations)
b_water <- numeric(num_iterations)
b_Past5s <- numeric(num_iterations)
b_Lickrate <- numeric(num_iterations)

# Perform the iterations
for (i in 1:num_iterations) {
  # Create a vector of unique group labels
  group_labels <- unique(combined$Bout)
  
  # Calculate the number of groups for training (rounded to the nearest integer)
  num_train_groups <- round(0.8*length(group_labels))
  num_test_groups <- length(group_labels) - num_train_groups
  
  # Randomly subset the training groups
  train_groups <- sample(group_labels, num_train_groups, replace=T)
  train_data <- combined[combined$Bout %in% train_groups, ]
  
  # Fit the model on the training data
  model <- glm(df_f_per ~ Lickrate + Past5s + water,
               data = train_data,
               family = gaussian(link = "identity"))
  
  # Calculate the partial R2 and standardized effect size
  rsq_Lickrate[i] <- rsq.partial(model)$partial.rsq[1]
  rsq_Past5s[i] <- rsq.partial(model)$partial.rsq[2]
  rsq_water[i] <- rsq.partial(model)$partial.rsq[3]
  b_Lickrate[i] <- summary(lm.beta(model))$coefficients["Lickrate", "Standardized"]
  b_Past5s[i] <- summary(lm.beta(model))$coefficients["Past5s", "Standardized"]
  b_water[i] <- summary(lm.beta(model))$coefficients["water", "Standardized"]
}

rsq_Lickrate_mean <- mean(rsq_Lickrate)
rsq_Past5s_mean <- mean(rsq_Past5s)
rsq_water_mean <- mean(rsq_water)
b_Lickrate_mean <- mean(b_Lickrate)
b_Past5s_mean <- mean(b_Past5s)
b_water_mean <- mean(b_water)

rsq_Lickrate_sd <- sd(rsq_Lickrate)
rsq_Past5s_sd <- sd(rsq_Past5s)
rsq_water_sd <- sd(rsq_water)
b_Lickrate_sd <- sd(b_Lickrate)
b_Past5s_sd <- sd(b_Past5s)
b_water_sd <- sd(b_water)

rsq_Lickrate_sem <- sd(rsq_Lickrate)/sqrt(num_iterations)
rsq_Past5s_sem <- sd(rsq_Past5s)/sqrt(num_iterations)
rsq_water_sem <- sd(rsq_water)/sqrt(num_iterations)
b_Lickrate_sem <- sd(b_Lickrate)/sqrt(num_iterations)
b_Past5s_sem <- sd(b_Past5s)/sqrt(num_iterations)
b_water_sem <- sd(b_water)/sqrt(num_iterations)

results <- data.frame(
  Covariate = c("Lickrate", "Past5s", "water"),
  Partial_R2_mean = c(rsq_Lickrate_mean, rsq_Past5s_mean, rsq_water_mean),
  Partial_R2_SEM = c(rsq_Lickrate_sem, rsq_Past5s_sem, rsq_water_sem),
  Effect_size_mean = c(b_Lickrate_mean, b_Past5s_mean, b_water_mean),
  Effect_size_SEM = c(b_Lickrate_sem, b_Past5s_sem, b_water_sem)
)

write.csv(results, file = "rsq_beta.csv")