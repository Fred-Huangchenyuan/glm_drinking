#' This code is to generate the GLM model using different subsets of variables
#' and to find the most optimal ones for the following prediction
#' @author Huang Chenyuan, Xu Lingyu
#' @date 2024-02-14

library(glm2)
library(readxl)

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

######### Created randomly shuffled data when needed
        # shuffled_data <- data.frame(
        #   df_f_per = sample(combined$df_f_per),
        #   Lickrate = sample(combined$Lickrate),
        #   Past5s = sample(combined$Past5s),
        #   water = sample(combined$water),
        #   Boutsize = sample(combined$Boutsize),
        #   Bout = sample(combined$Bout)
        #  )
        # combined <- shuffled_data

# Set the number of iterations
num_iterations <- 100

# Initialize a vector to store the MSE values
mse_values <- numeric(num_iterations)
adjusted_pseudo_rsq <- numeric(num_iterations)
aic_value <- numeric(num_iterations)
overall_p <- numeric(num_iterations)
Lickrate_p <- numeric(num_iterations)
Past5s_p <- numeric(num_iterations)
water_p <- numeric(num_iterations)
Boutsize_p <- numeric(num_iterations)

# Perform the iterations
for (i in 1:num_iterations) {
  # Create a vector of unique group labels
  group_labels <- unique(combined$Bout)
  
  # Calculate the number of groups for training (rounded to the nearest integer)
  num_train_groups <- round(0.8*length(group_labels))
  num_test_groups <- length(group_labels) - num_train_groups
  
  # Randomly select the training groups
  train_groups <- sample(group_labels, num_train_groups, replace=T)
  
  # Subset the data frame based on the selected group labels
  train_data <- combined[combined$Bout %in% train_groups, ]
  test_data <- combined[!(combined$Bout %in% train_groups), ]
  
  # Fit the model on the training data
  model <- glm(df_f_per ~ Lickrate + Past5s + water + Boutsize,
               data = train_data,
               family = gaussian(link = "identity"))
  
  # Get the predicted values for the testing data
  predicted_values <- predict(model, newdata = test_data)
  
  # Calculate the squared differences between predicted and actual values
  squared_differences <- (predicted_values-test_data$df_f_per) ^ 2
  
  # Calculate the mean of the squared differences to get the MSE
  mse_values[i] <- mean(squared_differences)
  adjusted_pseudo_rsq[i] <- 1 - model$deviance / model$null.deviance
  aic_value[i] <- AIC(model)
  overall_p[i] <- summary(model)$coefficients["(Intercept)","Pr(>|t|)"]
  Lickrate_p[i] <- summary(model)$coefficients["Lickrate","Pr(>|t|)"]
  Past5s_p[i] <- summary(model)$coefficients["Past5s","Pr(>|t|)"]
  water_p[i] <- summary(model)$coefficients["water","Pr(>|t|)"]
  Boutsize_p[i] <- summary(model)$coefficients["Boutsize","Pr(>|t|)"]
}

mse_value_mean <- mean(mse_values)
mse_value_sd <- sd(mse_values)
mse_value_sem <- mse_value_sd / sqrt(num_iterations)
adjusted_pseudo_rsq_mean <- mean(adjusted_pseudo_rsq)
adjusted_pseudo_rsq_sd <- sd(adjusted_pseudo_rsq)
adjusted_pseudo_rsq_sem <- adjusted_pseudo_rsq_sd / sqrt(num_iterations)
aic_value_mean <- mean(aic_value)
aic_value_sd <- sd(aic_value) / sqrt(num_iterations)
aic_value_sem <- aic_value_sd 
overall_p_mean <- mean(overall_p)
Lickrate_p_mean <- mean(Lickrate_p)
Past5s_p_mean <- mean(Past5s_p)
water_p_mean <- mean(water_p)
Boutsize_p_mean <- mean(Boutsize_p)

results <- data.frame(
  Output = c("mse_value_mean", "mse_value_sem", "aic_value_mean",
                "aic_value_sem", "overall_p_mean", "Lickrate_p_mean",
                "Past5s_p_mean", "water_p_mean", "Boutsize_p_mean"),
  Value = c(mse_value_mean, mse_value_sem, aic_value_mean,
            aic_value_sem, overall_p_mean, Lickrate_p_mean,
            Past5s_p_mean, water_p_mean, Boutsize_p_mean)
)

write.csv(results, file = "model_parameters.csv")