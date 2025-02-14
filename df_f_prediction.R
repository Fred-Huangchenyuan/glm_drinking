#' This script is for predicting the outcome based on optimized model
#' @author Huang Chenyuan, Xu Lingyu
#' @date 2024-02-14

library(glm2)
library(readxl)

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

# Fit the model on the data
model <- glm(df_f_per ~ Lickrate + Past5s + water,
             data = combined,
             family = gaussian(link = "identity"))

# Read the test data and get unique bout numbers
test_data <- combined
bout_numbers <- unique(test_data$Bout)

# Rename the fourth column - "df/f%"
names(test_data)[4] <- "df_f_per"

# Add water column to entire dataset at once
test_data$water <- 0
test_data[test_data$Osmolality == "water", "water"] <- 1

# Rename the fourth column - "df/f%"
names(test_data)[4] <- "df_f_per"

# Loop through each bout
for (bout in bout_numbers) {
  # Filter data for current bout
  bout_data <- test_data[test_data$Bout == bout,]
  
  # Get predictions
  predicted_values <- predict(model, newdata = bout_data)
  
  # Save to CSV 
  write.csv(predicted_values, file = paste0("bout", bout, ".csv"))
}