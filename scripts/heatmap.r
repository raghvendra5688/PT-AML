library(dplyr)
library(ggplot2)
data <- read.csv("Catboost_LS_Feat_Var_supervised_test_predictions_2.csv", header = T, sep = ",")
data$mae <- abs(data$X.predictions. - data$X.labels.)
data$mae <- data$mae / 300

# Create a matrix of MAE values (inhibitors vs. samples)
mae_matrix <- matrix(data$mae, nrow = length(unique(data$X.inhibitor.)), byrow = TRUE)

# Set row and column names for the matrix
rownames(mae_matrix) <- unique(data$X.inhibitor.)
colnames(mae_matrix) <- unique(data$X.dbgap_rnaseq_sample.)

# Create a heatmap
heatmap(mae_matrix,  col = colorRampPalette(c("blue", "red"))(50))

heatmap.2(mae_matrix, col = colorRampPalette(c("blue", "red"))(50),
          key = TRUE, key.title = "MAE/300", key.xlab = "Color Intensity",
          xlab = "Patients", ylab = "Drugs")