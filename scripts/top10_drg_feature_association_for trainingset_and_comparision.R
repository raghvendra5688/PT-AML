library(dplyr)
library(readr)
library(tidyr)
library(gridExtra)
library(ggpubr)
library(ggplot2)
predictions <- read.csv("Catboost_MFP_Feat_Var_supervised_train_predictions.csv", sep = "\t")
drug_smile <- read.csv("PT-AML-main/PT-AML-main/Data/Drug_Full_SMILES_Embedding.csv")

predictions_with_cid <- predictions %>%
  left_join(drug_smile[, c("CID", "Name")], by = c("inhibitor" = "Name"))


# View the result
head(predictions_with_cid)

# Path to the zip file
zip_path <- "Training_Set_Var_with_Drug_MFP_Cell_Info.zip"

# List the files inside the zip (to check the CSV file name)
zip_contents <- unzip(zip_path, list = TRUE)
print(zip_contents)

# Assuming the file inside is named "Test_Set_Var_with_Drug_MFP_Cell_Info.csv"
csv_name <- zip_contents$Name[1]  # Adjust this if needed

# Read the CSV file directly from the zip
data <- read.csv(unz(zip_path, csv_name))

# View the first few rows
head(data)

combined_df <- merge(predictions_with_cid, data, 
                     by = c("dbgap_rnaseq_sample", "CID"), 
                     all = FALSE)  # inner join

# View the result
head(combined_df)
# Step 1: Keep only inhibitor.x and drop inhibitor.y
combined_df$inhibitor <- combined_df$inhibitor.x
combined_df$inhibitor.x <- NULL
combined_df$inhibitor.y <- NULL

# Step 2: Remove all columns starting with "MFP"
combined_df <- combined_df[ , !grepl("^MFP", names(combined_df)) ]

# View cleaned result
head(combined_df)

# View the result
head(combined_df)
# Step 1: Keep only inhibitor.x and drop inhibitor.y
combined_df$inhibitor <- combined_df$inhibitor.x
combined_df$inhibitor.x <- NULL
combined_df$inhibitor.y <- NULL

# Step 2: Remove all columns starting with "MFP"
combined_df <- combined_df[ , !grepl("^MFP", names(combined_df)) ]

# View cleaned result
head(combined_df)
inhibitors_to_keep <- c(
  "Venetoclax",
  "Selumetinib (AZD6244)",
  "Trametinib (GSK1120212)",
  "Rapamycin",
  "Motesanib (AMG-706)",
  "17-AAG (Tanespimycin)",
  "Dasatinib",
  "Cabozantinib",
  "Elesclomol",
  "Sorafenib"
  
)

# Filter the dataframe
filtered_df <- combined_df[combined_df$inhibitor %in% inhibitors_to_keep, ]

# View the result
head(filtered_df)

####all top 10 drugs
# Define target inhibitors
target_inhibitors <- c(
  "Venetoclax",
  "Selumetinib (AZD6244)",
  "Trametinib (GSK1120212)",
  "Rapamycin",
  "Motesanib (AMG-706)",
  "17-AAG (Tanespimycin)",
  "Dasatinib",
  "Cabozantinib",
  "Elesclomol",
  "Sorafenib"
)

# List to store results
all_significant_results <- list()

for (drug in target_inhibitors) {
  
  # Filter for the current drug
  drug_df <- filtered_df[filtered_df$inhibitor == drug, ]
  
  # Initialize result data frame
  assoc_results <- data.frame(
    feature = character(),
    p_value = numeric(),
    q_value = numeric(),
    method = character(),
    beta = numeric(),
    r_squared = numeric(),
    stringsAsFactors = FALSE
  )
  
  # List to store p-values
  p_values <- c()
  
  for (col in colnames(drug_df)[34:ncol(drug_df)]) {
    feature_data <- drug_df[[col]]
    auc <- drug_df$predictions / 300
    
    if (all(is.na(feature_data)) || length(unique(na.omit(feature_data))) <= 1) {
      next
    }
    
    unique_vals <- unique(na.omit(feature_data))
    
    if (all(unique_vals %in% c(0, 1))) {
      group1 <- auc[feature_data == 0]
      group2 <- auc[feature_data == 1]
      
      if (length(group1) > 2 && length(group2) > 2) {
        test <- t.test(group1, group2)
        p_values <- c(p_values, test$p.value)
        assoc_results <- rbind(assoc_results, data.frame(
          feature = col,
          p_value = test$p.value,
          q_value = NA,  # No q-value for t-test
          method = "t-test (binary feature)",
          beta = mean(group2, na.rm = TRUE) - mean(group1, na.rm = TRUE),
          r_squared = NA  # Not applicable for t-test
        ))
      }
      
    } else {
      df_temp <- data.frame(auc = auc, x = feature_data)
      df_temp <- df_temp[complete.cases(df_temp), ]
      
      if (nrow(df_temp) > 3) {
        fit <- tryCatch(lm(auc ~ x, data = df_temp), error = function(e) NULL)
        if (!is.null(fit)) {
          summary_fit <- summary(fit)
          if ("x" %in% rownames(summary_fit$coefficients)) {
            p_val <- summary_fit$coefficients["x", "Pr(>|t|)"]
            beta <- summary_fit$coefficients["x", "Estimate"]
            r_squared <- summary_fit$r.squared
            
            p_values <- c(p_values, p_val)
            
            assoc_results <- rbind(assoc_results, data.frame(
              feature = col,
              p_value = p_val,
              q_value = NA,  # No q-value for linear regression
              method = "linear_regression (continuous/count feature)",
              beta = beta,
              r_squared = r_squared
            ))
          }
        }
      }
    }
  }
  
  # Calculate q-values using Benjamini-Hochberg method
  assoc_results$q_value <- p.adjust(p_values, method = "fdr")
  
  # Filter significant results (q-value < 0.05 and R-squared >= 0.1)
  significant_results <- assoc_results[assoc_results$q_value < 0.05 & assoc_results$r_squared > 0.1, ]
  
  # Save results for the current drug
  write.csv(significant_results, paste0(drug, "_trainingset_LR_significant_features.csv"), row.names = FALSE)
  
  # Store top 10 features by q-value
  top_10_features <- significant_results[order(significant_results$q_value), ][1:10, "feature"]
  all_significant_results[[drug]] <- top_10_features
  
  # Print top 10 features for the current drug
  cat("\nTop 10 Features for", drug, ":\n")
  print(top_10_features)
}

# Initialize a vector to store all top 10 features
all_top_10_features <- unlist(all_significant_results)

# Create a table of feature counts
feature_counts <- table(all_top_10_features)

# Convert to a data frame for easier viewing
feature_count_df <- as.data.frame(feature_counts)
colnames(feature_count_df) <- c("CommonFeature", "Drugs")

# Filter features that appear in at least 1 drug's top 10
feature_count_df <- feature_count_df[order(-feature_count_df$Drugs), ]

# Print the result
print(feature_count_df)

# Optionally, save the result to a CSV file
write.csv(feature_count_df, "feature_counts_Training_across_drugs.csv", row.names = FALSE)

####################all features common check
# Initialize a vector to store all significant features across all drugs
all_significant_features <- c()

# Loop through each drug's significant results
for (drug in target_inhibitors) {
  # Load the significant results for the current drug
  drug_significant_results <- read.csv(paste0(drug, "_trainingset_LR_significant_features.csv"))
  
  # Collect the features for this drug
  all_significant_features <- c(all_significant_features, drug_significant_results$feature)
}

# Create a table of feature counts across all drugs
feature_counts_all_drugs <- table(all_significant_features)

# Convert to a data frame for easier viewing
feature_count_df_all_drugs <- as.data.frame(feature_counts_all_drugs)
colnames(feature_count_df_all_drugs) <- c("CommonFeature", "Drugs")

# Sort the data frame by count in descending order
feature_count_df_all_drugs <- feature_count_df_all_drugs[order(-feature_count_df_all_drugs$Drugs), ]

# Print the result
print(feature_count_df_all_drugs)

# Optionally, save the result to a CSV file
write.csv(feature_count_df_all_drugs, "Training_feature_counts_across_all_drugs.csv", row.names = FALSE)

# Load the list of features to extract betas from the feature_counts_across_all_drugs.csv
feature_counts <- read.csv("Training_feature_counts_across_all_drugs.csv")

# Initialize a matrix to store the beta values for each feature across drugs
beta_matrix <- data.frame(Feature = feature_counts$CommonFeature, stringsAsFactors = FALSE)

# Loop through each drug and extract beta values
for (drug in target_inhibitors) {
  
  # Load the significant results for the current drug
  drug_significant_results <- read.csv(paste0(drug, "_LR_significant_features.csv"))
  
  # Create a vector for the current drug's beta values
  betas_for_drug <- numeric(nrow(feature_counts))
  
  # Find and store beta values for each feature in the feature_counts list
  for (i in 1:nrow(feature_counts)) {
    feature_name <- feature_counts$CommonFeature[i]
    
    # Check if the feature exists in the drug's significant results
    feature_row <- drug_significant_results[drug_significant_results$feature == feature_name, ]
    
    if (nrow(feature_row) > 0) {
      # If the feature is found, store its beta value
      betas_for_drug[i] <- feature_row$beta
    } else {
      # If the feature is not found, store NA
      betas_for_drug[i] <- NA
    }
  }
  
  # Add the beta values for this drug as a new column in the beta matrix
  beta_matrix[[drug]] <- betas_for_drug
}

# Print the beta matrix
print(beta_matrix)

# Optionally, save the beta matrix to a CSV file
write.csv(beta_matrix, "Training_betas_for_selected_features_across_drugs.csv", row.names = FALSE)

################Train and test set comparision
Trainig_beta <- read.csv("Training_betas_for_selected_features_across_drugs.csv")
Test_beta <- read.csv("betas_for_selected_features_across_drugs.csv")

# Step 1: Get the common features
common_features <- intersect(Trainig_beta$Feature, Test_beta$Feature)

# Step 2: Subset the Train and Test data to only include common features
train_common <- Trainig_beta[Trainig_beta$Feature %in% common_features, ]
test_common <- Test_beta[Test_beta$Feature %in% common_features, ]

# Step 3: Sort both data frames by Feature to align the rows
train_common <- train_common[order(train_common$Feature), ]
test_common <- test_common[order(test_common$Feature), ]

# Step 4: Rename columns in test to distinguish from train (except 'Feature')
colnames(test_common)[-1] <- paste0(colnames(test_common)[-1], "_Test")
colnames(train_common)[-1] <- paste0(colnames(train_common)[-1], "_Train")

# Step 5: Merge both on the Feature column
combined_betas <- merge(train_common, test_common, by = "Feature")

# Step 6: View the combined beta matrix
head(combined_betas)

# Optional: Save to CSV
write.csv(combined_betas, "common_features_betas_train_test.csv", row.names = FALSE)

