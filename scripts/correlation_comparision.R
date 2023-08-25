library(dplyr)
library(readr)
library(tidyr)
library(gridExtra)
library(ggpubr)
library(ggplot2)
data <- read.csv("Catboost_MFP_Feat_Var_supervised_test_predictions.csv", header = T, sep = "\t")

# correlations

correlations <- split(data, data$inhibitor) %>%
  lapply(function(subset) {
    if (nrow(subset) >= 10) {
      cor(subset$predictions, subset$labels, method = "pearson")
    } else {
      NA  # If there are fewer than 10 counts, assign NA
    }
  })

# Create a data frame to store correlations
correlations_df <- data.frame(drug = names(correlations), correlation = unlist(correlations))

# Remove rows with NA (less than 10 counts)
correlations_df <- correlations_df[!is.na(correlations_df$correlation), ]

# Order correlations in descending order
correlations_df <- correlations_df[order(-correlations_df$correlation), ]
View(correlations_df)

calculate_rsquared <- function(data) {
  if (nrow(data) >= 10) {
    model <- lm(labels ~ predictions, data = data)
    rsquared <- summary(model)$r.squared
  } else {
    rsquared <- NA
  }
  return(rsquared)
}

# Calculate R-squared for drugs with a minimum of 10 counts
rsquared <- tapply(data$inhibitor, data$inhibitor, function(d) {
  subset <- data[data$inhibitor == d, ]
  calculate_rsquared(subset)
})

# Create a dataframe for R-squared values
rsquared_df <- data.frame(drug = names(rsquared), r_squared = unlist(rsquared))

# Remove rows with NA (less than 10 counts)
rsquared_df <- rsquared_df[!is.na(rsquared_df$r_squared), ]

# Order R-squared values in descending order
rsquared_df <- rsquared_df[order(-rsquared_df$r_squared), ]

# Combine R-squared values into a dataframe
rsquared_df <- data.frame(drug = names(rsquared), r_squared = unlist(rsquared))
View(rsquared_df)
data_drug <- merge(correlations_df,rsquared_df)
View(data_drug)
write.table(data_drug, file = "drug_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)

#for pateints Rsquare calculation

# Define a function to calculate correlations with minimum 10 counts
calculate_correlation <- function(data) {
  if (nrow(data) >= 10) {
    correlation <- cor(data$predictions, data$labels)
  } else {
    correlation <- NA
  }
  return(correlation)
}

# Calculate correlations for groups with a minimum of 10 counts
correlations_patients <- split(data, data$dbgap_rnaseq_sample) %>%
  lapply(function(subset) {
    calculate_correlation(subset)
  })

# Create a data frame to store correlations
correlations_df_p <- data.frame(patients = names(correlations_patients), correlation = unlist(correlations_patients))

# Remove rows with NA (less than 10 counts)
correlations_df_p <- correlations_df_p[!is.na(correlations_df_p$correlation), ]

# Order correlations in descending order
correlations_df_p <- correlations_df_p[order(-correlations_df_p$correlation), ]

# Define a function to calculate R-squared with minimum 10 counts
calculate_rsquared_2 <- function(data) {
  if (nrow(data) >= 10) {
    model <- lm(data$labels ~ data$predictions, data = data)
    rsquared <- summary(model)$r.squared
  } else {
    rsquared <- NA
  }
  return(rsquared)
}

# Calculate R-squared for groups with a minimum of 10 counts
rsquared_patients <- split(data, data$dbgap_rnaseq_sample) %>%
  lapply(function(subset) {
    calculate_rsquared_2(subset)
  })

# Create a data frame to store R-squared values
rsquared_df_p <- data.frame(patients = names(rsquared_patients), r_squared = unlist(rsquared_patients))

# Remove rows with NA (less than 10 counts)
rsquared_df_p <- rsquared_df_p[!is.na(rsquared_df_p$r_squared), ]

# Order R-squared values in descending order
rsquared_df_p <- rsquared_df_p[order(-rsquared_df_p$r_squared), ]

data_sam <- merge(correlations_df_p,rsquared_df_2)
View(data_sam)
write.table(data_sam, file = "sample_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)

data_sam <- read.table("samples_correlations.txt", header = T, sep = "\t")
data_sam <- arrange(data_sam, desc(data_sam$r))
data_sam <- na.omit(data_sam)
top_10_sam_1 <- data_sam %>%
  arrange(desc(r)) %>%
  slice(1:10)
write.table(top_10_sam_1, file = "top10_samples_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)

down_10_sam <- data_sam %>%
  arrange(r) %>%
  slice(1:10)
View(down_10_sam)
write.table(down_10_sam, file = "bottom10_sam_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)
top_10_sam <- read.table("top10_samples_correlations.txt", header = T)
top_10_sam$patients <- factor(top_10_sam$patients)
top_10_sam <- top_10_sam %>%
  arrange(desc(r))
top_10_sam
top_10_sam$patient_order <- factor(top_10_sam$patients, levels = top_10_sam$patients)
data_long_1 <- top_10_sam %>%
  pivot_longer(cols = c(r, r_squared),
               names_to = "Correlation Type",
               values_to = "Correlation")
View(data_long_1)
# Create the side-by-side bar plot
sam_plot_1 <-ggplot(data_long_1, aes(x = patient_order, y = Correlation, fill = `Correlation Type`)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = NULL) +
  labs(title = "A",x = "Top 10 Patients",
       y = "Correlation") +
  scale_fill_manual(values = c("r" = "blue", "r_squared" = "red"),
                    labels = c("r" = "r", "r_squared" = expression(R^2))) +
  theme_minimal() + theme(plot.title = element_text(face = "bold"),
                          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
sam_plot_1

down_10_sam <- read.table("bottom10_sam_correlations.txt", header = T)
down_10_sam$patients <- factor(down_10_sam$patients)
down_10_sam <- down_10_sam %>%
  arrange(desc(r))
down_10_sam$patient_order <- factor(down_10_sam$patients, levels = down_10_sam$patients)
data_long_2 <- down_10_sam %>%
  pivot_longer(cols = c(r, r_squared),
               names_to = "Correlation Type",
               values_to = "Correlation")
View(data_long_2)
# Create the side-by-side bar plot
sam_plot_2 <-ggplot(data_long_2, aes(x = patient_order, y = Correlation, fill = `Correlation Type`)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = NULL) +
  labs(title = "B", x = "Bottom 10 Patients",
       y = "Correlation") +
  scale_fill_manual(values = c("r" = "blue", "r_squared" = "red"),
                    labels = c("r" = "r", "r_squared" = expression(R^2))) +
  theme_minimal() + theme(plot.title = element_text(face = "bold"),
                          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

sam_plot_2

data_drug_2 <- read.table("drug_correlations.txt", header = T,sep = "\t")
View(data_drug_2)

data_drug <- arrange(data_drug_2, desc(data_drug_2$correlation))
data_drug <- na.omit(data_drug)
top_10_drug_1 <- data_drug %>%
  arrange(desc(correlation)) %>%
  slice(1:10)
View(top_10_drug_1)
write.table(top_10_drug_1, file = "top10_drugs_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)

down_10_drug <- data_drug %>%
  arrange(correlation) %>%
  slice(1:10)
View(down_10_drug)
write.table(down_10_drug, file = "bottom10_drug_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)
top_10_drug <- read.table("top10_drugs_correlations.txt", header = T, sep = "\t")
top_10_drug$drug <- factor(top_10_drug$drug)
top_10_drug <- top_10_drug %>%
  arrange(desc(correlation))
top_10_drug
top_10_drug$drug_order <- factor(top_10_drug$drug, levels = top_10_drug$drug)
data_long_3 <- top_10_drug %>%
  pivot_longer(cols = c(correlation, r_squared),
               names_to = "Correlation Type",
               values_to = "Correlation")
View(data_long_3)
# Create the side-by-side bar plot
drug_plot_1 <-ggplot(data_long_3, aes(x = drug_order, y = Correlation, fill = `Correlation Type`)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = NULL) +
  labs(title = "C",x = "Top 10 drug",
       y = "Correlation") +
  scale_fill_manual(values = c("correlation" = "blue", "r_squared" = "red"),
                    labels = c("correlation" = "r", "r_squared" = expression(R^2))) +
  theme_minimal() + theme(plot.title = element_text(face = "bold"),
                          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
drug_plot_1

down_10_drug <- read.table("bottom10_drug_correlations.txt", header = T, sep = "\t")
View(down_10_drug)
down_10_drug$drug <- factor(down_10_drug$drug)
down_10_drug <- down_10_drug %>%
  arrange(desc(correlation))
down_10_drug$drug_order <- factor(down_10_drug$drug, levels = down_10_drug$drug)
data_long_4 <- down_10_drug %>%
  pivot_longer(cols = c(correlation, r_squared),
               names_to = "Correlation Type",
               values_to = "Correlation")
View(data_long_4)
# Create the side-by-side bar plot
drug_plot_2 <- ggplot(data_long_4, aes(x = drug_order, y = Correlation, fill = `Correlation Type`)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = NULL) +
  labs(title = "D", x = "Bottom 10 drug", y = "Correlation") +
  scale_fill_manual(values = c("correlation" = "blue", "r_squared" = "red"),
                    labels = c("correlation" = "r", "r_squared" = expression(R^2))) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

drug_plot_2

final = ggarrange(sam_plot_1, sam_plot_2,drug_plot_1,drug_plot_2, nrow=2, ncol = 2, common.legend = TRUE, legend="right")
final
ggsave("comparision_of_correlations_MFP_based.pdf", final, width = 10, height = 7, units = "in", dpi = 300)
