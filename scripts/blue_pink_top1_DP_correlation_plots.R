library(readr)
library(tidyr)
library(gridExtra)
library(ggpubr)
data_sam <- read.table("samples_correlations.txt", header = T, sep = "\t")
data_sam <- arrange(data_sam, desc(data_sam$r))
data_sam <- na.omit(data_sam)
down_10_sam<- tail(data_sam,10)
View(data_sam)
top_10_sam <- head(data_sam, 10)
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
  labs(title = "A",x = "Top Best Patients",
       y = "Correlation") +
  scale_fill_manual(values = c("r" = "blue", "r_squared" = "magenta"),
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
  labs(title = "B", x = "Bottom Worse Patients",
       y = "Correlation") +
  scale_fill_manual(values = c("r" = "blue", "r_squared" = "magenta"),
                    labels = c("r" = "r", "r_squared" = expression(R^2))) +
  theme_minimal() + theme(plot.title = element_text(face = "bold"),
                          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

sam_plot_2

sam = grid.arrange(sam_plot_1, sam_plot_2, ncol = 2, top = "Comparison of Correlations for Patients")+plot_layout(guides = 'collect')
sam_2 = ggarrange(sam_plot_1, sam_plot_2, ncol=2, common.legend = TRUE, legend="right")
sam_2
ggsave("combined_plot_patients.png", sam, width = 10, height = 5, units = "in", dpi = 300)

data_drug <- merge(correlations_df,correlations_df_sp)
View(data_drug)
write.table(drug_data_drug, file = "drug_correlations.txt", sep = "\t", quote = FALSE, row.names = FALSE)
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
View(top_10_drug)
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
  labs(title = "C",x = "Top Best Drug",
       y = "Correlation") +
  scale_fill_manual(values = c("correlation" = "blue", "r_squared" = "magenta"),
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
  labs(title = "D", x = "Bottom Worse Drug", y = "Correlation") +
  scale_fill_manual(values = c("correlation" = "blue", "r_squared" = "magenta"),
                    labels = c("correlation" = "r", "r_squared" = expression(R^2))) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

drug_plot_2

drug = grid.arrange(drug_plot_1, drug_plot_2, ncol = 2, top = "Comparison of Correlations for Drugs")
drug_2 = ggarrange(drug_plot_1, drug_plot_2, ncol=2, common.legend = TRUE, legend="right")
drug_2
final = ggarrange(sam_plot_1, sam_plot_2,drug_plot_1,drug_plot_2, nrow=2, ncol = 2, common.legend = TRUE, legend="right")
final
library(ggplot2.utils)
library(cowplot)
ggdraw()
final_title <- ggdraw() +
   draw_label("Comparison of Correlations", size = 16, x = 0.5, hjust = 0.5)
final_combined_plot <- ggarrange(final, nrow = 2, heights = c(0.1, 9))
final_combined_plot_with_margin <- final_combined_plot +
  theme(plot.margin = margin(t = 15, unit = "mm"))
final_combined_plot_with_margin


ggsave("1_check_comparision_of_correlations_MFP_based.pdf", final, width = 10, height = 7, units = "in", dpi = 300)

# Save as SVG
ggsave("1_check_comparision_of_correlations_MFP_based.svg", final, width = 10, height = 7, units = "in", dpi = 300)
