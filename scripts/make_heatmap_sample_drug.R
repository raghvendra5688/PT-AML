library(dplyr)
library(ggplot2)
library(ComplexHeatmap)
library(data.table)
library(circlize)
library(colorspace)
library(RColorBrewer)
library(Matrix)

#Setwd 
setwd(".")

identify_problematic_combs <- function(mat, min_shared_fields = 1) {
  exclude_rows <- NULL
  exclude_cols <- NULL
  stopifnot(is.matrix(mat))
  
  ## Loop over candidate removals
  for (k in 1:nrow(mat)) {
    candidate_rows <- setdiff(1:nrow(mat), exclude_rows)
    problem_row_combs <- NULL
    for (i in candidate_rows) {
      i_idx <- which(candidate_rows == i)
      for (j in candidate_rows[i_idx:length(candidate_rows)]) {
        if (sum(!is.na(mat[i, ]) & !is.na(mat[j, ])) <= min_shared_fields) {
          problem_row_combs <- rbind(problem_row_combs, c(i, j))
        }
      }
    }
    if (is.null(problem_row_combs)) break
    exclude_rows <- c(exclude_rows,
                      as.integer(names(which.max(table(problem_row_combs)))))
  }
  
  for (k in 1:ncol(mat)) {
    candidate_cols <- setdiff(1:ncol(mat), exclude_cols)
    problem_col_combs <- NULL
    for (i in candidate_cols) {
      i_idx <- which(candidate_cols == i)
      for (j in candidate_cols[i_idx:length(candidate_cols)]) {
        if (sum(!is.na(mat[, i]) & !is.na(mat[, j])) <= min_shared_fields) {
          problem_col_combs <- rbind(problem_col_combs, c(i, j))
        }
      }
    }
    if (is.null(problem_col_combs)) break
    exclude_cols <- c(exclude_cols,
                      as.integer(names(which.max(table(problem_col_combs)))))
  }
  
  return(list('row' = exclude_rows, 'column' = exclude_cols))
}

remove_problematic_combs <- function() {
  problematic_combs <- identify_problematic_combs(
    mat = mat, min_shared_fields = min_shared_fields)
  if (!is.null(problematic_combs$row)) {
    mat <- mat[-problematic_combs$row, ]
  }
  if (!is.null(problematic_combs$column)) {
    mat <- mat[, -problematic_combs$column]
  }
  return(mat)
}
formals(remove_problematic_combs) <- formals(identify_problematic_combs)

################################################################################
data_df <- fread("../Results/Catboost_MFP_Feat_Var_supervised_test_predictions.csv",header=T,sep="\t")
data_df <- as.data.frame(data_df)

#Make MAE
data_df$mae <- abs(data_df$labels-data_df$predictions)
data_df$nmae <- data_df$mae/300

#Create the matrix of Normalized MAE values
mae_matrix <- as.matrix(Matrix(table(data_df$dbgap_rnaseq_sample,data_df$inhibitor)))
mae_matrix[mae_matrix==0] <- NA
for (i in 1:nrow(data_df))
{
  sample_id <- data_df$dbgap_rnaseq_sample[i]
  inhibitor_id <- data_df$inhibitor[i]
  nmae_val <- data_df$nmae[i]
  mae_matrix[sample_id,inhibitor_id] <- nmae_val
}

rev_mae_matrix <- remove_problematic_combs(mae_matrix, min_shared_fields=50)

# correlations
correlations <- split(data_df, data_df$inhibitor) %>%
  lapply(function(data) cor(data$predictions, data$labels, method = "pearson"))
correlations_df <- data.frame(drug = names(correlations),correlation = unlist(correlations))
subset_correlations_df <- correlations_df[rownames(correlations_df) %in% colnames(rev_mae_matrix),]
correlation_vec <- subset_correlations_df$correlation
names(correlation_vec) <- rownames(subset_correlations_df)

# Create a heatmap
################################################################################
col_fun = colorRamp2(c(0,0.125,0.25),c("blue","white","red"))
col_fun2 = colorRamp2(c(0,0.4,0.8),c("blue","white","red"))
row_ha = rowAnnotation(r=correlation_vec, col=list(r=col_fun2))
pdf("../Results/Drug_vs_Sample_Best_Catboost.pdf", height = 12, width = 16)
ht <- ComplexHeatmap::Heatmap(t(rev_mae_matrix), 
              name="NMAE",
              column_title = "Drug vs Patient Heatmap",
              na_col = "grey",
              rect_gp = gpar(col = "white", lwd = 1),
              column_title_gp = gpar(fill = "grey", col = "white", border = "black"),
              cluster_columns = T,
              show_column_dend = F,
              show_row_dend = F,
              clustering_distance_rows = "pearson",
              clustering_distance_columns = "pearson",
              clustering_method_rows = "centroid",
              clustering_method_columns = "centroid",
              row_names_gp = gpar(fontsize = 8, fontface="plain"),
              column_names_gp = gpar(fontsize = 8, fontface="plain"),
              right_annotation = row_ha,
              col=col_fun)
draw(ht)
dev.off()