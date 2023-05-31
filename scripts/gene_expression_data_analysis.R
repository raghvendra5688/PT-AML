library(readr)
library(data.table)
library(dplyr)
library(ggplot2)
library(factoextra)
library(ggfortify)
library(tidyverse)
library(cluster)
library(fpc)

setwd("C:/Users/siddh/OneDrive/Desktop/sid/Immunoinformatics-main/Immunoinformatics-main/Data")
data1 <- fread("Revised_Training_Set_with_Expr_Clin_PA_CTS_P1.csv.gz",header=T)
data2 <- fread("Revised_Training_Set_with_Expr_Clin_PA_CTS_P2.csv.gz",header=F)
data1 <- as.data.frame(data1)
data2 <- as.data.frame(data2)
colnames(data2) <- colnames(data1)
gene_expression_df <- rbind(data1,data2)
write.table(gene_expression_df, file = "gene_expression.txt", col.names = TRUE, row.names = TRUE)
rm(data,data1,data2)
head(gene_expression_df)
gene_expression_df <- gene_expression_df[, 1:22844]
gene_expression_df <- gene_expression_df[-171, ]
variances <- apply(gene_expression_df, 2, var)
variances <- sort(variances, decreasing = TRUE)
View(variances)
top100_columns <- names(variances)[1:100]
top100_dataframe <- gene_expression_df[, top100_columns]
View(top100_dataframe)
top100 <- sapply(top100_dataframe[, -1], as.numeric)

pca_100 = prcomp(top100[,-1], center = TRUE, scale = TRUE)
summary(pca_100)
pca_transform = as.data.frame(-pca_100$x[,1:2])
fviz_nbclust(pca_transform, kmeans, method = 'wss')
fviz_nbclust(pca_transform, kmeans, method = 'silhouette')
kmeans_result <- kmeans(pca_transform, centers = k)
fviz_nbclust(pca_transform, kmeans, method = 'gap_stat')
k = 6
kmeans_pca = kmeans(pca_transform, centers = k, nstart = 50)
diss_matrix <- dist(pca_transform)
avg_sil_width <- cluster.stats(diss_matrix, kmeans_pca$cluster)$avg.silwidth
avg_sil_width
sil_widths <- silhouette(kmeans_pca$cluster, dist(pca_transform))

fviz_cluster(kmeans_pca, data = pca_transform)
