library(data.table)
library(ComplexHeatmap)
library(circlize)
library(rstatix)
library(GetoptLong)
library(ggplot2)
library(sysfonts)
library(extrafont)
library(reticulate)
loadfonts()

setwd("~/TII/Projects/Raghav/Immunoinformatics/scripts/")

#Get the revised cell line data frame
cell_line_info_part1_df <- fread("../Data/Training_Set_Mod.csv",header=T,sep="\t")
cell_line_info_part1_df <- as.data.frame(cell_line_info_part1_df)
all_columns <- colnames(cell_line_info_part1_df)

sample_id <- all_columns[1]

#Gene Names
gene_names <- all_columns[2:653]

#Tsne dims
tsne_names <- all_columns[c(654,655)]

cell_line_age <- cell_line_info_part1_df$ageAtDiagnosis
cell_line_sex <- cell_line_info_part1_df$consensus_sex
cell_line_stage <- cell_line_info_part1_df$diseaseStageAtSpecimenCollection
cell_line_vital_status <- cell_line_info_part1_df$vitalStatus
cell_line_os <- cell_line_info_part1_df$overallSurvival

#Clinical characteristics
clinical_characteristics <- all_columns[c(661:674)]

#Pathway names 
pathway_names <- all_columns[675:728]

#Module names
celltype_names <- all_columns[729:734]

module_names <- all_columns[735:748]

mutated_genes <- all_columns[749:1121]

mutation_types <- all_columns[1122:1131]

#Make the matrix subset for each
gene_matrix <- scale(cell_line_info_part1_df[,c(gene_names,tsne_names)])
clinical_matrix <- cell_line_info_part1_df[,clinical_characteristics]
for (i in 1:ncol(clinical_matrix))
{
  clinical_matrix[,i] <- as.numeric(as.vector(clinical_matrix[,i]))
}
clinical_matrix <- scale(clinical_matrix)
celltype_matrix <- cell_line_info_part1_df[,celltype_names]
module_matrix <- cell_line_info_part1_df[,module_names]
mutation_gene_matrix <- cell_line_info_part1_df[,mutated_genes]
mutation_type_matrix <- scale(cell_line_info_part1_df[,mutation_types])
pathway_matrix <- cell_line_info_part1_df[,pathway_names]

order_ids <- order(cell_line_sex, cell_line_vital_status)

#Make the heatmap
ha = HeatmapAnnotation(
  Age = cell_line_age[order_ids],
  Sex = cell_line_sex[order_ids],
  Status = cell_line_vital_status[order_ids],
  OS = (cell_line_os[order_ids])/365,
  simple_anno_size = unit(5, "mm"),
  col = list(Age = colorRamp2(c(0,40,100),c("blue","white","red")),
             Sex = c("Female"="blue","Male"="red"),
             Status = c("Dead"="red","Unknown"="grey","Alive"="blue"),
             OS = colorRamp2(c(0,10),c("white","red"))),
  annotation_name_gp = gpar(fontsize=10, family="Open Sans", col="black"),
  annotation_legend_param = list(direction="horizontal", legend_gp = gpar(fontsize=10, family="Open Sans"), title_gp=gpar(fontsize=10, family="Open Sans")),
  show_legend=T)
col_fun_pathway <- colorRamp2(c(-0.5,0,0.5),c("blue","white","red"))
ht_pathway = Heatmap(matrix=t(as.matrix(pathway_matrix[order_ids,])), col_fun_pathway, 
                     name = "Pathway Activities",
                     width = unit(20, "cm"),
                     height = unit(3, "cm"),
                     cluster_columns = F,
                     cluster_rows = T,
                     row_names_centered = F,
                     show_row_names = F,
                     show_row_dend = F,
                     row_names_gp = gpar(fontsize=10, family="Open Sans"),
                     show_column_names = F,
                     #raster_quality = 2,
                     use_raster = F,
                     column_gap = unit(2, "mm"),
                     border_gp = gpar(col = "black", lty = 1),
                     border = T,
                     heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))
col_func_gene <- colorRamp2(c(-5,0,5),c("blue","white","red"))
ht_gene = Heatmap(matrix=t(as.matrix(gene_matrix[order_ids,])), col_func_gene,
                      name = "Gene Expr",
                      width = unit(20, "cm"),
                      height = unit(4, "cm"),
                      cluster_columns = F,
                      cluster_rows = T,
                      row_names_centered = F,
                      show_row_names = F,
                      show_row_dend = F,
                      show_column_names = F,
                      #raster_quality = 2,
                      use_raster = F,
                      column_gap = unit(2, "mm"),
                      border_gp = gpar(col = "black", lty = 1),
                      border = T,
                      heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))
col_fun_clinical <- colorRamp2(c(-2.0,0,2.0),c("blue","white","red"))
ht_clinical = Heatmap(matrix=t(as.matrix(clinical_matrix[order_ids,])), col_fun_clinical, 
                     name = "Clinical Traits",
                     width = unit(20, "cm"),
                     height = unit(3, "cm"),
                     cluster_columns = F,
                     cluster_rows = T,
                     row_names_centered = F,
                     show_row_names = F,
                     show_row_dend = F,
                     row_names_gp = gpar(fontsize=10, family="Open Sans"),
                     show_column_names = F,
                     top_annotation = ha,
                     column_title_rot = 90,
                     column_title_gp = gpar(fontsize=10, family="Open Sans"),
                     #raster_quality = 2,
                     use_raster = F,
                     column_gap = unit(2, "mm"),
                     border_gp = gpar(col = "black", lty = 1),
                     border = T,
                     heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))
col_fun_mut_genes <- colorRamp2(c(0,2),c("white","red"))
ht_mut_genes = Heatmap(matrix=t(as.matrix(mutation_gene_matrix[order_ids,])), col_fun_mut_genes, 
                     name = "Mutated Genes",
                     width = unit(20, "cm"),
                     height = unit(3, "cm"),
                     cluster_columns = F,
                     cluster_rows = T,
                     row_names_centered = F,
                     show_row_names = F,
                     show_row_dend = F,
                     row_names_gp = gpar(fontsize=10, family="Open Sans"),
                     show_column_names = F,
                     #raster_quality = 2,
                     use_raster = F,
                     column_gap = unit(2, "mm"),
                     border_gp = gpar(col = "black", lty = 1),
                     border = T,
                     heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))

col_fun_celltype <- colorRamp2(c(0,1),c("white","red"))
ht_celltype = Heatmap(matrix=t(as.matrix(celltype_matrix[order_ids,])), col_fun_celltype, 
                     name = "Celltype",
                     width = unit(20, "cm"),
                     height = unit(3, "cm"),
                     cluster_columns = F,
                     cluster_rows = T,
                     row_names_centered = F,
                     show_row_names = T,
                     show_row_dend=F,
                     row_names_gp = gpar(fontsize=10, family="Open Sans"),
                     show_column_names = F,
                     #raster_quality = 2,
                     use_raster = F,
                     column_gap = unit(2, "mm"),
                     border_gp = gpar(col = "black", lty = 1),
                     border = T,
                     heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))

col_fun_module <- colorRamp2(c(-0.5,0,0.5),c("blue","white","red"))
ht_module = Heatmap(matrix=t(as.matrix(module_matrix[order_ids,])), col_fun_module, 
                     name = "Module",
                     width = unit(20, "cm"),
                     height = unit(3, "cm"),
                     cluster_columns = F,
                     cluster_rows = T,
                     row_names_centered = F,
                     show_row_names = F,
                     show_row_dend=F,
                     row_names_gp = gpar(fontsize=10, family="Open Sans"),
                     show_column_names = F,
                     #raster_quality = 2,
                     use_raster = F,
                     column_gap = unit(2, "mm"),
                     border_gp = gpar(col = "black", lty = 1),
                     border = T,
                     heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))

col_fun_mut_type <- colorRamp2(c(0,3),c("white","red"))
ht_mut_type = Heatmap(matrix=t(as.matrix(mutation_type_matrix[order_ids,])), col_fun_mut_type, 
                     name = "Mutation Type",
                     width = unit(20, "cm"),
                     height = unit(3, "cm"),
                     cluster_columns = F,
                     cluster_rows = T,
                     row_names_centered = F,
                     show_row_names = T,
                     show_row_dend=F,
                     row_names_gp = gpar(fontsize=10, family="Open Sans"),
                     show_column_names = F,
                     #raster_quality = 2,
                     use_raster = F,
                     column_gap = unit(2, "mm"),
                     border_gp = gpar(col = "black", lty = 1),
                     border = T,
                     heatmap_legend_param = list(direction = "horizontal", title_gp = gpar(fontsize=10, family="Open Sans"), legend_gp = gpar(fontsize=10, family="Open Sans")))

ht_list = ht_clinical %v% ht_gene %v% ht_pathway %v% ht_mut_genes %v% ht_mut_type %v% ht_celltype %v% ht_module
tiff(file="../Results/Cell_Type_Fig1b.tiff", width = 12, height=15, units="in", family = "Open Sans", res=300)
draw(ht_list, heatmap_legend_side="bottom")
dev.off()
