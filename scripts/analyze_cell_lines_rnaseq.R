library(data.table)
library(ggplot2)
library(gplots)
library(ComplexHeatmap)
library(BiocManager)
library(GSVA)
library(BiocParallel)
library(doParallel)
library(stringr)
library(circlize)
library(RColorBrewer)
library(ggpubr)
library(stringr)
library(GSA)
library(Matrix)
library(extrafont)
library(Rtsne)
library(immunedeconv)
library(Polychrome)
loadfonts()
registerDoParallel(20)
ht_opt$message = FALSE

setwd("/research/groups/kannegrp/home/rmall/projects/Raghav/Drug_Sensitivity_PANoptosis/")
source("scripts/all_cells_functions.R")

#Get the RNAseq data from all the cell lines
cell_lines_df <- read.table("Data/Cell_Lines_Full_Expr.csv",header=T,sep="\t")
cell_names <- rownames(cell_lines_df)
gene_names <- colnames(cell_lines_df)
id <- which(gene_names=="GSDME")
gene_names[id] <- "DFNA5"
colnames(cell_lines_df) <- gene_names
  
#Get the cell line meta annotation
cell_lines_metadata <- fread("Data/sample_info.csv",header=T)
cell_lines_metadata_of_interest <- cell_lines_metadata[cell_lines_metadata$DepMap_ID %in% cell_names,]

#Consider only those cell lines which have COSMICID
cell_lines_moi_COSMIC <- cell_lines_metadata_of_interest[!is.na(cell_lines_metadata_of_interest$COSMICID),]
cell_lines_COSMIC_df <- cell_lines_df[rownames(cell_lines_df) %in% cell_lines_moi_COSMIC$DepMap_ID,]
cell_lines_COSMIC_df <- cell_lines_COSMIC_df[sort(rownames(cell_lines_COSMIC_df)),]
cell_lines_moi_COSMIC <- cell_lines_moi_COSMIC[order(cell_lines_moi_COSMIC$DepMap_ID),]

#Make the t-SNE plot using full cell line dataset
full_expr_tsne_out <- Rtsne(X=as.matrix(cell_lines_COSMIC_df), dims=2, perplexity = 10.0, pca_center = F, pca_scale = F, pca = F)
full_expr_tsne_df <- as.data.frame(full_expr_tsne_out[["Y"]])
full_expr_tsne_df$Pheno <- cell_lines_moi_COSMIC$lineage
unique_phenotypes <- unique(full_expr_tsne_df$Pheno)
full_expr_tsne_df$Pheno <- factor(full_expr_tsne_df$Pheno, levels = unique_phenotypes)

# create your own color palette based on `seedcolors`
P28 = createPalette(28,  c("#ff0000", "#00ff00", "#0000ff"))
palette(P28)
#swatch(P28)
g_tsne <- ggplot(data=full_expr_tsne_df, aes(x=V1, y=V2, color=Pheno)) + geom_point(aes(color=Pheno), size=2) + theme_bw() + xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
  scale_color_manual(name="Sample",labels = unique_phenotypes, values = as.vector(P28) )+
  theme(legend.title.align=0.0, legend.text.align = 0.0) +
  theme(axis.text.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        strip.text = element_text(color = "white", size=20, angle=0, hjust = 0.5, vjust = 0.5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 16, angle = 0, hjust = 1, vjust = 0, face = "plain"),
        axis.title.x = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(color="grey20", size=18, angle = 0, face="plain"),
        title=element_text(color="grey20", size=20, face="plain"))
ggsave(filename=paste0("results/Full_Expr_T-SNE_plot.pdf"), plot = g_tsne, device = pdf(), height = 6, width=12, units="in")
dev.off()
#############################################################################################################################################
#Build the cell line dataset of interest with COSMICID, Sanger ID, DepMap_ID
rev_cell_lines_moi_COSMIC <- cell_lines_moi_COSMIC[,c("DepMap_ID","cell_line_name","stripped_cell_line_name","CCLE_Name","COSMICID","Sanger_Model_ID","sex","age","lineage","primary_or_metastasis")]

cell_lines_COSMIC_mat <- as.matrix(cell_lines_COSMIC_df)
rownames(cell_lines_COSMIC_mat) <- rownames(cell_lines_COSMIC_df)
colnames(cell_lines_COSMIC_mat) <- colnames(cell_lines_COSMIC_df)

#Get the scaled expression for the full list of panoptosis + inflammasome markers
scaled_cell_lines_COSMIC_mat <- as.data.frame(t(scale(t(cell_lines_COSMIC_mat))))

##############################################################################################################333
# #Get the ids of cancer cell lines associated with pancreatic cancer along with expression of our panoptosis genes of interest
# ids <- c(grep("BXPC",rev_cell_lines_moi_COSMIC$stripped_cell_line_name),grep("HPAFII",rev_cell_lines_moi_COSMIC$stripped_cell_line_name),
#          grep("CAPAN",rev_cell_lines_moi_COSMIC$stripped_cell_line_name),grep("SW1990",rev_cell_lines_moi_COSMIC$stripped_cell_line_name),
#          grep("PANC",rev_cell_lines_moi_COSMIC$stripped_cell_line_name))

panoptosis_full_genes <- c("ADAR", "AIM2", "CASP1", "CASP10", "CASP14", "CASP2","CASP3","CASP4","CASP5","CASP6","CASP7","CASP8","CASP9","CFLAR","DDX3X",
                           "DDX58","FADD","GSDMB","GSDMC","GSDMD","DFNA5","IFIH1","IL18","IL1A","IL1B","IRF1","IRF2","IRF3","IRF4","IRF5","IRF6","IRF7",
                           "IRF8","IRF9","MAVS","MEFV","MLKL","MYD88","NLRP3","NLRP1","NLRP9","NR2C2","PANX1","PANX2","PANX3","PSTPIP1","PSTPIP2","PYCARD","RBCK1",
                           "RIPK1","RIPK2","RIPK3","RNF31","SHARPIN","SYK","TICAM1","TNF","TNFAIP3","TP53","ZBP1")
# pancreatic_cancer_matrix <- t(scaled_cell_lines_COSMIC_mat[ids,panoptosis_full_genes])
# colnames(pancreatic_cancer_matrix) <- rev_cell_lines_moi_COSMIC[ids,]$cell_line_name
# ht <- make_heatmap(pancreatic_cancer_matrix)
# pdf("results/Pancreatic_Cancer_Expr_Matrix_for_Ratnakar.pdf", height = 12, width =4)
# draw(ht, heatmap_legend_side="bottom")
# dev.off()

##Get the ICR genes
icr_genes <- c("IFNG", "TXB21", "CD8B", "CD8A", "IL12B", "STAT1", "IRF1", "CXCL9", "CXCL10", "CCL5", 
               "GNLY", "PRF1", "GZMA", "GZMB", "GZMH", "CD274", "PDCD1", "CTLA4", "FOXP3", "IDO1")

#Get the inflammasome genesets (from REACTIOME, GO BP, KEGG)
inflammasome_geneset <- read.table("Data/Inflammasome_geneset.txt",header=T)
go_inflammasome_geneset <- read.table("Data/GO_Inflammasome_geneset.txt",header=T)
nlr_geneset <- read.table("Data/NLR_geneset.txt",header=T)
pyroptosis_geneset <- read.table("Data/Pyroptosis_geneset.txt",header=T)
necroptosis_geneset <- read.table("Data/Necroptosis_geneset.txt",header=T)

genes_of_interest <- union(union(union(union(union(union(panoptosis_full_genes, icr_genes), inflammasome_geneset$REACTOME_INFLAMMASOMES),
                           nlr_geneset$KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY), go_inflammasome_geneset$GOBP_INFLAMMASOME_COMPLEX_ASSEMBLY),
                           pyroptosis_geneset$REACTOME_PYROPTOSIS),necroptosis_geneset$GOBP_NECROPTOTIC_SIGNALING_PATHWAY)
inflammasome_expr_scaled_cell_lines_COSMIC_mat <- scaled_cell_lines_COSMIC_mat[,colnames(scaled_cell_lines_COSMIC_mat) %in% genes_of_interest]

#Get cancer associated pathways and panoptosis pathway with its activity for each cell lines
#load("Data/Selected.pathways.3.4.RData")
#apoptosis_geneset <- Selected.pathways$`[HM] Apoptosis`
Selected.pathways = list()
#Selected.pathways[["[HM] Apoptosis"]] <- apoptosis_geneset
Selected.pathways[["[MRK] PANoptosis"]] <- panoptosis_full_genes
Selected.pathways[["[DB] ICD"]] <- icr_genes
Selected.pathways[["[KG] NLR Signaling"]] <- nlr_geneset$KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY
Selected.pathways[["[RT] Inflammasome"]] <- inflammasome_geneset$REACTOME_INFLAMMASOMES
Selected.pathways[["[GO] Inflammasome Complex"]] <- go_inflammasome_geneset$GOBP_INFLAMMASOME_COMPLEX_ASSEMBLY
Selected.pathways[["[RT] Pyroptosis"]] <- pyroptosis_geneset$REACTOME_PYROPTOSIS
Selected.pathways[["[GO] Necroptosis"]] <- necroptosis_geneset$GOBP_NECROPTOTIC_SIGNALING_PATHWAY

rev_hallmark_pathways_of_interest <- Selected.pathways
rev_ssgsea_results <- t(gsva(expr = t(cell_lines_COSMIC_mat), gset.idx.list = rev_hallmark_pathways_of_interest, method="ssgsea", kcdf = "Gaussian", parallel.sz=20))

#There is no immune compartment in cancer cell lines
################################################################################
##Get the bindea gene list of immune cell type categorizing geneset and estimate its activity for each cell line
#load("Data/immune.gene.lists.v3.Rdata")
#immune_activity_ssgsea_results <- t(gsva(expr = t(cell_lines_COSMIC_mat), gset.idx.list = Bindea_REV1 , method="ssgsea", kcdf = "Gaussian", parallel.sz=20))

##Get the immune deconvolution using MCP counter and transform the data accordingly
#immune_cell_type_conc <- immunedeconv::deconvolute(t(cell_lines_COSMIC_mat), "mcp_counter")
#immune_cell_type_conc <- as.data.frame(immune_cell_type_conc)
#cell_type_info <- immune_cell_type_conc[,1]
#immune_cell_type_conc <- immune_cell_type_conc[,c(2:ncol(immune_cell_type_conc))]
#rownames(immune_cell_type_conc) <- cell_type_info
#immune_cell_mcp_counter_results <- t(immune_cell_type_conc)

#Order all the cell lines + panoptosis gene expression + pathway activity information
rev_cell_lines_moi_COSMIC <- rev_cell_lines_moi_COSMIC[order(rev_cell_lines_moi_COSMIC$DepMap_ID),]
scaled_cell_lines_COSMIC_mat <- scaled_cell_lines_COSMIC_mat[order(rownames(scaled_cell_lines_COSMIC_mat)),]
cell_lines_COSMIC_mat <- cell_lines_COSMIC_mat[order(rownames(cell_lines_COSMIC_mat)),]

#Get the SSGSEA results for pathways
rev_ssgsea_results <- rev_ssgsea_results[order(rownames(rev_ssgsea_results)),]
colnames(rev_ssgsea_results) <- paste0("SSGSEA_",colnames(rev_ssgsea_results))

##Get the SSGSEA results for Immune Celltypes
#immune_activity_ssgsea_results <- immune_activity_ssgsea_results[order(rownames(immune_activity_ssgsea_results)),]
#colnames(immune_activity_ssgsea_results) <- paste0("SSGSEA_",colnames(immune_activity_ssgsea_results))

##Get the Immune Cell type composition information
#immune_cell_mcp_counter_results <- immune_cell_mcp_counter_results[order(rownames(immune_cell_mcp_counter_results)),]
#colnames(immune_cell_mcp_counter_results) <- paste0("MCP_Counter_",colnames(immune_cell_mcp_counter_results))

#Get all the genes' expression matrix which is scaled (as required by most ML models)
inflammasome_expr_scaled_cell_lines_COSMIC_mat <- inflammasome_expr_scaled_cell_lines_COSMIC_mat[order(rownames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)),]

#Get the mutation information for each cell line, gene combination
################################################################################
mutational_info <- fread("Data/gene_cellline_somatic_mutation_edges.txt",header=T)
mutational_info <- as.data.frame(mutational_info)
mutational_info$CellLine <- gsub(pattern="-",replacement = "",mutational_info$CellLine)
mutational_info$CellLine <- gsub(pattern=" ",replacement = "",mutational_info$CellLine)
mutational_info$CellLine <- gsub(pattern="_",replacement = "",mutational_info$CellLine)
mutational_info$CellLine <- gsub(pattern="\\/",replacement = "",mutational_info$CellLine)
unique_cell_lines <- unique(mutational_info$CellLine)
genes_of_interest_present <- genes_of_interest[genes_of_interest %in% colnames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)]
subset_mutational_info <- mutational_info[mutational_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name & mutational_info$GeneSym %in% genes_of_interest_present,]

#Make the mutation matrix
mutation_matrix <- Matrix(0, nrow=nrow(rev_cell_lines_moi_COSMIC), ncol=ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
colnames(mutation_matrix) <- colnames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)
for (i in 1:nrow(rev_cell_lines_moi_COSMIC))
{
  temp <- rep(0, ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
  sub_mutational_info <- subset_mutational_info[subset_mutational_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name[i],]
  if (nrow(sub_mutational_info)>0)
  {
    for (j in 1:nrow(sub_mutational_info))
    {
      gene_name <- sub_mutational_info$GeneSym[j]
      weight <- sub_mutational_info$weight[j]
      mutation_matrix[i,gene_name] <- weight
    }
  }
  else{
    mutation_matrix[i,] <- temp 
  }
}
rownames(mutation_matrix) <- rev_cell_lines_moi_COSMIC$stripped_cell_line_name
mutation_matrix <- as.matrix(mutation_matrix)
colnames(mutation_matrix) <- paste0("Mutation_",colnames(mutation_matrix))

#Make the CNV matrix
################################################################################
cnv_info <- fread("Data/gene_cellline_cnv_edges.txt",header=T)
cnv_info <- as.data.frame(cnv_info)
cnv_info[cnv_info$CellLine=="TT",]$CellLine <- "TDOTT"
cnv_info$CellLine <- gsub(pattern="-",replacement = "",cnv_info$CellLine)
cnv_info$CellLine <- gsub(pattern=" ",replacement = "",cnv_info$CellLine)
cnv_info$CellLine <- gsub(pattern="_",replacement = "",cnv_info$CellLine)
cnv_info$CellLine <- gsub(pattern="\\/",replacement = "",cnv_info$CellLine)
unique_cell_lines <- unique(cnv_info$CellLine)
subset_cnv_info <- cnv_info[cnv_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name & cnv_info$GeneSym %in% genes_of_interest_present,]

#Make the cnv matrix
cnv_matrix <- Matrix(0, nrow=nrow(rev_cell_lines_moi_COSMIC), ncol=ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
colnames(cnv_matrix) <- colnames(inflammasome_expr_scaled_cell_lines_COSMIC_mat)
for (i in 1:nrow(rev_cell_lines_moi_COSMIC))
{
  temp <- rep(0, ncol(inflammasome_expr_scaled_cell_lines_COSMIC_mat))
  sub_cnv_info <- subset_cnv_info[subset_cnv_info$CellLine %in% rev_cell_lines_moi_COSMIC$stripped_cell_line_name[i],]
  if (nrow(sub_cnv_info)>0)
  {
    for (j in 1:nrow(sub_cnv_info))
    {
      gene_name <- sub_cnv_info$GeneSym[j]
      weight <- sub_cnv_info$weight[j]
      cnv_matrix[i,gene_name] <- weight
    }
  }
  else{
    cnv_matrix[i,] <- temp 
  }
}
rownames(cnv_matrix) <- rev_cell_lines_moi_COSMIC$stripped_cell_line_name
cnv_matrix <- as.matrix(cnv_matrix)
colnames(cnv_matrix) <- paste0("CNV_",colnames(cnv_matrix))

#Combine the cell line meta data + pathway activity + immune cell activity + all gene expression
###############################################################################
combined_COSMIC_cell_line_df <- as.data.frame(cbind(rev_cell_lines_moi_COSMIC,
                                                    rev_ssgsea_results, 
                                                    mutation_matrix, 
                                                    cnv_matrix,
                                                    inflammasome_expr_scaled_cell_lines_COSMIC_mat))

#Make tSNE plot with the modified dataset for each cell line that we have
mod_tsne_out <- Rtsne(X=as.matrix(combined_COSMIC_cell_line_df[,c(11:ncol(combined_COSMIC_cell_line_df))]), dims=2, perplexity = 5.0, pca_center = F, pca_scale = F, pca = F)
mod_tsne_df <- as.data.frame(mod_tsne_out[["Y"]])
mod_tsne_df$Pheno <- rev_cell_lines_moi_COSMIC$lineage
mod_unique_phenotypes <- unique(mod_tsne_df$Pheno)
mod_tsne_df$Pheno <- factor(mod_tsne_df$Pheno, levels = mod_unique_phenotypes)

g_tsne_mod <- ggplot(data=mod_tsne_df, aes(x=V1, y=V2, color=Pheno)) + geom_point(aes(color=Pheno), size=2) + theme_bw() + xlab("T-SNE Dim1") + ylab("T-SNE Dim2") +
  scale_color_manual(name="Sample",labels = mod_unique_phenotypes, values = as.vector(P28))+
  theme(legend.title.align=0.0, legend.text.align = 0.0) +
  theme(axis.text.x = element_text(color = "grey20", size = 16, angle = 0, hjust = .5, vjust = .5, face = "plain"),
        strip.text = element_text(color = "white", size=20, angle=0, hjust = 0.5, vjust = 0.5, face = "plain"),
        axis.text.y = element_text(color = "grey20", size = 16, angle = 0, hjust = 1, vjust = 0, face = "plain"),
        axis.title.x = element_text(color = "grey20", size = 20, angle = 0, hjust = .5, vjust = 0, face = "plain"),
        axis.title.y = element_text(color = "grey20", size = 20, angle = 90, hjust = .5, vjust = .5, face = "plain"),
        legend.text = element_text(color="grey20", size=18, angle = 0, face="plain"),
        title=element_text(color="grey20", size=20, face="plain"))
ggsave(filename=paste0("results/Mod_T-SNE_plot.pdf"), plot = g_tsne_mod, device = pdf(), height = 6, width=12, units="in")
dev.off()

#Remove columns with no variance
col_ids_to_remove <- which(unlist(lapply(combined_COSMIC_cell_line_df[,c(11:ncol(combined_COSMIC_cell_line_df))],sd))==0)+10
rev_combined_COSMIC_cell_line_df <- combined_COSMIC_cell_line_df[,-c(col_ids_to_remove)]

#Make the metadata information to save and use for ablation studies
metadata_info <- as.data.frame(cbind(c(1:ncol(combined_COSMIC_cell_line_df)),colnames(combined_COSMIC_cell_line_df)))
colnames(metadata_info) <- c('Index','Columns')
metadata_info$Index <- as.numeric(as.vector(metadata_info$Index))
metadata_info$Columns <- as.character(as.vector(metadata_info$Columns))
metadata_info$Type <- "Clinical_Features"
metadata_info[11:17,]$Type <- "Pathway_Features"
metadata_info[c(18:185),]$Type <- "Mutation_Features"
metadata_info[c(186:353),]$Type <- "CNV_Features"
metadata_info[c(354:521),]$Type <- "Expr_Features"
rev_metadata_info  <- metadata_info[-c(col_ids_to_remove),]

write.table(rev_metadata_info, file = "Data/Cell_Lines_Metdata_Description_Inflammasome.csv", row.names=T, col.names=T, quote=F, sep="\t")
write.table(rev_combined_COSMIC_cell_line_df, file="Data/Cell_Lines_Metadata_Pathway_Activities_PANoptosis_Markers_Expr_Inflammasome.csv", col.names=T, row.names=F, sep="\t", quote=F)
write.table(scaled_cell_lines_COSMIC_mat, file="Data/Cell_Lines_Full_Scaled_Expr.csv",row.names=T, col.names=T, sep="\t", quote=F)

#Convert pathway geneset to an edgelist
pathway_gene_edgelist <- NULL
for (i in 1:length(Selected.pathways))
{
  pathway_name <- names(Selected.pathways)[i]
  geneset <- Selected.pathways[[i]]
  temp <- cbind(rep(pathway_name,length(geneset)),geneset)
  pathway_gene_edgelist <- rbind(pathway_gene_edgelist, temp)
}
pathway_gene_edgelist <- as.data.frame(pathway_gene_edgelist)
colnames(pathway_gene_edgelist) <- c("Pathway","Gene")
write.table(pathway_gene_edgelist, file="Data/Pathways_Genes_Edgelist_Inflammasome.csv", row.names=F, col.names=T, quote=F, sep="\t")