### INTRO ###

# In this file we provide the code to reproduce the the single cell RNA sequencing plots in our hypoblast gene expression activation paper titled,
# "Pdgfra expression in the early ICM and later hypoblast lineage confirms the existence of an undefined population from which Epiblast and Hypoblast arise".
# The majority of this workflow is performed in python. However, the pseudotime analysis is performed in R because Slingshot is coded in R. In the workflow
# we make it clear when the user must switch to an R envionment.

### Download the data ###

# The single cell RNA-sequencing data and related files are inherited from the Radley et al. 2022 paper titled,
# "Entropy sorting of single-cell RNA sequencing data reveals the inner cell mass in the human pre-implantation embryo".
# To carry out the follow workflow, start by downloading the single cell data from the following online repository, https://data.mendeley.com/datasets/689pm8s7jc

### Set path to data ###
# Set the path variable to wherever you decide to store the downlaoded Meistermann pre-implantation embryo data.
path = "/home/ahr35/ES_Paper_Data/ES_Paper_Data/Pre_Implantation_Human_Embryo_Data/Meistermann_Data/"

### Import packages ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

def logifunc(x,A,x0,k,off):
    return A / (1 + np.exp(-k*(x-x0)))+off

palette = ["#ebac23", "#b80058", "#008cf9", "#5954d6", "#00bbad", 
           "#006e00", "#d163e6", "#b24502", "#00c6f8", 
           "#878500", "#00a76c", "#ff9287", "#bdbdbd",
           "#F61503","#3ED544","#B784ED"]

sns.set_palette(sns.color_palette(palette, 16))

### Load data ###
# Meistermann et al. 2021 Data
Human_Sample_Info = pd.read_csv(path+"Human_Sample_Info.csv",header=0,index_col=0)
Human_Embryo_Counts = pd.read_csv(path+"exprDatRaw.tsv",sep='\t',header=0,index_col=0).T
# Radley et al. 2022 UMAP embedding
UMAP_Embedding = np.load(path+"Chosen_Genes_Embedding.npy")
# Radley et al. 2022 highly structured genes
Chosen_Gene_IDs = np.load(path+"Chosen_Gene_IDs.npy",allow_pickle=True)
Chosen_Gene_ID_Clusters = np.load(path+"Chosen_Gene_ID_Clusters.npy")
Chosen_Gene_IDs = Chosen_Gene_IDs[np.where(np.isin(Chosen_Gene_ID_Clusters,np.array([0,3])))[0]]

### Plot UMAP Embedding ###

### UMAP embedding with Stirparo et al. 2018 cell type labels

plt.figure(figsize=(4,4))
Labels = np.asarray(Human_Sample_Info["Stirparo_Labels"])
Unique_Labels = np.unique(Labels[pd.isnull(Labels)==0])
plt.title("Stirparo et al. Cell Labels", fontsize=14)
for i in np.arange(Unique_Labels.shape[0]):
    IDs = np.where(Labels == Unique_Labels[i])
    Layer = 1
    if Unique_Labels[i] == 'TE':
        Layer = -1
    plt.scatter(UMAP_Embedding[IDs,0],UMAP_Embedding[IDs,1],s=6,zorder=Layer,label=Unique_Labels[i])

plt.scatter(UMAP_Embedding[:,0],UMAP_Embedding[:,1],s=3,zorder=-2,label="N/A",facecolors="None",edgecolors="grey")
plt.xticks([])
plt.yticks([])
plt.legend(prop={"size":8})
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(path + "_Celltypes.png",dpi=800)
plt.close()

### Subset the data down to the Morula -> Epi/Hyp cells by segmenting the Radley et al. 2022 UMAP embedding

x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -1.7, UMAP_Embedding[:,0] < 7))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > -1, UMAP_Embedding[:,1] < 8.5))[0]
Localised_Cells = np.intersect1d(x_range,y_range)
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 2, UMAP_Embedding[:,0] < 2.7))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 0.5, UMAP_Embedding[:,1] < 1.1))[0]
Remove_Cells_1 = np.intersect1d(x_range,y_range)
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -2.5, UMAP_Embedding[:,0] < 0.6))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 2.25, UMAP_Embedding[:,1] < 4.4))[0]
Remove_Cells_2 = np.intersect1d(x_range,y_range)
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -2.5, UMAP_Embedding[:,0] < -0.25))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 7.23, UMAP_Embedding[:,1] < 8.6))[0]
Remove_Cells_3 = np.intersect1d(x_range,y_range)
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -0.4, UMAP_Embedding[:,0] < -0.6))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 2.8, UMAP_Embedding[:,1] < 4.4))[0]
Remove_Cells_4 = np.intersect1d(x_range,y_range)
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -2, UMAP_Embedding[:,0] < -0.55))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 4.3, UMAP_Embedding[:,1] < 6))[0]
Remove_Cells_5 = np.intersect1d(x_range,y_range)
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 4.4, UMAP_Embedding[:,0] < 6.9))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > -0.7, UMAP_Embedding[:,1] < 1.5))[0]
Remove_Cells_6 = np.intersect1d(x_range,y_range)

Localised_Cells = np.delete(Localised_Cells,np.where(np.isin(Localised_Cells,Remove_Cells_1)))
Localised_Cells = np.delete(Localised_Cells,np.where(np.isin(Localised_Cells,Remove_Cells_2)))
Localised_Cells = np.delete(Localised_Cells,np.where(np.isin(Localised_Cells,Remove_Cells_3)))
Localised_Cells = np.delete(Localised_Cells,np.where(np.isin(Localised_Cells,Remove_Cells_4)))
Localised_Cells = np.delete(Localised_Cells,np.where(np.isin(Localised_Cells,Remove_Cells_5)))
Localised_Cells = np.delete(Localised_Cells,np.where(np.isin(Localised_Cells,Remove_Cells_6)))
Localised_Cells_IDs = Human_Sample_Info.index[Localised_Cells]

### Re-plot the UMAP embedding with Stirparo et al. 2018 cell type labels for this subset of the data

Labels = np.asarray(Human_Sample_Info["Stirparo_Labels"])
All_Unique_Lables = np.unique(Labels[pd.isnull(Labels)==0])
plt.figure(figsize=(2,2))
Labels = np.asarray(Human_Sample_Info["Stirparo_Labels"].iloc[Localised_Cells])
Unique_Labels = np.unique(Labels[pd.isnull(Labels)==0])
plt.title("Morula - Hypoblast Subset",fontsize=10)
for i in np.arange(Unique_Labels.shape[0]):
    IDs = np.where(Labels == Unique_Labels[i])
    UMAP_IDs = Localised_Cells[IDs]
    Colour_Ind = int(np.where(All_Unique_Lables == Unique_Labels[i])[0])
    Layer = 1
    if Unique_Labels[i] == 'TE':
        Layer = -1
    if UMAP_IDs.shape[0] > 0:
        plt.scatter(UMAP_Embedding[UMAP_IDs,0],UMAP_Embedding[UMAP_IDs,1],s=6,zorder=Layer,label=Unique_Labels[i],c=palette[Colour_Ind])

plt.scatter(UMAP_Embedding[Localised_Cells,0],UMAP_Embedding[Localised_Cells,1],s=5,zorder=-2,label="N/A",facecolors="None",edgecolors="grey")
plt.xticks([])
plt.yticks([])
#plt.legend(prop={"size":7})
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(path +"Subset_Celltypes.png",dpi=800)
plt.close()

### The subset of the data forms a clear bifurcation. Use K-means clustering to segment the bifurcation so that we me select 
# the tips of each of the three branches for pseudo-time analysis.

kmeans = KMeans(n_clusters=5, random_state=0).fit(UMAP_Embedding[Localised_Cells,:])
Kmeans_Labels = kmeans.labels_
plt.scatter(UMAP_Embedding[Localised_Cells,0],UMAP_Embedding[Localised_Cells,1],s=8,c=Kmeans_Labels)

Early_ICM_UMAP = UMAP_Embedding[Localised_Cells,:]
Early_ICM_Clusters = Kmeans_Labels
Early_ICM_UMAP = pd.DataFrame(Early_ICM_UMAP,index=Localised_Cells_IDs,columns=np.array(["UMAP 1","UMAP 2"]))
Early_ICM_UMAP.to_csv(path + "Early_ICM_UMAP.csv")

### Save the clusters because we now have to move to an R environment to use the pseudotime software, Slingshot.
np.savetxt(path+'Early_ICM_Clusters.txt', Early_ICM_Clusters, delimiter=" ", fmt="%s") 


######### Swith to an R environment to run Slingshot pseudotime analysis (don't close your python environment if you can avoid it) #########

library(slingshot)
library(Seurat)
#
path <- "/home/ahr35/ES_Paper_Data/ES_Paper_Data/Pre_Implantation_Human_Embryo_Data/Meistermann_Data/"
#
UMAP <- read.csv(paste(path,"Early_ICM_UMAP.csv",sep=""),row.names=1)
Early_ICM_Clusters = read.csv(paste(path,"Early_ICM_Clusters.txt",sep=""),header = FALSE, row.names = NULL)
Early_ICM_Clusters <- as.vector(t(Early_ICM_Clusters))
# Run Slingshot
sds <- slingshot(UMAP, clusterLabels = Early_ICM_Clusters, start.clus = 0, end.clus=c(1,4))
slingshot_pseudotimes = slingPseudotime(sds)
# Save pseudotimes
write.csv(x=slingshot_pseudotimes, file=paste(path,"slingshot_pseudotimes.csv",sep=""))

######### Swith back to the python environment to use the calculated pseudotimes #########
#(if you closed your python environment, you'll need to re-load the data at the start of the workflow)

### Load Slingshot pseudotimes
slingshot_pseudotimes = pd.read_csv(path + "slingshot_pseudotimes.csv",header=0,index_col=0)

### Overlay the hypoblast pseudotime onto the birfucating UMAP
fig, ax = plt.subplots(figsize=(2,2))
plt.title("Hypoblast psueodtime",fontsize=10)
plt.scatter(UMAP_Embedding[Localised_Cells,0],UMAP_Embedding[Localised_Cells,1],c="grey",s=6)
cmap_plot = ax.scatter(UMAP_Embedding[Localised_Cells,0],UMAP_Embedding[Localised_Cells,1],s=6,c=slingshot_pseudotimes["Lineage2"])
divider = make_axes_locatable(ax)
cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.8,0.35,0.05,0.6)))
plt.xticks([])
plt.yticks([])
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(path+"_Pseudotime_Branch.png",dpi=800)
plt.close()

Hypoblast_Pseudotimes = np.asarray(slingshot_pseudotimes["Lineage2"])

### Select markers to analyse and plot
Plot_Markers = np.array(["PDGFRA", "GATA4", "SOX17", "FOXA2","BMP2","COL4A1","CPN1","FLRT3","FRZB","IGF1","LGALS2","OTX2","RSPO3","SALL1","SYT13","VIL1","GATA6","NANOG","POU5F1","SOX2","SOX7"])

### Nearest neighbour smoothed plots ###
# Calclacting a smoothed represenation of the gene expression data by averaging the expression of each cell by it's 30 most similar cells.
# 30 neighbours was selected because this is the number of neighbours that were used by Radley et al. 2022 to generate their UMAP embedding.
k = 30

### Matrix of pairwise correlation distances. 
# The correlaition distance metric is used because this is the distance metric used by
# Radley et al. 2022 to generate their UMAP embedding. Likewise, Chosen_Gene_IDs refers to subsetting the entire gene expression matrix
# down to the 3700 highly structured genes identifed by Radley et al. 2022 for this human pre-implantation embryo data.
distmat = squareform(pdist(Human_Embryo_Counts[Chosen_Gene_IDs], 'correlation'))

### Identify each cells neighbourhood of 30 cells.
neighbors = np.sort(np.argsort(distmat, axis=1)[:, 0:k])

### Calculate the smoothed expression matrix
Smoothed_Expressions = np.zeros((distmat.shape[0],Plot_Markers.shape[0]))

for i in np.arange(Plot_Markers.shape[0]):
    Raw_Expression = np.log2(np.array(Human_Embryo_Counts[Plot_Markers[i]])+1)
    Smoothed_Expression = np.zeros(Raw_Expression.shape[0])
    for j in np.arange(Smoothed_Expression.shape[0]):
        Smoothed_Expression[j] = np.mean(Raw_Expression[neighbors[j,:]])
    #
    Smoothed_Expressions[:,i] = Smoothed_Expression

Smoothed_Expressions = pd.DataFrame(Smoothed_Expressions,columns=Plot_Markers)

### Plot each of the smoothed/fitted expression curves. For each plot, we plot the original expression values in grey,
# the smoothed expression values in blue and the logistic curve fitted to the smoothed expression values with a red line.

for i in np.arange(Plot_Markers.shape[0]):
    Gene = Plot_Markers[i]
    #
    Gene_Expression = np.log2(Human_Embryo_Counts[Gene]+1)[Localised_Cells]
    Smoothed_Gene_Expression = np.asarray(Smoothed_Expressions[Gene])[Localised_Cells]
    Smoothed_Gene_Expression[np.isnan(Hypoblast_Pseudotimes)] = np.nan
    #
    x = Hypoblast_Pseudotimes[np.isnan(Hypoblast_Pseudotimes)==0]
    Smoothen_x = x
    Smoothen_x = Smoothen_x + np.random.normal(0,0.000001,Smoothen_x.shape[0])
    Sort = np.argsort(Smoothen_x)
    Smoothen_x = Smoothen_x[Sort]
    Smoothen_y = Smoothed_Gene_Expression[np.isnan(Hypoblast_Pseudotimes)==0][Sort]
    popt, pcov = curve_fit(logifunc, Smoothen_x, Smoothen_y, p0=[50,np.max(Smoothen_y),0.1,0], maxfev=10000)
    #
    plt.figure(figsize=(2,2))
    plt.title(Gene,fontsize=11)
    plt.scatter(Hypoblast_Pseudotimes,Gene_Expression,s=3,label="Raw expression",c="tab:gray")
    plt.scatter(Hypoblast_Pseudotimes,Smoothed_Gene_Expression,s=4,label="kNN smoothed expression",c="tab:blue")
    plt.plot(Smoothen_x, logifunc(Smoothen_x, *popt),label='Fitted function',c="tab:red")
    #plt.xlabel("Hypoblast pseudotime",fontsize=12)
    #plt.ylabel("$log_2$(Expression)",fontsize=12)
    plt.subplots_adjust(0.02,0.02,0.98,0.9)
    plt.xticks(np.arange(min(x), max(x)+1, 1))
    plt.yticks(np.arange(min(Gene_Expression), max(Gene_Expression)+1, 2))
    plt.tight_layout()
    #plt.legend()
    plt.savefig(path+str(Plot_Markers[i])+"_Pseudotime_Scatter.png",dpi=800)
    plt.close()


### Plot smoothed expression of the selected markers onto the bifurcating UMAP
for i in np.arange(Plot_Markers.shape[0]):
    fig, ax = plt.subplots(figsize=(2,2))
    colour = "k"
    if i <= 15:
        colour = palette[i]
    plt.title(Plot_Markers[i],fontsize=10,c=colour)
    #
    Smoothed_Expression = np.asarray(Smoothed_Expressions[Plot_Markers[i]])[Localised_Cells]
    Layer=np.argsort(Smoothed_Expression)
    cmap_plot = ax.scatter(UMAP_Embedding[Localised_Cells,0][Layer],UMAP_Embedding[Localised_Cells,1][Layer],s=6,c=Smoothed_Expression[Layer],cmap="seismic")
    divider = make_axes_locatable(ax)
    cb = fig.colorbar(cmap_plot,cax=ax.inset_axes((0.8,0.35,0.05,0.6)),format=FormatStrFormatter('%.0f'))
    cb.set_label('$log_2$(Expression)', labelpad=-35,fontsize=7)
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(0.02,0.02,0.98,0.9)
    plt.savefig(path+str(Plot_Markers[i])+"_Subset.png",dpi=800)
    plt.close()


### Identify the markers that we generated validation through immunstaining for.
Staining_Markers = np.array(["PDGFRA","SOX17","GATA4","FOXA2"])

### Halfway intercepts will identify when the middle of logistic curve, which we use as a metric to represent,
# the boundary between a gene being inactive or active.
Halfway_Intercepts = np.zeros(16)

### Plot the normalised logistic curves to visualise the sequential activation of each gene along the hypoblast pseudotime

plt.figure(figsize=(4,4))
plt.title("Sequential Hypoblast Marker Upregulation",fontsize=11)
plt.xlabel("Hypoblast pseudotime",fontsize=10)
plt.ylabel("Normalised lNN smoothed expression",fontsize=10)
for i in np.arange(16):
    Gene = Plot_Markers[i]
    Smoothed_Gene_Expression = np.asarray(Smoothed_Expressions[Plot_Markers[i]])[Localised_Cells]
    #
    x = Hypoblast_Pseudotimes[np.isnan(Hypoblast_Pseudotimes)==0]
    Smoothen_x = x
    Smoothen_x = Smoothen_x + np.random.normal(0,0.000001,Smoothen_x.shape[0])
    Sort = np.argsort(Smoothen_x)
    Smoothen_x = Smoothen_x[Sort]
    Smoothen_y = Smoothed_Gene_Expression[np.isnan(Hypoblast_Pseudotimes)==0][Sort]
    popt, pcov = curve_fit(logifunc, Smoothen_x, Smoothen_y, p0=[50,np.max(Smoothen_y),0.1,0])
    #
    y_values = logifunc(Smoothen_x, *popt)
    y_values = y_values - np.min(y_values)
    y_values = y_values / np.max(y_values)
    #
    Halfway_Intercepts[i] = Smoothen_x[np.argsort(np.absolute(y_values-0.5))[0]]
    #
    LW = 0.8
    Order = -1
    if np.isin(Gene,Staining_Markers) == True:
        LW = 2.3
        Order = 1
    plt.plot(Smoothen_x, y_values,label=Gene,linewidth=LW,zorder=Order)

plt.legend(prop={"size":7})
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.tight_layout()
plt.savefig(path+"All_Sequential_Upregulation.png",dpi=800)
plt.close() 

# Upregulation order
Plot_Markers[np.argsort(Halfway_Intercepts)]

### Manual partitioning of the Radley et al. 2022 UMAP embedding.

x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 0.014, UMAP_Embedding[:,0] < 1.9))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 3.13, UMAP_Embedding[:,1] < 5.506))[0]
ICM_Cells = np.intersect1d(x_range,y_range)
#
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -1.15, UMAP_Embedding[:,0] < 0.650))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 5.501, UMAP_Embedding[:,1] < 6.9))[0]
Epiblast_Cells = np.intersect1d(x_range,y_range)
#
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 0.827, UMAP_Embedding[:,0] < 1.841))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 5.875, UMAP_Embedding[:,1] < 8.1))[0]
Hypoblast_Cells = np.intersect1d(x_range,y_range)
#
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 1.55, UMAP_Embedding[:,0] < 3))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 1.236, UMAP_Embedding[:,1] < 2.762))[0]
Early_ICM_Cells = np.intersect1d(x_range,y_range)
#
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > -5, UMAP_Embedding[:,0] < 3))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > 2, UMAP_Embedding[:,1] < 11.5))[0]
TE_Cells = np.intersect1d(x_range,y_range)
TE_Cells = np.delete(TE_Cells,np.isin(TE_Cells,ICM_Cells))
TE_Cells = np.delete(TE_Cells,np.isin(TE_Cells,Epiblast_Cells))
TE_Cells = np.delete(TE_Cells,np.isin(TE_Cells,Hypoblast_Cells))
TE_Cells = np.delete(TE_Cells,np.isin(TE_Cells,Early_ICM_Cells))
#
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 5.3, UMAP_Embedding[:,0] < 6.9))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > -1, UMAP_Embedding[:,1] < 1.5))[0]
Morula_Cells = np.intersect1d(x_range,y_range)
#
x_range = np.where(np.logical_and(UMAP_Embedding[:,0] > 9.7, UMAP_Embedding[:,0] < 11.7))[0]
y_range = np.where(np.logical_and(UMAP_Embedding[:,1] > -1, UMAP_Embedding[:,1] < 1.5))[0]
Eight_Cells = np.intersect1d(x_range,y_range)

### Plot the new cell labels
plt.figure(figsize=(4,4))
plt.scatter(UMAP_Embedding[ICM_Cells,0],UMAP_Embedding[ICM_Cells,1],s=6,label="ICM group",c="tab:blue")
plt.scatter(UMAP_Embedding[Epiblast_Cells,0],UMAP_Embedding[Epiblast_Cells,1],s=6,label="Epiblast group",c="tab:orange")
plt.scatter(UMAP_Embedding[Hypoblast_Cells,0],UMAP_Embedding[Hypoblast_Cells,1],s=6,label="Hypoblast group",c="tab:green")
plt.scatter(UMAP_Embedding[Early_ICM_Cells,0],UMAP_Embedding[Early_ICM_Cells,1],s=6,label="Early ICM group",c="tab:red")
plt.scatter(UMAP_Embedding[TE_Cells,0],UMAP_Embedding[TE_Cells,1],s=6,label="Trophectoderm group",c="tab:purple")
plt.scatter(UMAP_Embedding[Morula_Cells,0],UMAP_Embedding[Morula_Cells,1],s=6,label="Morula group",c="navy")
plt.scatter(UMAP_Embedding[Eight_Cells,0],UMAP_Embedding[Eight_Cells,1],s=6,label="8-Cell group",c="darkkhaki")
plt.scatter(UMAP_Embedding[:,0],UMAP_Embedding[:,1],s=3,zorder=-2,label="N/A",facecolors="None",edgecolors="grey")
plt.title("Manual Cell Annotations", fontsize=14)
plt.xticks([])
plt.yticks([])
plt.legend(prop={"size":8})
plt.subplots_adjust(0.02,0.02,0.98,0.9)
plt.savefig(path+"_Manual_Cluster_Annotations.png",dpi=800)
plt.close()

### Name each label for plotting
Manual_Cell_Labels = np.repeat("    N/A    ",Smoothed_Expressions.shape[0])
Manual_Cell_Labels[ICM_Cells] = "ICM"
Manual_Cell_Labels[Epiblast_Cells] = "Epiblast"
Manual_Cell_Labels[Hypoblast_Cells] = "Hypoblast"
Manual_Cell_Labels[Early_ICM_Cells] = "Early ICM"
Manual_Cell_Labels[TE_Cells] = "TE"
Manual_Cell_Labels[Morula_Cells] = "Morula"
Manual_Cell_Labels[Eight_Cells] = "8-Cell"

### Add some additional markers that are not Hypoblast specific to act as controls.
Plot_Markers_Plus = np.append(Plot_Markers,np.array(["NODAL","TDGF1","SLC7A2","GATA3","FGF4"]))

### Re-calculate smoothed expression matrix
Smoothed_Expressions = np.zeros((distmat.shape[0],Plot_Markers_Plus.shape[0]))

for i in np.arange(Plot_Markers_Plus.shape[0]):
    Raw_Expression = np.log2(np.array(Human_Embryo_Counts[Plot_Markers_Plus[i]])+1)
    Smoothed_Expression = np.zeros(Raw_Expression.shape[0])
    for j in np.arange(Smoothed_Expression.shape[0]):
        Smoothed_Expression[j] = np.mean(Raw_Expression[neighbors[j,:]])
    #
    Smoothed_Expressions[:,i] = Smoothed_Expression

Smoothed_Expressions = pd.DataFrame(Smoothed_Expressions,columns=Plot_Markers_Plus)
Smoothed_Expressions["Manual_Labels"] = Manual_Cell_Labels
Inds = np.where(Smoothed_Expressions["Manual_Labels"] != "    N/A    ")[0]
#
my_pal = {"8-Cell": "darkkhaki", "Morula": "navy", "TE": "tab:purple", "Early ICM": "tab:red", "Hypoblast": "tab:green", "Epiblast": "tab:orange", "ICM": "tab:blue"}
Cell_Type_Order = ["8-Cell", "Morula", "Early ICM", "ICM", "Hypoblast", "Epiblast", "TE"]

### Plot violin plots of expression for each gene across each cell type

for i in np.arange(Plot_Markers_Plus.shape[0]):
    Gene = Plot_Markers_Plus[i]
    #
    fig, ax = plt.subplots(figsize=(2,3))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    sns.violinplot(data=Smoothed_Expressions, x="Manual_Labels", y=Gene,order=Cell_Type_Order,scale='width',palette=my_pal)
    plt.title(Gene,fontsize=12)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(rotation=90)
    plt.subplots_adjust(0.02,0.02,0.98,0.9)
    plt.tight_layout()
    plt.savefig(path+str(Plot_Markers_Plus[i])+"_Violin.png",dpi=800)
    plt.close()



### END ###

