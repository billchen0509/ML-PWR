# Load mrbin library
library(mrbin)


# Load the root data directory
data_dir <- "/Users/billhikari/Library/CloudStorage/OneDrive-McGillUniversity/Matthias Klein's files - GWG/"

# Read the samples 
samples <- list.dirs(data_dir, recursive = FALSE, full.names = TRUE)
samples <- samples[grepl("^GWG",basename(samples))]

# Remove GWG-4137-V4-lost/GWG-4137-V5-lost
exclude_names <- c("GWG-4137-V4-lost", "GWG-4137-V5-lost")
samples <- samples[!basename(samples) %in% exclude_names]
# join path
nmr_path <- file.path(samples,"2","pdata","1")

#use mrbin
results<-mrbin(silent=FALSE,parameters=list(binwidth2D=0.04,binheight=1,
                                            reference2D=c(0.04,-0.04,-2,2),signal_to_noise2D=4,cropHSQC="Yes",
                                            noiseRange2d=c(3.3,2.3,90,110),croptopRight=c(0,-1.5),croptopLeft=c(0,3.5),
                                            cropbottomRight=c(160,6),cropbottomLeft=c(160,10),dimension="2D",
                                            binMethod="Rectangular bins",binRegion=c(9.5,0.5,10,156),
                                            referenceScaling="Yes",removeSolvent="Yes",solventRegion=c(4.95,4.65),
                                            removeAreas="Yes",
                                            removeAreaList=matrix(c(
                                              3.825,3.51,63,67.5,3.825,3.536,73.5,76.5,3.68,3.6,65,66.5,
                                              1.341,1.29,21.9,23.1,3.53,3.46,78,79,4.14,4.1,71,72,
                                              3.5,3.46,78,79
                                            ),ncol=4,byrow=TRUE),
                                            sumBins="No",noiseRemoval="Yes",noiseThreshold=0.5,dilutionCorrection="No",
                                            PQNScaling="No",fixNegatives="No",logTrafo="No",unitVarianceScaling="No",
                                            PQNminimumFeatures=40,PQNIgnoreSugarArea="Yes",PQNsugarArea=c(5.4,3.35,72,100),
                                            saveFiles="Yes",useAsNames="Folder names",
                                            outputFileName="/Users/billhikari/OneDrive - McGill University/mrbin_result/mrbin_2025-08-12_17-33-59.044437",
                                            PCAtitlelength=8,PCA="Yes",NMRvendor="Bruker",
                                            NMRfolders=c(nmr_path)))


# Load the bined datset
#load('~/OneDrive - McGill University/mrbin_result/Result_final/updated/mrbin_2025-08-12_15-56-05.945403.Rdata')
metaboliteIdentities <- read.csv("~/OneDrive - McGill University/Matthias Klein's files - Hybrid/Blood/BloodMatrixNew3.csv")
rownames(metaboliteIdentities) <- make.unique(metaboliteIdentities$metabolite)

annotations <- annotatemrbin(results, metaboliteIdentities=metaboliteIdentities)
annotated <- annotations$metadata$annotations

#write.csv(data.frame(item = unlist(annotated)), "~/OneDrive - McGill University/annotated.csv", row.names = FALSE)

annotated_1d <- results2$metadata$annotations
#write.csv(data.frame(item = unlist(annotated_1d)), "~/OneDrive - McGill University/peaks/annotated_1d.csv", row.names = FALSE)

annotated_2d <- results2$metadata$annotations
#write.csv(data.frame(item = unlist(annotated_2d)), "~/OneDrive - McGill University/peaks/annotated_2d.csv", row.names = FALSE)
