# this script serves to validate the automated columns (columns 15 to 37) in the file GELATO-population-glottocode-mapping.csv
# note that columns 1 to 14 are asscociated with GeLaTo (columns 1 to 8) or manually entered (columns 9 to 14)

rm(list=ls())

library(tidyverse)
library(testthat)
library(densify)

# NA conversion function
na_convert <- function(data){
  data[data == "?"] <- NA
  data[data == "NA"] <- NA
  data[data == ""] <- NA
  return(data)
}

# read in glottolog taxonomy
taxonomy <- as_flat_taxonomy_matrix(glottolog_languoids)
names(taxonomy)[1]<-"glottocode"

glottolog_level <- glottolog_languoids

# however note that in gelato we still have a location column...
gelato_mapping_decisions <- read.csv("input/GELATO-population-glottocode-mapping.csv")
gelato_mapping_decisions[gelato_mapping_decisions == "NA"] <- NA
gelato_mapping_decisions[gelato_mapping_decisions == ""] <- NA

### check all densities are correct
gelato <- gelato_mapping_decisions %>% select(1:14)

# retrieve proxies
proxies <- gelato[,c(11:14)]

base_status <- apply(as.data.frame(gelato),1,function(x)if(x[7]%in%glottolog_level$glottocode==F){NA}else{filter(glottolog_level,glottocode==x[7])[1,4]})
upstream_glottocode <- apply(as.data.frame(gelato),1,function(x)if(x[7]%in%taxonomy$glottocode==F){NA}else{filter(taxonomy,glottocode==x[7])[1,sum(!is.na(filter(taxonomy,glottocode==x[7])))-1]})
upstream_glottocode_status <- apply(as.data.frame(upstream_glottocode),1,function(x)if(x%in%glottolog_level$glottocode==F){NA}else{filter(glottolog_level,glottocode==x[1])[1,4]})

gelato <- cbind(gelato,data.frame(status.glottolog = base_status,
                                  glottocodeUpstream = upstream_glottocode,
                                  glottocodeUpstream.status = upstream_glottocode_status))

# read in GBI data: logical and statistical, full 
gbilogfull <- read.csv("input/GBI/logicalGBI.csv", row.names = "glottocode") %>% select(-X) %>% na_convert()
gbistafull <- read.csv("input/GBI/statisticalGBI.csv", row.names = "glottocode") %>% select(-X) %>% na_convert()

# read in TLI data: logical and statistical, full
tlilogfull <- read.csv("input/TLI/logicalTLI_full.csv", row.names = "glottocode") %>% select(-X) %>% na_convert()
tlistafull <- read.csv("input/TLI/statisticalTLI_full.csv", row.names = "glottocode") %>% select(-X) %>% na_convert()

# which languages are in any of the GBI and/or TLI curations?
glottocodesInAnyGBI <- unique(c(rownames(gbilogfull),rownames(gbistafull)))
glottocodesInAnyTLI <- unique(c(rownames(tlilogfull),rownames(tlistafull)))

# variable counts for each dataset
gbilogfull.count <- apply(as.data.frame(glottocodesInAnyGBI),1,function(x)if(x%in%rownames(gbilogfull)==F){NA}else{length(na.omit(t(gbilogfull[which(rownames(gbilogfull)==x),])))})
gbistafull.count <- apply(as.data.frame(glottocodesInAnyGBI),1,function(x)if(x%in%rownames(gbistafull)==F){NA}else{length(na.omit(t(gbistafull[which(rownames(gbistafull)==x),])))})

tlilogfull.count <- apply(as.data.frame(glottocodesInAnyTLI),1,function(x)if(x%in%rownames(tlilogfull)==F){NA}else{length(na.omit(t(tlilogfull[which(rownames(tlilogfull)==x),])))})
tlistafull.count <- apply(as.data.frame(glottocodesInAnyTLI),1,function(x)if(x%in%rownames(tlistafull)==F){NA}else{length(na.omit(t(tlistafull[which(rownames(tlistafull)==x),])))})

# densities for all datasets
gbilogfull.density <- apply(as.data.frame(glottocodesInAnyGBI),1,function(x)if(x%in%rownames(gbilogfull)==F){NA}else{length(na.omit(t(gbilogfull[which(rownames(gbilogfull)==x),])))/ncol(gbilogfull)})
gbistafull.density <- apply(as.data.frame(glottocodesInAnyGBI),1,function(x)if(x%in%rownames(gbistafull)==F){NA}else{length(na.omit(t(gbistafull[which(rownames(gbistafull)==x),])))/ncol(gbistafull)})

tlilogfull.density <- apply(as.data.frame(glottocodesInAnyTLI),1,function(x)if(x%in%rownames(tlilogfull)==F){NA}else{length(na.omit(t(tlilogfull[which(rownames(tlilogfull)==x),])))/ncol(tlilogfull)})
tlistafull.density <- apply(as.data.frame(glottocodesInAnyTLI),1,function(x)if(x%in%rownames(tlistafull)==F){NA}else{length(na.omit(t(tlistafull[which(rownames(tlistafull)==x),])))/ncol(tlistafull)})

# subset taxonomy to lgs that are in GeLaTo or GBI/TLI (each curation)
taxonomy_for_densities <- taxonomy
taxonomy_for_densities$GelatoBase <- apply(taxonomy,1,function(x)x[1]%in%gelato$glottocodeBase)
taxonomy_for_densities$GelatoUpstream <- apply(taxonomy,1,function(x)x[1]%in%gelato$glottocodeUpstream)
taxonomy_for_densities$InGBI <- apply(taxonomy,1,function(x)x[1]%in%glottocodesInAnyGBI)
taxonomy_for_densities$InTLI <- apply(taxonomy,1,function(x)x[1]%in%glottocodesInAnyTLI)
taxonomy_for_densities <- taxonomy_for_densities %>% filter(GelatoBase == T | GelatoUpstream == T | InGBI == T | InTLI == T)

lgs_in_any_GBI <- data.frame(glottocode = glottocodesInAnyGBI,
                            features.count.gbi.logical.full = gbilogfull.count,
                            features.count.gbi.statistical.full = gbistafull.count,
                            density.gbi.logical.full= gbilogfull.density,
                            density.gbi.statistical.full= gbistafull.density)

lgs_in_any_TLI <- data.frame(glottocode = glottocodesInAnyTLI,
                            features.count.tli.logical.full = tlilogfull.count,
                            features.count.tli.statistical.full = tlistafull.count,
                            density.tli.logical.full= tlilogfull.density,
                            density.tli.statistical.full= tlistafull.density)
                            
taxonomy_for_densities <- left_join(taxonomy_for_densities,lgs_in_any_GBI)
taxonomy_for_densities <- left_join(taxonomy_for_densities,lgs_in_any_TLI)

## helper columns: counts and densities for all datasets for glottocodeBase
gelato <- left_join(gelato,select(filter(taxonomy_for_densities,GelatoBase == T),c(glottocode,
                                                                            features.count.gbi.logical.full,
                                                                            features.count.gbi.statistical.full,
                                                                            features.count.tli.logical.full,
                                                                            features.count.tli.statistical.full,
                                                                            density.gbi.logical.full,
                                                                            density.gbi.statistical.full,
                                                                            density.tli.logical.full,
                                                                            density.tli.statistical.full)), by=c("glottocodeBase"="glottocode"))


## helper columns: counts and densities for all datasets for glottocodeUpstream
gelato <- left_join(gelato,select(filter(taxonomy_for_densities,GelatoUpstream == T),c(glottocode,
                                                                                       features.count.gbi.logical.full,
                                                                                       features.count.gbi.statistical.full,
                                                                                       features.count.tli.logical.full,
                                                                                       features.count.tli.statistical.full,
                                                                                       density.gbi.logical.full,
                                                                                       density.gbi.statistical.full,
                                                                                       density.tli.logical.full,
                                                                                       density.tli.statistical.full)), by=c("glottocodeUpstream"="glottocode"))


names(gelato)[18:33]<-c("base.features.count.gbi.logical.full","base.features.count.gbi.statistical.full","base.features.count.tli.logical.full","base.features.count.tli.statistical.full",
                       "base.density.gbi.logical.full","base.density.gbi.statistical.full","base.density.tli.logical.full","base.density.tli.statistical.full",
                       "upstream.features.count.gbi.logical.full","upstream.features.count.gbi.statistical.full","upstream.features.count.tli.logical.full","upstream.features.count.tli.statistical.full",
                       "upstream.density.gbi.logical.full","upstream.density.gbi.statistical.full","upstream.density.tli.logical.full","upstream.density.tli.statistical.full")


proxy.gbi.logical.full.density <- unlist(apply(as.data.frame(proxies[,1]),1,function(x)if(is.na(x)){NA}else{taxonomy_for_densities[which(taxonomy_for_densities$glottocode==x),"density.gbi.logical.full"]}))
proxy.gbi.statistical.full.density <- unlist(apply(as.data.frame(proxies[,2]),1,function(x)if(is.na(x)){NA}else{taxonomy_for_densities[which(taxonomy_for_densities$glottocode==x),"density.gbi.statistical.full"]}))
proxy.tli.logical.full.density <- unlist(apply(as.data.frame(proxies[,3]),1,function(x)if(is.na(x)){NA}else{taxonomy_for_densities[which(taxonomy_for_densities$glottocode==x),"density.tli.logical.full"]}))
proxy.tli.statistical.full.density <- unlist(apply(as.data.frame(proxies[,4]),1,function(x)if(is.na(x)){NA}else{taxonomy_for_densities[which(taxonomy_for_densities$glottocode==x),"density.tli.statistical.full"]}))

# add columns specifying coverage for all proxies
gelato <- cbind(gelato,proxy.gbi.logical.full.density, proxy.gbi.statistical.full.density,
                proxy.tli.logical.full.density, proxy.tli.statistical.full.density)


gelato_automated_glottologStatus <- gelato[,15:17]
gelato_automated_numeric <- gelato[,18:ncol(gelato)]

# now check everything is correct:
expect_true(all(gelato_automated_glottologStatus==gelato_mapping_decisions[15:17], na.rm = T))
expect_true(all(round(gelato_automated_numeric,digits=5)==round(gelato_mapping_decisions[18:ncol(gelato_mapping_decisions)],digits=5), na.rm = T))
