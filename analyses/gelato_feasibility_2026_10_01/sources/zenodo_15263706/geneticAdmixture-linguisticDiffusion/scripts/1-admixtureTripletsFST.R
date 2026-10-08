rm(list=ls())

library(reshape)
library(tidyr)
library(tidyverse)
library(janitor)
library(readxl)
library(testthat)

# specify the minimum and maximum number of K for the analysis
minK <- 12
maxK <- 30
K_range <- c(minK:maxK)

infoID <- read.csv("input/MegaAdmixtureCatalogue/ADMIXTURE/GeneticInfoID.csv", header=T, as.is=T , comment.char = "", fill=T) %>% select(-X)
names(infoID)[1] <- "PopName"

# make a dataframe containing information on all populations, including number of individuals per population (sample size)
pops <- table(infoID$PopName) # sample size
infopop <- infoID[!duplicated(infoID$PopName),] 
infopop$samplesize <- apply(infopop,1,function(x)pops[which(names(pops)==x[1])])

proxies <- read.csv("input/GeLaTo-population-glottocode-mapping.csv") %>%
  select(c("PopName","glottocodeBase","glottologFamily","gbi.logical.full.proxy","gbi.statistical.full.proxy","tli.logical.full.proxy","tli.statistical.full.proxy",
           "proxy.gbi.logical.full.density","proxy.gbi.statistical.full.density","proxy.tli.logical.full.density","proxy.tli.statistical.full.density"))
infopop <- left_join(select(infopop,-"glottocodeBase"),proxies, by="PopName")

infopop$gbi.logical.full.proxy[which(infopop$gbi.logical.full.proxy=="NA")]<-NA
infopop$gbi.statistical.full.proxy[which(infopop$gbi.statistical.full.proxy=="NA")]<-NA
infopop$tli.logical.full.proxy[which(infopop$tli.logical.full.proxy=="NA")]<-NA
infopop$tli.statistical.full.proxy[which(infopop$tli.statistical.full.proxy=="NA")]<-NA

infopop$proxy.gbi.logical.full.density[which(is.na(infopop$proxy.gbi.logical.full.density))]<-0
infopop$proxy.gbi.statistical.full.density[which(is.na(infopop$proxy.gbi.statistical.full.density))]<-0
infopop$proxy.tli.logical.full.density[which(is.na(infopop$proxy.tli.logical.full.density))]<-0
infopop$proxy.tli.statistical.full.density[which(is.na(infopop$proxy.tli.statistical.full.density))]<-0

# are logical and statistical distinct? we don't expect so --> confirmed
expect_true(all(infopop$gbi.logical.full.proxy==infopop$gbi.statistical.full.proxy, na.rm = T))
expect_true(all(infopop$tli.logical.full.proxy==infopop$tli.statistical.full.proxy, na.rm = T))

table(infopop$gbi.logical.full.proxy==infopop$glottocodeBase)
table(infopop$tli.logical.full.proxy==infopop$glottocodeBase)

infopop[infopop==""]<-NA
infopop[infopop=="ND"]<-NA

infopop$GBIstatus<-apply(infopop,1,function(x) if(is.na(x[11])){"no proxy"} else if(x[8]==x[11]){"glottocodeBase"} else{"glottocodeGBIproxy"})
infopop$TLIstatus<-apply(infopop,1,function(x) if(is.na(x[13])){"no proxy"} else if(x[8]==x[13]){"glottocodeBase"} else{"glottocodeTLIproxy"})

rownames(infopop)<-infopop$PopName

## table S1

stable1 <- select(infopop, c(PopName, samplesize, glottologFamily, glottocodeBase, gbi.statistical.full.proxy, tli.statistical.full.proxy, Publication))
rownames(stable1) <- NULL
names(stable1) <- c("Population", "N_Individuals", "Language_Family", "GeLaTo_Glottocode", "GBI_Glottocode", "TLI_Glottocode", "Reference")

stable1[stable1 == "Jeong_2019"] <- "Jeong, C., et al. (2019). The genetic history of admixture across inner Eurasia. Nature Ecology & Evolution, 3(6), 966-976."
stable1[stable1 == "LazaridisNature2014"] <- "Lazaridis, I., et al. (2014). Ancient human genomes suggest three ancestral populations for present-day Europeans. Nature, 513(7518), 409-413."
stable1[stable1 == "PattersonGenetics2012"] <- "Patterson, N., et al. (2012). Ancient admixture in human history. Genetics, 192(3), 1065-1093."
stable1[stable1 == "SkoglundNature2015"] <- "Skoglund, P., et al. (2015). Genetic evidence for two founding populations of the Americas. Nature, 525(7567), 104-108."
stable1[stable1 == "SkoglundNature2016"] <- "Skoglund, P., et al. (2016). Genomic insights into the peopling of the Southwest Pacific. Nature, 538(7626), 510-513."
stable1[stable1 == "LazaridisNature2016"] <- "Lazaridis, I., et al. (2016). Genomic insights into the origin of farming in the ancient Near East. Nature, 536(7617), 419-424."
stable1[stable1 == "QinStonekingMBE2015"] <- "Qin, P., & Stoneking, M. (2015). Denisovan ancestry in East Eurasian and native American populations. Molecular Biology and Evolution, 32(10), 2665-2674."
stable1[stable1 == "Lipson_2020"] <- "Lipson, M., et al. (2020). Ancient West African foragers in the context of African population history. Nature, 577(7792), 665-670."
stable1[stable1 == "Barbieri_2018"] <- "Barbieri, C., et al. (2019). The current genomic landscape of western south America: Andes, amazonia, and Pacific coast. Molecular Biology and Evolution, 36(12), 2698-2713."
stable1[stable1 == "PickrellNatureCommunications2012"] <- "Pickrell, J. K., et al. (2012). The genetic prehistory of southern Africa. Nature communications, 3(1), 1143."
stable1[stable1 == "Flegontov_2019"] <- "Flegontov, P., et al. (2019). Palaeo-Eskimo genetic ancestry and the peopling of Chukotka and North America. Nature, 570(7760), 236-240."
stable1[stable1 == "VyasAJPA2017"] <- "Vyas, D. N., et al. (2017). Testing support for the northern and southern dispersal routes out of Africa: an analysis of Levantine and southern Arabian populations. American Journal of Physical Anthropology, 164(4), 736-749."
stable1[stable1 == "Broushaki2016"] <- "Broushaki, F., et al. (2016). Early Neolithic genomes from the eastern Fertile Crescent. Science, 353(6298), 499-503."
stable1[stable1 == "Barbieri_2018&LazaridisNature2014"] <- "Barbieri, C., et al. (2019). The current genomic landscape of western south America: Andes, amazonia, and Pacific coast. Molecular Biology and Evolution, 36(12), 2698-2713. & Lazaridis, I., et al. (2014). Ancient human genomes suggest three ancestral populations for present-day Europeans. Nature, 513(7518), 409-413."
stable1[stable1 == "SkoglundCell2017"] <- "Skoglund, P., et al. (2017). Reconstructing prehistoric African population structure. Cell, 171(1), 59-71."
stable1[stable1 == "Lipson_2018"] <- "Lipson, M., et al. (2018). Population turnover in remote Oceania shortly after initial settlement. Current Biology, 28(7), 1157-1165."

write.csv(stable1, "tables/tableS1.csv", row.names = F)

# filter out populations exhibiting a minimum threshold of their 2 most represented components at 70%, 80% and 90%
# the second-strongest component must be present at at least 5%, such that we also consider cases of admixture with minor contributions from the second source, but simultaneously ensure that such contributions are large enough that they do not represent a background noise effect
thresholdMIN_1 <- 0.9
thresholdMIN_2 <- 0.8
thresholdMIN_3 <- 0.7
threshold_pop2 <- 0.05

# system("mkdir threshold90percent")
# system("mkdir threshold80percent")
# system("mkdir threshold70percent")

thresholdSource<- 0.8 # a potential source pop has to have the target component at least at this frequency (for the FST closest source proxy)

# load FST data
fstData <- read.csv("input/FST.csv") %>% select(-X)
fstData <- rbind(c("Abazin","Abazin",NA),fstData,c("Zoro","Zoro",NA)) # to enable pivot_wider to be 558x558
fstData <- pivot_wider(fstData, names_from = Pop1, values_from = FST)
fstMatrix<-as.matrix(select(fstData,-Pop2))
fstMatrix[fstMatrix < 0] <- 0 # set all negative FST-values to zero (these are artifacts)
fstMatrix<-apply(fstMatrix,1,as.numeric)
fstMatrix[lower.tri(fstMatrix)] <- t(fstMatrix)[lower.tri(fstMatrix)]
colnames(fstMatrix)<-fstData$Pop2
rownames(fstMatrix)<-fstData$Pop2

fstMatrix<-fstMatrix[infopop$PopName,infopop$PopName] # reorder the rows and columns in fstMatrix to match infopop
expect_true(all(colnames(fstMatrix)==infopop$PopName)) # sanity check

# this is a function to find admixture candidates as well as candidate populations representing possible source populations for different admixture components
FindAdmixtureCandidates <- function(thresholdMIN){
  admixPopsREDtot_best<-NA # here we find the **best** candidates for source populations via FST and list all other possibilities according to our criteria in descending order 
  admixPopsREDtot_long<-NA # here we create a long data frame listing each possibility as an own row (makes finding reoccuring triplets easier downstream)
  for (i in K_range){ # for all K-values considered
    ADM<-read.table(paste0("input/MegaAdmixtureCatalogue/ADMIXTURE/best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K", i,".Q"))
    megaADM<-cbind(infoID, ADM)  # add info to the admixture percentages for each individual
    freqK<-aggregate(megaADM[,(ncol(infoID)+1):(ncol(megaADM))], list(megaADM$PopName), mean) # collect average admixture percentages per population for K
    rownames(freqK)<-freqK$Group.1
    admixPopsK<-freqK[infopop$PopName,-1] # reorder the rows in freqK to match infopop
    
    ## sanity check
    expect_true(all(rownames(admixPopsK)==infopop$PopName))
    
    admixPops_best<-cbind(infopop,admixPopsK)
    
    ## sanity check
    expect_true(all(rownames(admixPopsK)==infopop$PopName))
    expect_true(all(rownames(admixPopsK)==rownames(admixPops_best)))
    expect_true(all(rownames(admixPopsK)==rownames(fstMatrix)))
    
    admixPops_best$K_level<-i
    
    admixPops_best$FIRSTcomponent<-apply(admixPopsK,1,function(x) sort(x,decreasing=T))[1,]
    admixPops_best$FIRSTname<-(apply(admixPopsK,1,function(x) names(x)[which(x==sort(x,decreasing=T)[1])]))
    admixPops_best$SECONDcomponent<-apply(admixPopsK,1,function(x) sort(x,decreasing=T))[2,]
    admixPops_best$SECONDname<-(apply(admixPopsK,1,function(x) names(x)[which(x==sort(x,decreasing=T)[2])][1])) # a few pops have near 0 for several components (often all but the FIRSTcomponent); the [1] at the end here just selects the first of the other components to ensure the analysis works (does not interfere with the analysis; these are non-admixed populations)
    admixPops_best$CANDIDATE<-NA
    
    admixPops_best$FIRSTpopWithFSTwarning<-NA
    admixPops_best$FIRSTpopWithFSTbest<-NA
    admixPops_best$FIRSTpopWithFSTother<-NA
    admixPops_best$SECONDpopWithFSTwarning<-NA
    admixPops_best$SECONDpopWithFSTbest<-NA
    admixPops_best$SECONDpopWithFSTother<-NA
    
    admixPops_long <- slice(data.frame(PopName=NA,
                                 K_level=NA,
                                 FIRSTpopFST=NA,
                                 SECONDpopFST=NA),0)
    
    for (k in 1:nrow(admixPops_best)){ # populationwise: 
      if (admixPops_best$FIRSTcomponent[k]+admixPops_best$SECONDcomponent[k]>thresholdMIN&admixPops_best$SECONDcomponent[k]>threshold_pop2){ # if the population is admixed according to the defined thresholds:
        admixPops_best$CANDIDATE[k]<-"admixed" #  record it as "admixed"

        #if the population is admixed, assign candidate populations to admixture components based on FST-values:
        source_pop_1_candidates<-rownames(admixPops_best[(admixPops_best[,admixPops_best$FIRSTname[k]]>thresholdSource),]) # identify populations that are candidates to represent the source population for the first ancestry component found in the admixed population, according to the defined threshold (80%)
        source_pop_2_candidates<-rownames(admixPops_best[(admixPops_best[,admixPops_best$SECONDname[k]]>thresholdSource),]) # identify populations that are candidates to represent the source population for the second ancestry component found in the admixed population, according to the defined threshold (80%)
        
        if(length(source_pop_1_candidates)!=0 & length(source_pop_2_candidates)!=0){
          permutations<-length(source_pop_1_candidates)*length(source_pop_2_candidates)
          new_grid<-(cbind(rep(admixPops_best$PopName[k],permutations), # PopName
                           rep(i,permutations), # K_level
                           expand.grid(source_pop_1_candidates,source_pop_2_candidates)))          
          names(new_grid)<-c("PopName","K_level","FIRSTpopFST","SECONDpopFST")
          admixPops_long<-rbind(admixPops_long,new_grid)
        }
        else if(length(source_pop_1_candidates)|length(source_pop_2_candidates)==0 &length(source_pop_1_candidates)+length(source_pop_2_candidates)>0){
          permutations<-max(length(source_pop_1_candidates),length(source_pop_2_candidates))
          nonzero_source<-which(c(length(source_pop_1_candidates),length(source_pop_2_candidates))==permutations)
          new_grid<-data.frame(PopName=rep(admixPops_best$PopName[k],permutations), # PopName
                               K_level=rep(i,permutations), # K_level
                               FIRSTpopFST=rep(NA,permutations),
                               SECONDpopFST=rep(NA,permutations))
          new_grid[,2+nonzero_source]<-c(source_pop_1_candidates,source_pop_2_candidates)
          names(new_grid)<-c("PopName","K_level","FIRSTpopFST","SECONDpopFST")
          admixPops_long<-rbind(admixPops_long,new_grid)
        } 
        
        # component 1
        if(length(source_pop_1_candidates)==0){  # when there is no candidate to represent source population 1, specify so
          admixPops_best$FIRSTpopWithFSTbest[k]<-"none"
          admixPops_best$FIRSTpopWithFSTwarning[k]<-"no candidate"
          admixPops_best$FIRSTpopWithFSTother[k]<-"none"
        } else if(length(source_pop_1_candidates)==1){  # when there is one candidate to represent source population 1, this source might be good or might be drift; to be determined later!
          admixPops_best$FIRSTpopWithFSTbest[k]<-source_pop_1_candidates
          admixPops_best$FIRSTpopWithFSTwarning[k]<-"unique candidate"
          admixPops_best$FIRSTpopWithFSTother[k]<-"none"
        } else{ # when there are several candidates to represent source population 1, list the best in the corresponding column, but keep track of the others too
          ordered_candidates_1 <- names(na.omit(sort(fstMatrix[rownames(admixPops_best)[k],source_pop_1_candidates])))
          admixPops_best$FIRSTpopWithFSTbest[k]<-ordered_candidates_1[1]
          admixPops_best$FIRSTpopWithFSTwarning[k]<-"several candidates"
          admixPops_best$FIRSTpopWithFSTother[k]<-paste(ordered_candidates_1[2:length(ordered_candidates_1)],collapse=", ")
        }
        
        # component 2
        if(length(source_pop_2_candidates)==0){  # when there is no candidate to represent source population 1, specify so
          admixPops_best$SECONDpopWithFSTbest[k]<-"none"
          admixPops_best$SECONDpopWithFSTwarning[k]<-"no candidate"
          admixPops_best$SECONDpopWithFSTother[k]<-"none"
        } else if(length(source_pop_2_candidates)==1){  # when there is one candidate to represent source population 1, this source might be good or might be drift; to be determined later!
          admixPops_best$SECONDpopWithFSTbest[k]<-source_pop_2_candidates
          admixPops_best$SECONDpopWithFSTwarning[k]<-"unique candidate"
          admixPops_best$SECONDpopWithFSTother[k]<-"none"
        } else{ # when there are several candidates to represent source population 1, list the best in the corresponding column, but keep track of the others too
          ordered_candidates_2 <- names(na.omit(sort(fstMatrix[rownames(admixPops_best)[k],source_pop_2_candidates])))
          admixPops_best$SECONDpopWithFSTbest[k]<-ordered_candidates_2[1]
          admixPops_best$SECONDpopWithFSTwarning[k]<-"several candidates"
          admixPops_best$SECONDpopWithFSTother[k]<-paste(ordered_candidates_2[2:length(ordered_candidates_2)],collapse=", ")
        }
  
      } else if (admixPops_best$FIRSTcomponent[k]>0.97) { # out of interest, also capture populations that are made of only one component are not admixed and representative of component source
        admixPops_best$CANDIDATE[k]<-admixPops_best$FIRSTname[k]
      }
    }
      
    # add a further level to find the best population source for each component: take note of which population(s) maximise each component
    admixPops_best$PopulationMaximizingComponent<-NA
    for (j in 1:i){
      admixPops_best$PopulationMaximizingComponent[which(admixPopsK[,j]==max(admixPopsK[,j]))]<-colnames(admixPopsK)[j]
    }
    
    # exclude populations that are not potentially admixed or a potential source for a component; exclude all columns listing the component percentages so we can bind the output with other K values within this loop
    admixPopsRED<-admixPops_best[-which(is.na(admixPops_best$CANDIDATE)&is.na(admixPops_best$PopulationMaximizingComponent)),]
    admixPopsRED<-admixPopsRED[,-c(which(colnames(admixPopsRED)%in%colnames(admixPopsK)))]
    
    # as a last proxy to identify the populations representing the source populations for each admixed population, specify which population(s) maximise the main 2 components 
    PopMaxComp<-admixPops_best%>%select(PopName,PopulationMaximizingComponent)%>%na.omit()
    MatchingTable<-data.frame(Component=unique(PopMaxComp$PopulationMaximizingComponent),Pops=NA)
    MatchingTable$Pops<-apply(MatchingTable,1,function(x) paste(filter(PopMaxComp,PopulationMaximizingComponent==x[1])$PopName,collapse=" | "))
    admixPopsRED$FIRSTpopViaComponent<-MatchingTable$Pops[match(admixPopsRED$FIRSTname,MatchingTable$Component)]
    admixPopsRED$SECONDpopViaComponent<-MatchingTable$Pops[match(admixPopsRED$SECONDname,MatchingTable$Component)]
    admixPopsREDtot_best<-rbind(admixPopsREDtot_best,admixPopsRED)
    admixPopsREDtot_long<-rbind(admixPopsREDtot_long,admixPops_long)
  }
  
  admixPopsREDtot_best<-admixPopsREDtot_best[-1,] # the first row of admixPopsREDtot_best consists of NAs; remove this row
  admixPopsREDtot_long<-admixPopsREDtot_long[-1,] # the first row of admixPopsREDtot_long consists of NAs; remove this row
  
  # now tabulate which target populations we accept as potentially admixed: 
  # consider populations to be admixture candidates if they appear as admixed in 5 or more K levels
  
  cand_pops<-names(which(table(unique(select(admixPopsREDtot_long,PopName,K_level))$PopName)>4)) 
  filtered_triplets_long<-filter(admixPopsREDtot_long, PopName %in% cand_pops)

  # for the considered candidate admixture populations, we take note of how often 
  # a) each of the targets, b) each of the triplets, c) each of the first source candidates, d) each of the second source candidates
  # occur over all Ks to guide us in assigning the source populations 
  
  filtered_triplets_long$TripletMerged <- apply(filtered_triplets_long, 1, function(x) paste(x[1], "~", x[3], ",", x[4],sep=" "))
  
  TargetFreqCount<-as.data.frame(table(unique(select(admixPopsREDtot_long,PopName,K_level))$PopName))
  TripletFreqCount<-as.data.frame(table(filtered_triplets_long$TripletMerged))
  FirstFreqCount<-as.data.frame(tabyl(unique(select(filtered_triplets_long, PopName,FIRSTpopFST,K_level)),PopName, FIRSTpopFST))
  SecondFreqCount<-as.data.frame(tabyl(unique(select(filtered_triplets_long, PopName,SECONDpopFST,K_level)),PopName, SECONDpopFST))
  
  filtered_triplets_long$TargetFreqCount <- TargetFreqCount$Freq[match(filtered_triplets_long$PopName,TargetFreqCount$Var1)]
  filtered_triplets_long$TripletFreqCount <- TripletFreqCount$Freq[match(filtered_triplets_long$TripletMerged,TripletFreqCount$Var1)]
  filtered_triplets_long$FirstGivenTargetFreqCount <- apply(filtered_triplets_long, 1, function(x) if (is.na(x[3])){NA} else {FirstFreqCount[which(FirstFreqCount$PopName==x[1]),x[3]]})
  filtered_triplets_long$SecondGivenTargetFreqCount <- apply(filtered_triplets_long, 1, function(x) if (is.na(x[4])){NA} else {SecondFreqCount[which(SecondFreqCount$PopName==x[1]),x[4]]})
  

  # reorder the data frame for case-by-case consideration by decreasing relevance
  filtered_triplets_long<-arrange(unique(select(filtered_triplets_long,-K_level)),
                                  desc(TargetFreqCount),
                                  PopName,
                                  desc(TripletFreqCount),
                                  desc(FirstGivenTargetFreqCount),
                                  desc(SecondGivenTargetFreqCount))
  
  # keep track of the glottocodes and families assignable to the populations in each triplet
  # (Family ==  family according to Glottolog classification)
  
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","glottocodeBase")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","glottocodeBase")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","glottocodeBase")),by=c("SECONDpopFST"="PopName"))
   
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","GBIstatus")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","GBIstatus")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","GBIstatus")),by=c("SECONDpopFST"="PopName"))
   
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","gbi.logical.full.proxy")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","gbi.logical.full.proxy")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","gbi.logical.full.proxy")),by=c("SECONDpopFST"="PopName"))
  
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","proxy.gbi.logical.full.density")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","proxy.gbi.logical.full.density")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","proxy.gbi.logical.full.density")),by=c("SECONDpopFST"="PopName"))
  
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","TLIstatus")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","TLIstatus")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","TLIstatus")),by=c("SECONDpopFST"="PopName"))
  
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","tli.logical.full.proxy")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","tli.logical.full.proxy")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","tli.logical.full.proxy")),by=c("SECONDpopFST"="PopName"))
  
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","proxy.tli.logical.full.density")),by="PopName")
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","proxy.tli.logical.full.density")),by=c("FIRSTpopFST"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,select(infopop,c("PopName","proxy.tli.logical.full.density")),by=c("SECONDpopFST"="PopName"))
  
  names(filtered_triplets_long)<-c("TargetPop","SourcePop1","SourcePop2","TripletMerged",
                                   "TargetFreqCount","TripletFreqCount","FirstGivenTargetFreqCount","SecondGivenTargetFreqCount",
                                   "TargetPopGlottocodeBase","SourcePop1GlottocodeBase","SourcePop2GlottocodeBase",
                                   "TargetPopGBIstatus","SourcePop1GBIstatus","SourcePop2GBIstatus",
                                   "TargetPopGBIproxy","SourcePop1GBIproxy","SourcePop2GBIproxy",
                                   "TargetPopGBIproxyCoverage","SourcePop1GBIproxyCoverage","SourcePop2GBIproxyCoverage",
                                   "TargetPopTLIstatus","SourcePop1TLIstatus","SourcePop2TLIstatus",
                                   "TargetPopTLIproxy","SourcePop1TLIproxy","SourcePop2TLIproxy",
                                   "TargetPopTLIproxyCoverage","SourcePop1TLIproxyCoverage","SourcePop2TLIproxyCoverage")
  
  pops=unique(c(unique(filtered_triplets_long$TargetPop),unique(filtered_triplets_long$SourcePop1),unique(filtered_triplets_long$SourcePop2)))
  matchingTable<-select(infopop[infopop$PopName%in%pops,],c("PopName","glottologFamily"))
  names(matchingTable)[2]<-"Family"

  filtered_triplets_long<-left_join(filtered_triplets_long,matchingTable,by=c("TargetPop"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,matchingTable,by=c("SourcePop1"="PopName"))
  filtered_triplets_long<-left_join(filtered_triplets_long,matchingTable,by=c("SourcePop2"="PopName"))
  names(filtered_triplets_long)[30:32]<-c("TargetFamily",
                                            "Source1Family",
                                            "Source2Family")
  
  # the language family spoken by target population must be of the same language family as that of one but not both of its source populations
  filtered_triplets_long_familyfilter<-filter(filtered_triplets_long,
                                                (Source1Family!=Source2Family) &
                                                  ((TargetFamily==Source1Family)|(TargetFamily==Source2Family)))
  
  
  output=list(admixPopsREDtot_best,filtered_triplets_long_familyfilter)
  return(output)
}

admixPops90 <- FindAdmixtureCandidates(thresholdMIN = thresholdMIN_1)
onlyAdmixed90_short <- admixPops90[[1]][which(admixPops90[[1]]$CANDIDATE=="admixed"),]
triplets_filtered_for_family_90 <- admixPops90[[2]]

admixPops80<-FindAdmixtureCandidates(thresholdMIN = thresholdMIN_2)
onlyAdmixed80_short<-admixPops80[[1]][which(admixPops80[[1]]$CANDIDATE=="admixed"),]
triplets_filtered_for_family_80<-admixPops80[[2]]

admixPops70<-FindAdmixtureCandidates(thresholdMIN = thresholdMIN_3)
onlyAdmixed70_short<-admixPops70[[1]][which(admixPops70[[1]]$CANDIDATE=="admixed"),]
triplets_filtered_for_family_70<-admixPops70[[2]]

write.csv(triplets_filtered_for_family_90,"output/longlists/threshold90percent/triplet_longlist_90.csv")
write.csv(triplets_filtered_for_family_80,"output/longlists/threshold80percent/triplet_longlist_80.csv")
write.csv(triplets_filtered_for_family_70,"output/longlists/threshold70percent/triplet_longlist_70.csv")

write.csv(onlyAdmixed90_short,"output/longlists/threshold90percent/assignment_aid_fst_90.csv")
write.csv(onlyAdmixed80_short,"output/longlists/threshold80percent/assignment_aid_fst_80.csv")
write.csv(onlyAdmixed70_short,"output/longlists/threshold70percent/assignment_aid_fst_70.csv")
