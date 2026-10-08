rm(list=ls())

library(tidyr)
library(tidyverse)
library(ggfortify)
library(ggrepel)
library(testthat)

# read in glottolog family register
threshold_70 <- read.csv("output/longlists/threshold70percent/triplet_longlist_70.csv", row.names=1)
threshold_80 <- read.csv("output/longlists/threshold80percent/triplet_longlist_80.csv", row.names=1)
threshold_90 <- read.csv("output/longlists/threshold90percent/triplet_longlist_90.csv", row.names=1)

extra_assessment_70 <- read.csv("output/longlists/threshold70percent/assignment_aid_fst_70.csv", row.names=1)
extra_assessment_80 <- read.csv("output/longlists/threshold80percent/assignment_aid_fst_80.csv", row.names=1)
extra_assessment_90 <- read.csv("output/longlists/threshold90percent/assignment_aid_fst_90.csv", row.names=1)

# confirm that the the triplets retrieved from the 80% or 90% thresholds are all also retrieved from the 70% threshold
expect_false(any(threshold_80$TripletMerged%in%threshold_70$TripletMerged==F))
expect_false(any(threshold_90$TripletMerged%in%threshold_70$TripletMerged==F))

extra_assessment_70[extra_assessment_70==""]<-NA
extra_assessment_70[extra_assessment_70=="ND"]<-NA

extra_assessment_80[extra_assessment_80==""]<-NA
extra_assessment_80[extra_assessment_80=="ND"]<-NA

extra_assessment_90[extra_assessment_90==""]<-NA
extra_assessment_90[extra_assessment_90=="ND"]<-NA

# add a column specifying whether component1 or component2 represent the component with the same family as the target:
threshold_70$WhichSourcePopHasSameFamilyAsTarget <-
  apply(threshold_70,1,function(x) 
    if(is.na(x[30])){"target info missing"}
    else if (is.na(x[31])){"source1 info missing"}
    else if (is.na(x[32])){"source2 info missing"}
    else if(x[31]==x[30]&x[32]!=x[30])
    {"1"} 
    else if(x[31]!=x[30]&x[32]==x[30])
    {"2"}
    else if(x[31]==x[30]&x[32]==x[30]){"both"}
    else{"neither"})

threshold_70$WhichSourcePopHasDifferentFamilyAsTarget<-
  apply(threshold_70,1,function(x) 
    if(is.na(x[30])){"target info missing"}
    else if (is.na(x[31])){"source1 info missing"}
    else if (is.na(x[32])){"source2 info missing"}
    else if(x[31]==x[30]&x[32]!=x[30])
    {"2"} 
    else if(x[31]!=x[30]&x[32]==x[30])
    {"1"}
    else if(x[31]==x[30]&x[32]==x[30]){"neither"}
    else{"both"})

threshold_80$WhichSourcePopHasSameFamilyAsTarget <-
  apply(threshold_80,1,function(x) 
    if(is.na(x[30])){"target info missing"}
    else if (is.na(x[31])){"source1 info missing"}
    else if (is.na(x[32])){"source2 info missing"}
    else if(x[31]==x[30]&x[32]!=x[30])
    {"1"} 
    else if(x[31]!=x[30]&x[32]==x[30])
    {"2"}
    else if(x[31]==x[30]&x[32]==x[30]){"both"}
    else{"neither"})

threshold_80$WhichSourcePopHasDifferentFamilyAsTarget<-
  apply(threshold_80,1,function(x) 
    if(is.na(x[30])){"target info missing"}
    else if (is.na(x[31])){"source1 info missing"}
    else if (is.na(x[32])){"source2 info missing"}
    else if(x[31]==x[30]&x[32]!=x[30])
    {"2"} 
    else if(x[31]!=x[30]&x[32]==x[30])
    {"1"}
    else if(x[31]==x[30]&x[32]==x[30]){"neither"}
    else{"both"})

threshold_90$WhichSourcePopHasSameFamilyAsTarget <-
  apply(threshold_90,1,function(x) 
    if(is.na(x[30])){"target info missing"}
    else if (is.na(x[31])){"source1 info missing"}
    else if (is.na(x[32])){"source2 info missing"}
    else if(x[31]==x[30]&x[32]!=x[30])
    {"1"} 
    else if(x[31]!=x[30]&x[32]==x[30])
    {"2"}
    else if(x[31]==x[30]&x[32]==x[30]){"both"}
    else{"neither"})

threshold_90$WhichSourcePopHasDifferentFamilyAsTarget<-
  apply(threshold_90,1,function(x) 
    if(is.na(x[30])){"target info missing"}
    else if (is.na(x[31])){"source1 info missing"}
    else if (is.na(x[32])){"source2 info missing"}
    else if(x[31]==x[30]&x[32]!=x[30])
    {"2"} 
    else if(x[31]!=x[30]&x[32]==x[30])
    {"1"}
    else if(x[31]==x[30]&x[32]==x[30]){"neither"}
    else{"both"})

# keep only well-defined triplets
threshold_70_candidates <- filter(threshold_70,WhichSourcePopHasSameFamilyAsTarget%in%c("1","2","both"))
threshold_80_candidates <- filter(threshold_80,WhichSourcePopHasSameFamilyAsTarget%in%c("1","2","both"))
threshold_90_candidates <- filter(threshold_90,WhichSourcePopHasSameFamilyAsTarget%in%c("1","2","both"))

# keep only triplets, for which we have linguistic assignments/proxies for ... 
# ... the target:
threshold_70_candidates <- filter(threshold_70_candidates,(TargetPopGBIstatus!="no proxy")|(TargetPopTLIstatus!="no proxy"))
threshold_80_candidates <- filter(threshold_80_candidates,(TargetPopGBIstatus!="no proxy")|(TargetPopTLIstatus!="no proxy"))
threshold_90_candidates <- filter(threshold_90_candidates,(TargetPopGBIstatus!="no proxy")|(TargetPopTLIstatus!="no proxy"))

# ... as well as for the different-family component:
threshold_70_candidates <- filter(threshold_70_candidates,
                                  (WhichSourcePopHasDifferentFamilyAsTarget=="1"&
                                     SourcePop1GBIstatus=="no proxy" & SourcePop1TLIstatus=="no proxy")==F)
threshold_70_candidates <- filter(threshold_70_candidates,
                                  (WhichSourcePopHasDifferentFamilyAsTarget=="2"&
                                     SourcePop2GBIstatus=="no proxy" & SourcePop2TLIstatus=="no proxy")==F)

threshold_80_candidates <- filter(threshold_80_candidates,
                                  (WhichSourcePopHasDifferentFamilyAsTarget=="1"&
                                     SourcePop1GBIstatus=="no proxy" & SourcePop1TLIstatus=="no proxy")==F)
threshold_80_candidates <- filter(threshold_80_candidates,
                                  (WhichSourcePopHasDifferentFamilyAsTarget=="2"&
                                     SourcePop2GBIstatus=="no proxy" & SourcePop2TLIstatus=="no proxy")==F)

threshold_90_candidates <- filter(threshold_90_candidates,
                                  (WhichSourcePopHasDifferentFamilyAsTarget=="1"&
                                     SourcePop1GBIstatus=="no proxy" & SourcePop1TLIstatus=="no proxy")==F)
threshold_90_candidates <- filter(threshold_90_candidates,
                                  (WhichSourcePopHasDifferentFamilyAsTarget=="2"&
                                     SourcePop2GBIstatus=="no proxy" & SourcePop2TLIstatus=="no proxy")==F)

## continue only with the 70% threshold only, but log for each triplet at which thresholds it is found
threshold_70_candidates$Thresholds <- apply(threshold_70_candidates, 1, function(x)if(x[4]%in%threshold_90_candidates$TripletMerged){"70%, 80%, 90%"}else if(x[4]%in%threshold_80_candidates$TripletMerged){"70%, 80%"}else{"70%"})

## add descriptive columns
threshold_70_candidates$FamilySourceWhichSameAsTarget<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[33]=="1"){x[31]}
    else if (x[33]=="2"){x[32]})

threshold_70_candidates$FamilySourceWhichDifferentFromTarget<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[33]=="1"){x[32]}
    else if (x[33]=="2"){x[31]})

threshold_70_candidates$TripletFamily<-
  apply(threshold_70_candidates, 1, function(x) 
    paste(x[30], "~", x[36], ",", x[37],sep=" "))

threshold_70_candidates$AlterFamilySourcePop<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[2]}
    else if (x[34]=="2"){x[3]})

threshold_70_candidates$AlterFamilySourceGivenTargetFreqCount<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[7]}
    else if (x[34]=="2"){x[8]})

threshold_70_candidates$AlterFamilySourceGlottocodeBase<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[10]}
    else if (x[34]=="2"){x[11]})

threshold_70_candidates$AlterFamilySourceGBIstatus<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[13]}
    else if (x[34]=="2"){x[14]})

threshold_70_candidates$AlterFamilySourceGBIproxy<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[16]}
    else if (x[34]=="2"){x[17]})

threshold_70_candidates$AlterFamilySourceGBIproxyCoverage<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[19]}
    else if (x[34]=="2"){x[20]})

threshold_70_candidates$AlterFamilySourceTLIstatus<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[22]}
    else if (x[34]=="2"){x[23]})

threshold_70_candidates$AlterFamilySourceTLIproxy<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[25]}
    else if (x[34]=="2"){x[26]})

threshold_70_candidates$AlterFamilySourceTLIproxyCoverage<-
  apply(threshold_70_candidates,1,function(x) 
    if(x[34]=="1"){x[28]}
    else if (x[34]=="2"){x[29]})


massively_filtered <- unique(select(
  threshold_70_candidates,c("TargetPop", "AlterFamilySourcePop", "Thresholds", "TargetFamily","FamilySourceWhichDifferentFromTarget", "TripletFamily", "TargetFreqCount","AlterFamilySourceGivenTargetFreqCount",
                   "TargetPopGlottocodeBase","AlterFamilySourceGlottocodeBase","TargetPopGBIstatus","AlterFamilySourceGBIstatus", "TargetPopGBIproxy","AlterFamilySourceGBIproxy", "TargetPopGBIproxyCoverage",
                   "AlterFamilySourceGBIproxyCoverage", "TargetPopTLIstatus","AlterFamilySourceTLIstatus", "TargetPopTLIproxy","AlterFamilySourceTLIproxy", "TargetPopTLIproxyCoverage","AlterFamilySourceTLIproxyCoverage")))

# log for each combination whether it is available in GBI, TLI, both or neither
massively_filtered$GBIavailabilityStatus<-apply(massively_filtered,1,function(x) 
  if(x[11]!="no proxy" & x[12]!="no proxy"){"GBI available"} 
  else {"GBI NOT available"})

massively_filtered$TLIavailabilityStatus<-apply(massively_filtered,1,function(x) 
  if(x[17]!="no proxy" & x[18]!="no proxy"){"TLI available"} 
  else {"TLI NOT available"})

massively_filtered$OverallAvailabilityStatus<-apply(massively_filtered,1,function(x) 
  if(x[23]=="GBI available" & x[24]=="TLI available"){"both"} 
  else if(x[23]=="GBI NOT available" & x[24]=="TLI NOT available"){"neither"}
  else if(x[23]=="GBI NOT available" & x[24]=="TLI available"){"TLI only"} 
  else if(x[23]=="GBI available" & x[24]=="TLI NOT available"){"GBI only"})

table(massively_filtered$OverallAvailabilityStatus) # 28 triplets absent in both GBI and TLI

massively_filtered <- massively_filtered %>% filter(OverallAvailabilityStatus != "neither")

write_csv(massively_filtered,"output/longlists/longlist_for_manual_curation.csv")

