library(ranger)
library(tidyverse)
library(patchwork)
library(ggplot2)
library(pROC)

theme_set(theme_minimal(base_size = 20)%+replace% theme(panel.grid.minor.y = element_blank()))

#pre-prepaired file in Takou et al (2022), BioRXiv.
load("allGenes_motifs.RData")

motifsnames=unique(motifs$motifid)
motifsnames_lyr=paste(motifsnames,"lyr",sep="_")
motifsnames_hal=paste(motifsnames,"hal",sep="_")
motifsnames_tha=paste(motifsnames,"tha",sep="_")

features=unique(motifs$feature)

motifnames_all=character(0)
for (f in features){
  for (s in c("lyr","hal","tha")){
    motifnames_all=c(motifnames_all,paste(motifsnames,s,f,sep="_"))
  }
}

#add clustering information column
clusttab=read_csv("matrix-clustering_cluster_root_motifs.tf.csv")

clusttab=dplyr::select(clusttab,cluster,motif)

motifs=left_join(motifs,clusttab,by=c("motifid"="motif"))

motifs$cluster[is.na(motifs$cluster)]=motifs$motifid[is.na(motifs$cluster)]

motifs$clust_feat=interaction(motifs$cluster,motifs$feature)

##Keep only the Athaliana information
motifs2 <- motifs[motifs$genome == "thaliana",]
save(motifs2, file="motifs2.RData")

######Hannah et al. 2006: read in Gene data#########
###load the data
col <- read.table("../Data/col_degs.csv", sep=",")

##function to run the random forest
#degs --> the table of DEG status per gene
#string_of_RData --> name where to save the results
#type --> which is the DEG class we need to check
#notype --> the DEG class we exclude
runRandomForest <- function(degs, string_of_RData, type, notype){
  #prepare the input dataset
  colnames(degs) <- c("Genes", "p", "DEG")
  motifsnames=unique(motifs2$tfname)
  degs[,motifsnames]=0
  
  #merge with motifs
  for (i in 1:nrow(degs)){
    if (i%%1000==0) {print(i)}
    gene=degs$Genes[i]
    if (gene %in% motifs2$thaliana_bestmatch){
      curr_motifs=filter(motifs2,thaliana_bestmatch==gene)
    }
    else{degs[i,motifsnames]=NA;next()}
    #this method takes number of matches into account
    t=table(curr_motifs$tfname)
    degs[i,names(t)]=t
  }
  
  
  degs$total=rowSums(degs[,motifsnames],na.rm = T)
  
  #remove the ones that have 0 total of DEGs (it is the result of NAs)
  degs <- degs[degs$total > 0,]
  
  names(degs)=make.names(names(degs))
  motifsnames=make.names(motifsnames)
  tmp <- replace(degs$DEG, degs$DEG == "no", FALSE)
  tmp <- replace(tmp, tmp == type, TRUE)
  tmp <- replace(tmp, tmp == notype, NA)
  summary(tmp)
  degs$DEG <- as.factor(tmp)
  degs <- degs[!is.na(degs$DEG),]
  
  #run the random forest
  deg_rf=ranger(data=degs[,colnames(degs) %in% c("DEG",motifsnames,"total")],formula = DEG~.,num.trees = 500,mtry=200,importance = "impurity_corrected")
  deg_rf
  
  save(deg_rf,file=string_of_RData)
  
  cor.pred <- sum(degs$DEG==deg_rf$predictions) / length(degs$DEG)
  write.table(cor.pred, paste("cor_prediction",string_of_RData, "txt", sep="."))
  barplot(sort(importance(deg_rf))[1:50],horiz = T)
  
  #permutations to estimate the importance
  importance=as.data.frame(importance_pvalues(deg_rf))
  importance=as.data.frame(importance_pvalues(deg_rf,method = "altmann",
                                              data=degs[,colnames(degs) %in% c("DEG",motifsnames,"total")],
                                              formula = DEG~.,num.threads=7,num.permutations = 100))
  importance=importance[order(importance[,1],decreasing = T),]
  importance$motif=rownames(importance)
  importance$sig=importance$pvalue<0.05
  
  save(importance,file=paste("importance",string_of_RData, sep="_"))
  
  summary(importance)
  
  svg(paste(string_of_RData, "png", sep="."),height=12,width=10)
  print(ggplot(data=importance[1:50,],aes(x=reorder(motif,importance),y=importance,fill=sig))+geom_bar(stat="identity")+coord_flip()+xlab("factor")+
          theme(panel.grid.major.y = element_blank(),legend.position = "top")+scale_fill_manual(name="significant?",values=c("grey25","#BADA55")))
  dev.off()
  
  return(cor.pred)
  
}

#call the function for up-DEG and no-DEG seperately
cor.pred.col <- runRandomForest(col, "col_up_randomForest.RData", "up", "down")
cor.pred.col <- runRandomForest(col, "col_down_randomForest.RData", "down", "up")

###Estimate and permute calculations for prAUC
##function to estimate AUC
#degs --> the table of DEG status per gene
#string_of_RData --> name where to save the results
#type --> which is the DEG class we need to check
#notype --> the DEG class we exclude
permuteAUC <- function(degs, string_of_RData, motifs2, type, notype){
  colnames(degs) <- c("Genes", "p", "DEG")
  motifsnames=unique(motifs2$tfname)
  degs[,motifsnames]=0
  
  #merge with motifs
  for (i in 1:nrow(degs)){
    if (i%%1000==0) {print(i)}
    gene=degs$Genes[i]
    if (gene %in% motifs2$thaliana_bestmatch){
      curr_motifs=filter(motifs2,thaliana_bestmatch==gene)
    }
    else{degs[i,motifsnames]=NA;next()}
    #this method takes number of matches into account
    t=table(curr_motifs$tfname)
    degs[i,names(t)]=t
  }
  degs$total=rowSums(degs[,motifsnames],na.rm = T)
  
  #remove the ones that have 0 total of DEGs (it is the result of NAs)
  degs <- degs[degs$total > 0,]
  
  names(degs)=make.names(names(degs))
  motifsnames=make.names(motifsnames)
  tmp <- replace(degs$DEG, degs$DEG == "no", FALSE)
  tmp <- replace(tmp, tmp == type, TRUE)
  tmp <- replace(tmp, tmp == notype, NA)
  summary(tmp)
  degs$DEG <- as.factor(tmp)
  degs <- degs[!is.na(degs$DEG),]
  
  #get the 20% of the set as test
  x <- round((nrow(degs) * 20) / 100)
  #print(head(degs))
  test_s <- sample(degs$Genes, x)
  #print(test_s)
  test_degs <- degs[degs$Genes %in% test_s,]
  train_degs <- degs[!(degs$Genes %in% test_s),]
  
  ###permute the train set
  perauc <- NULL
  for (i in 1:100){
    print(i)
    tmp <- train_degs$DEG
    train_degs$DEG <- sample(tmp)
    ##Rerun the rf here
    deg_rf=ranger(data=train_degs[,colnames(train_degs) %in% c("DEG",motifsnames,"total")],formula = DEG~.,num.trees = 500,mtry=200)
    #deg_rf
    predsRF <- predict(deg_rf, dat = test_degs[,colnames(test_degs) %in% c("DEG",motifsnames,"total")])
    #return(predsRF)
    #get the AUC
    rocRFall=roc(as.numeric(test_degs$DEG), as.numeric(predsRF$predictions))
    aucRF=pROC::auc(rocRFall)
    perauc <- c(perauc, aucRF)
  }
  
  return(perauc)
}


##function to permute for pvalue significance level based on prAUC
#degs --> the table of DEG status per gene
#string_of_RData --> name where to save the results
#type --> which is the DEG class we need to check
#notype --> the DEG class we exclude
estimateAUC <- function(degs, string_of_RData, motifs2, type, notype){
  colnames(degs) <- c("Genes", "p", "DEG")
  motifsnames=unique(motifs2$tfname)
  degs[,motifsnames]=0
  
  #merge with motifs
  for (i in 1:nrow(degs)){
    if (i%%1000==0) {print(i)}
    gene=degs$Genes[i]
    if (gene %in% motifs2$thaliana_bestmatch){
      curr_motifs=filter(motifs2,thaliana_bestmatch==gene)
    }
    else{degs[i,motifsnames]=NA;next()}
    #this method takes number of matches into account
    t=table(curr_motifs$tfname)
    degs[i,names(t)]=t
  }
  degs$total=rowSums(degs[,motifsnames],na.rm = T)
  
  #remove the ones that have 0 total of DEGs (it is the result of NAs)
  degs <- degs[degs$total > 0,]
  
  names(degs)=make.names(names(degs))
  motifsnames=make.names(motifsnames)
  tmp <- replace(degs$DEG, degs$DEG == "no", FALSE)
  tmp <- replace(tmp, tmp == type, TRUE)
  tmp <- replace(tmp, tmp == notype, NA)
  summary(tmp)
  degs$DEG <- as.factor(tmp)
  degs <- degs[!is.na(degs$DEG),]
  
  #get the 20% as test
  x <- round((nrow(degs) * 20) / 100)
  #print(head(degs))
  test_s <- sample(degs$Genes, x)
  #print(test_s)
  test_degs <- degs[degs$Genes %in% test_s,]
  train_degs <- degs[!(degs$Genes %in% test_s),]
  
  ##Rerun the rf here
  deg_rf=ranger(data=train_degs[,colnames(train_degs) %in% c("DEG",motifsnames,"total")],formula = DEG~.,num.trees = 500,mtry=200)
  #deg_rf
  predsRF <- predict(deg_rf, dat = test_degs[,colnames(test_degs) %in% c("DEG",motifsnames,"total")])
  #return(predsRF)
  #get the AUC
  rocRFall=roc(as.numeric(test_degs$DEG), as.numeric(predsRF$predictions))
  aucRF=pROC::auc(rocRFall)
  print(aucRF)
  return(aucRF)
}


###Run
col_auc_up <- estimateAUC(col, "col_randomForest.RData", motifs2, "up", "down") 
col_auc_down <- estimateAUC(col, "col_randomForest.RData", motifs2, "down", "up") 
save.image("permuteAUC.RData")
load("permuteAUC.RData")
col_per_up <- permuteAUC(col, "col_randomForest.RData", motifs2, "up", "down") 
col_per_down <- permuteAUC(col, "col_randomForest.RData", motifs2, "down", "up") 
save.image("permuteAUC.RData")
col_auc_up
col_auc_down

pval_down = sum(col_per_down > col_auc_down) / 100
hist(col_per_down)
pval_up = sum(col_per_up > col_auc_up) / 100
hist(col_per_up)
