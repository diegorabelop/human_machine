##### RESUMO E TITULOS
library(readxl)
random_data <- read_excel("Downloads/random_data_primary_consensus.xlsx")

data1 <- random_data[,c("Title", "Abstract", "Final_decision")]
#coombining Abstract and Title
data1$Text <- paste(data1$Title,data1$Abstract)

data <- data1[,c("Final_decision", "Text")]
print(subset(data,select =c(1, Text)))

# Preparing the data
data <- as.data.frame(data)
data$Final_decision = as.factor(data$Final_decision)
data$Text = as.character(data$Text)
print(subset(data,select =c(1, Final_decision)))

str(data)
table(data$Final_decision)

library(tm)
library(SnowballC)
glimpse(data)
# Step 1 - Create a corpus text
corpus = Corpus(VectorSource(data$Text))
corpus[[1]][1]
data$Final_decision[1]
data$Text[1]
## Step 2 - Conversion to Lowercase
corpus = tm_map(corpus, PlainTextDocument)
corpus = tm_map(corpus, tolower)
corpus[[1]][1]
#Step 3 - Removing Punctuation
corpus = tm_map(corpus, removePunctuation)
corpus[[1]][1]
#Step 4 - Removing Stopwords and other words
corpus = tm_map(corpus, removeWords, c("objective","background", "introduction", "purpose", "method", "result", "conclusion", "limitation", 
                                       stopwords("english")))

corpus[[1]][1]  
# Step 5 - Stemming: reducing the number of inflectional forms of words
corpus = tm_map(corpus, stemDocument)
corpus[[1]][1]  
# Step 6 - Create Document Term Matrix
frequencies = DocumentTermMatrix(corpus)
sparse = removeSparseTerms(frequencies, 0.995) #remove sparse terms
tSparse = as.data.frame(as.matrix(sparse)) #convert into data frame
colnames(tSparse) = make.names(colnames(tSparse)) #all the variable names R-friendly
tSparse$final_decision = data$Final_decision #add the outcome

prop.table(table(tSparse$final_decision))
table(tSparse$final_decision)
# Hold out
library(caret)
set.seed(123)
intrain <- createDataPartition(y = tSparse$final_decision, p= 0.8, list = FALSE)
trainSparse <-  tSparse[intrain,]
testSparse <- tSparse[-intrain,]
dim(trainSparse); dim(testSparse)
# % outcome
prop.table(table(tSparse$final_decision))
prop.table(table(trainSparse$final_decision))
prop.table(table(testSparse$final_decision))

table(trainSparse$final_decision)
table(testSparse$final_decision)


## Parallel processing
library(doParallel)
cl <- makeCluster(8)
registerDoParallel(cl)
foreach(i=1:4) %dopar% sqrt(i)

#Training
library(randomForest)
set.seed(100)
trainSparse$final_decision = as.factor(trainSparse$final_decision)
testSparse$final_decision = as.factor(testSparse$final_decision)

levels(trainSparse$final_decision)
levels(trainSparse$final_decision) <- c("no", "yes")
levels(testSparse$final_decision)
levels(testSparse$final_decision) <- c("no", "yes")

############ ELastic Net ######
library(caret)
library(glmnet)
#up regulation
set.seed(108) 
ctrl.EN_up <- trainControl(method="repeatedcv", number = 10, repeats = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "up",preProcess=c("center", "scale"))
up_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_up, tuneLength = 10, metric = "ROC")
#Smote
library(DMwR)
set.seed(108) 
ctrl.EN_smote <- trainControl(method="repeatedcv", number = 10,repeats = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "smote", preProcess=c("center", "scale"))
smote_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_smote, tuneLength = 10, metric = "ROC")
#Rose
library(ROSE)
set.seed(108) 
ctrl.EN_rose <- trainControl(method="repeatedcv", number = 10, repeats = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "rose", preProcess=c("center", "scale"))
rose_outside <- train(final_decision ~., data = trainSparse, method = "glmnet", trControl = ctrl.EN_rose, tuneLength = 10, metric = "ROC")

#Choosing the best training model 
library(mlbench)
library(caret)
result_model_EN <- resamples(list(up=up_outside, rose=rose_outside, smote=smote_outside))
summary(result_model_EN) 
bwplot(result_model_EN) 
dotplot(result_model_EN)

# confusion matrix
library(e1071)
test_pred_rose_EN <- predict(rose_outside, newdata = testSparse[,-2054])
matrixConfusao_rose_EN <- confusionMatrix(test_pred_rose_EN, testSparse$final_decision, positive = "yes")
matrixConfusao_rose_EN
# ROC curve plot
library(caret)
library(pROC)
p_rose_EN <- predict (rose_outside, testSparse[,-2054])
p_prob_rose_EN <- predict (rose_outside, testSparse[,-2054], type = "prob")
confusionMatrix(p_rose_EN, testSparse$final_decision)
print(confusionMatrix(p_rose_EN, testSparse$final_decision, positive = "yes"))
r_rose_EN <- roc (testSparse$final_decision, p_prob_rose_EN[,"yes"])
plot(r_rose_EN)
r_rose_EN$auc


#### changing the threshold ####
library(SDMTools)
obs_text <- testSparse$final_decision
levels(obs_text)
levels(obs_text) <- c("0", "1")
obs_text <- as.numeric(as.character(obs))
obs_text

# model predictions in probabilities
rf.predict_text <- predict(rose_outside, testSparse[, -2054], type = "prob")
predictions_text <- as.vector(rf.predict_text[, 2])
predictions_text

confusion_df_text <- data.frame(obs_text, predictions_text)
threshold_seq <- seq(0, 1, by = 0.01)

confMatrix <- function(i, obs, predictions, ...){
  require(caret)
  require(SDMTools)
  
  conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
  cm_table <- as.table(conf_matrix)
  cm <- confusionMatrix(cm_table, positive = "1")
  p_acc <- cm$overall[6]
  acc <- cm$overall[1]
  null_acc <- cm$overall[5]
  ppv <- cm$byClass[3]
  npv <- cm$byClass[4]
  
  result <-  c("limiar" = i, cm$byClass[1], cm$byClass[2], 
               "AB" = as.numeric(c((cm$byClass[1]+cm$byClass[2]) / 2)), 
               acc, p_acc, null_acc, ppv, npv)
  result
}

library(purrr)
library(plyr)
confMatrix(0.5, obs, predictions)
result_list_text <- map(threshold_seq, confMatrix, obs = obs_text, predictions = predictions_text)
result_df_text <- ldply(result_list_text)
result_df_text



##threshold plot
plotPerformancesMeasures <- function(result_df_text){
  require(ggplot2)
  require(reshape2)
  
  colnames(result_df_text)
  result_selected <- result_df_text[, c("limiar", "AB", "Pos Pred Value", "Neg Pred Value")]
  colnames(result_selected) <- c("Cutoff", "Balanced Accuracy", "Pos Pred Value", "Neg Pred Value")
  result_long <- melt(result_selected, id.vars = "Cutoff")
  
  p1 <-ggplot(result_long, aes(x=Cutoff, y=value, group=variable)) +
    geom_line(aes(color=variable)) +
    geom_point(aes(color=variable)) +
    theme(panel.background = element_blank(), 
          axis.line = element_line(colour = "grey"),
          legend.title = element_blank())
  p1
  
  result_selected <- result_df_text[, c("limiar", "AB", "Sensitivity", "Specificity")]
  colnames(result_selected) <- c("Cutoff", "Balanced Accuracy", "Sensitivity", "Specificity")
  result_long <- melt(result_selected, id.vars = "Cutoff")
  
  p2 <-ggplot(result_long, aes(x=Cutoff, y=value, group=variable)) +
    geom_line(aes(color=variable)) +
    geom_point(aes(color=variable)) +
    theme(panel.background = element_blank(), 
          axis.line = element_line(colour = "grey"),
          legend.title = element_blank())
  
  p2
  
  return(list("BA_PPV_NPV_Vs_Cutoff" = p1 , "BA_Sen_Spe_Vs_Cutoff" = p2))
  
}

getPerformanceVsCutoff <- function(confusion_df, threshold_seq, positive_class){
  message("positive_class = as.character(0) ou as.character(1)")
  
  require(purrr)
  require(plyr)
  
  
  confMatrix <- function(i, obs, predictions, ...){
    require(caret)
    require(SDMTools)
    
    conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
    cm_table <- as.table(conf_matrix)
    cm <- confusionMatrix(cm_table, positive = positive_class)
    p_acc <- cm$overall[6]
    acc <- cm$overall[1]
    null_acc <- cm$overall[5]
    ppv <- cm$byClass[3]
    npv <- cm$byClass[4]
    
    result <-  c("limiar" = i, cm$byClass[1], cm$byClass[2], 
                 "AB" = as.numeric(c((cm$byClass[1]+cm$byClass[2]) / 2)), 
                 acc, p_acc, null_acc, ppv, npv)
    result
  }
  
  result_list_text <- map(threshold_seq, confMatrix, obs = obs_text, predictions = predictions_text) #aqui se muda em cada modelo
  result_df_text <- ldply(result_list_text)
  return(result_df_text)
}

result_df_text <- getPerformanceVsCutoff(result_df_text, threshold_seq, "1")
plots_onda1 <- plotPerformancesMeasures(result_df_text)
plots_onda1$BA_PPV_NPV_Vs_Cutoff
plots_onda1$BA_Sen_Spe_Vs_Cutoff



###### Random Forest #############
#parallel processing
library(doParallel)
gc()
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

gc()
library(caret)
library(randomForest)
set.seed(108)
grid.rf = expand.grid(.mtry=c(1,2,3,4,5,6,7,8,9,10,11,12,13,14))
#UP sampling
set.seed(108)
ctrl.rf_RF_up = trainControl(method="repeatedcv", number = 10, repeats = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "up")
rf_model_up <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_up, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")

#Smote
set.seed(108)
library(DMwR)
ctrl.rf_RF_smote = trainControl(method="repeatedcv", number = 10, repeats = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "smote")
rf_model_smote <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_smote, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")
#ROSE
set.seed(108)
library(ROSE)
ctrl.rf_RF_rose = trainControl(method="repeatedcv", number = 10, repeats = 10, classProbs = TRUE,savePredictions = TRUE, summaryFunction = twoClassSummary, sampling = "rose")
rf_model_rose <- train(final_decision ~ ., data = trainSparse, method = "rf", trControl = ctrl.rf_RF_rose, tuneGrid= grid.rf, tuneLength = 10, metric = "ROC")


# choosing the best model
library(mlbench)
library(caret)
result_model2 <- resamples(list(up=rf_model_up, rose=rf_model_rose, smote=rf_model_smote))
summary(result_model2)
bwplot(result_model2) 
dotplot(result_model2)

# confusion matrix
library(e1071)
test_pred_up_rf <- predict(rf_model_up, newdata = testSparse[,-2018])
matrixConfusao_up_rf <- confusionMatrix(test_pred_up_rf, testSparse$final_decision, positive = "yes")
matrixConfusao_up_rf
# ROC curve plot
library(caret)
library(pROC)
p_up_rf <- predict (rf_model_up, testSparse[,-2018])
p_prob_up_rf <- predict (rf_model_up, testSparse[,-2018], type = "prob")
confusionMatrix(p_up_rf, testSparse$final_decision)
print(confusionMatrix(p_up_rf, testSparse$final_decision, positive = "yes"))
r_up_rf <- roc (testSparse$final_decision, p_prob_up_rf[,"yes"])
plot(r_up_rf)
r_up_rf$auc

###### Changing the threshold 

# 0 = No, 1 = Yes
library(SDMTools)
obs_text <- testSparse$final_decision
levels(obs_text)
levels(obs_text) <- c("0", "1")
obs_text <- as.numeric(as.character(obs))
obs_text

# model predictions in probabilities
rf.predict_text <- predict(rf_model_up, testSparse[, -2054], type = "prob")
predictions_text <- as.vector(rf.predict_text[, 2])
predictions_text

confusion_df_text <- data.frame(obs_text, predictions_text)
threshold_seq <- seq(0, 1, by = 0.01)

confMatrix <- function(i, obs, predictions, ...){
  require(caret)
  require(SDMTools)
  
  conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
  cm_table <- as.table(conf_matrix)
  cm <- confusionMatrix(cm_table, positive = "1")
  p_acc <- cm$overall[6]
  acc <- cm$overall[1]
  null_acc <- cm$overall[5]
  ppv <- cm$byClass[3]
  npv <- cm$byClass[4]
  
  result <-  c("limiar" = i, cm$byClass[1], cm$byClass[2], 
               "AB" = as.numeric(c((cm$byClass[1]+cm$byClass[2]) / 2)), 
               acc, p_acc, null_acc, ppv, npv)
  result
}


library(purrr)
library(plyr)
confMatrix(0.5, obs, predictions)
result_list_text <- map(threshold_seq, confMatrix, obs = obs_text, predictions = predictions_text)
result_df_text <- ldply(result_list_text)

result_df_text



## Threshold plot
plotPerformancesMeasures <- function(result_df_text){
  require(ggplot2)
  require(reshape2)
  
  colnames(result_df_text)
  result_selected <- result_df_text[, c("limiar", "AB", "Pos Pred Value", "Neg Pred Value")]
  colnames(result_selected) <- c("Cutoff", "Balanced Accuracy", "Pos Pred Value", "Neg Pred Value")
  result_long <- melt(result_selected, id.vars = "Cutoff")
  
  p1 <-ggplot(result_long, aes(x=Cutoff, y=value, group=variable)) +
    geom_line(aes(color=variable)) +
    geom_point(aes(color=variable)) +
    theme(panel.background = element_blank(), 
          axis.line = element_line(colour = "grey"),
          legend.title = element_blank())
  p1
  
  result_selected <- result_df_text[, c("limiar", "AB", "Sensitivity", "Specificity")]
  colnames(result_selected) <- c("Cutoff", "Balanced Accuracy", "Sensitivity", "Specificity")
  result_long <- melt(result_selected, id.vars = "Cutoff")
  
  p2 <-ggplot(result_long, aes(x=Cutoff, y=value, group=variable)) +
    geom_line(aes(color=variable)) +
    geom_point(aes(color=variable)) +
    theme(panel.background = element_blank(), 
          axis.line = element_line(colour = "grey"),
          legend.title = element_blank())
  
  p2
  
  return(list("BA_PPV_NPV_Vs_Cutoff" = p1 , "BA_Sen_Spe_Vs_Cutoff" = p2))
  
}

getPerformanceVsCutoff <- function(confusion_df, threshold_seq, positive_class){
  message("positive_class = as.character(0) ou as.character(1)")
  
  require(purrr)
  require(plyr)
  
  
  confMatrix <- function(i, obs, predictions, ...){
    require(caret)
    require(SDMTools)
    
    conf_matrix <- confusion.matrix(obs, predictions, threshold = i)
    cm_table <- as.table(conf_matrix)
    cm <- confusionMatrix(cm_table, positive = positive_class)
    p_acc <- cm$overall[6]
    acc <- cm$overall[1]
    null_acc <- cm$overall[5]
    ppv <- cm$byClass[3]
    npv <- cm$byClass[4]
    
    result <-  c("limiar" = i, cm$byClass[1], cm$byClass[2], 
                 "AB" = as.numeric(c((cm$byClass[1]+cm$byClass[2]) / 2)), 
                 acc, p_acc, null_acc, ppv, npv)
    result
  }
  
  result_list_text <- map(threshold_seq, confMatrix, obs = obs_text, predictions = predictions_text) #aqui se muda em cada modelo
  result_df_text <- ldply(result_list_text)
  return(result_df_text)
}

result_df_text <- getPerformanceVsCutoff(result_df_text, threshold_seq, "1")
plots_onda1 <- plotPerformancesMeasures(result_df_text)
plots_onda1$BA_PPV_NPV_Vs_Cutoff
plots_onda1$BA_Sen_Spe_Vs_Cutoff

# Training neural networks based on titles + abstracts

library(devtools)
#load("sessions/session_neural_network_caret_keras.RData")

load("text_mining.RData")

library(caret)

nzv <- nearZeroVar(trainSparse, saveMetrics= TRUE)
nzv[nzv$nzv,][1:10,]

nzv_vector <- nzv$nzv
nzv_vector[length(nzv_vector)] <- FALSE
trainSparse_filtered <- trainSparse[, !nzv_vector]
testSparse_filtered <- testSparse[, !nzv_vector]

pp <- preProcess(trainSparse_filtered[, -ncol(trainSparse_filtered)], 
                 method = c("center", "scale"))
pp
train_transformed <- predict(pp, newdata = trainSparse_filtered[, -ncol(trainSparse_filtered)])
test_transformed <- predict(pp, newdata = testSparse_filtered[, -ncol(testSparse_filtered)])

library(purrr)
map(train_transformed, hist)


train_transformed <- data.frame(train_transformed, final_decision = trainSparse_filtered$final_decision)
test_transformed <- data.frame(test_transformed, final_decision = testSparse_filtered$final_decision)

levels(train_transformed$final_decision) <- c("No", "Yes")
levels(test_transformed$final_decision) <- c("No", "Yes")


fitControl_up <- trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 10, 
                              classProbs = TRUE, 
                              summaryFunction = twoClassSummary,
                              sampling = "up")

fitControl_smote <- trainControl(method = "repeatedcv", 
                                 number = 10, 
                                 repeats = 10, 
                                 classProbs = TRUE, 
                                 summaryFunction = twoClassSummary,
                                 sampling = "smote")


fitControl_rose <- trainControl(method = "repeatedcv", 
                                number = 10, 
                                repeats = 10, 
                                classProbs = TRUE, 
                                summaryFunction = twoClassSummary,
                                sampling = "rose")


kerasGrid <-  expand.grid(size = c(700),
                          dropout = c(0, 0.3, 0.75, 0.9),
                          batch_size = 1,
                          lr = 2e-06,
                          rho = 0.9,
                          decay = 0,
                          activation = "relu"
)


class(trainSparse_filtered$final_decision)

library(doParallel)
library(keras)
library(tensorflow)
library(caret)
tf$constant("Hellow Tensorflow")


# smote ----
time_start <- Sys.time()

cl <- makePSOCKcluster(16)
registerDoParallel(cl)

set.seed(1234)
#kerasFit_up <- caret::train(final_decision ~ ., 
#                data = train_transformed,
#               method = "mlpKerasDropout",
#              metric = "ROC",
#             trControl = fitControl,
#            tuneGrid = kerasGrid,
#           verbose = FALSE)


set.seed(1234)
kerasFit_smote <- caret::train(final_decision ~ ., 
                               data = train_transformed,
                               method = "mlpKerasDropout",
                               metric = "ROC",
                               trControl = fitControl_smote,
                               tuneGrid = kerasGrid,
                               verbose = FALSE)


stopCluster(cl)
time_end <- Sys.time()
time_end - time_start

varImp(kerasFit_smote)

plot(kerasFit_smote)

kerasFit_smote$bestTune


data_test_pred = predict(kerasFit_smote, test_transformed)
data_test_pred

# confusion matrix
library(caret)
confusionMatrix(data_test_pred, test_transformed$final_decision, positive = "Yes")

# ROC plot

library(ROCR)
predvec <- ifelse(data_test_pred == "Yes", 1, 0)
realvec <- ifelse(test_transformed$final_decision == "Yes", 1, 0)
pred = prediction(predictions = predvec, labels = realvec)
perf = performance(pred, measure = "tpr", x.measure = "fpr")
plot(perf, main = "ROC curve", col = "blue", lwd = 3)
abline(a=0, b=1, lwd=2, lty=2)

# AUC
perf.auc = performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)

prop.table(table(train_transformed$final_decision))
time_start <- Sys.time()

cl <- makePSOCKcluster(16)
registerDoParallel(cl)

set.seed(1234)

kerasFit_rose <- caret::train(final_decision ~ ., 
                              data = train_transformed,
                              method = "mlpKerasDropout",
                              metric = "ROC",
                              trControl = fitControl_rose,
                              tuneGrid = kerasGrid,
                              verbose = FALSE)


stopCluster(cl)
time_end <- Sys.time()
time_end - time_start

varImp(kerasFit_rose)

plot(kerasFit_rose)

kerasFit_rose$bestTune


data_test_pred = predict(kerasFit_rose, test_transformed)
data_test_pred

data_test_pred_prob = predict(kerasFit_rose, test_transformed, type = "prob")
data_test_pred_prob

library(caret)
confusionMatrix(data_test_pred, test_transformed$final_decision, positive = "Yes")

# ROC curve

library(ROCR)
predvec <- ifelse(data_test_pred == "Yes", 1, 0)
realvec <- ifelse(test_transformed$final_decision == "Yes", 1, 0)
pred = prediction(predictions = predvec, labels = realvec)

# AUC
perf.auc = performance(pred, measure = "auc")
unlist(perf.auc@y.values)

prop.table(table(train_transformed$final_decision))

# Performance vs. cut-off
df_rose <- data.frame("predictions" = data_test_pred_prob, "obs" = test_transformed$final_decision)

saveRDS(df_rose, file = "cache/obs_predictions_prob_df_rose.rds")

# up ----

library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

set.seed(1234)

time_start <- Sys.time()

kerasFit_up <- caret::train(final_decision ~ ., 
                            data = train_transformed,
                            method = "mlpKerasDropout",
                            metric = "ROC",
                            trControl = fitControl_up,
                            tuneGrid = kerasGrid,
                            verbose = FALSE)


stopCluster(cl)
time_end <- Sys.time()
time_end - time_start

#saveRDS(kerasFit_up, file = "cache/neural_network_caret_keras_smote.rds")
save.image("sessions/session_network_caret_keras_31072020.RData")