---
title: "Machine Learning Course Project"
output: html_document
---

The goal of your project is to predict the manner in which subjects did a physical exercise, based on physical measurements.


```r
setwd("C:\\Users\\Oleg\\Documents\\coursera\\courses\\08_PracticalMachineLearning\\project")
library(caret)
dat <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```

Filter out variables that are mostly NA, or are factors, or are not related to measured variables, i.e. arm/forarm/dumbell.  The lda model is used because the data is largely numeric measurements, and it is computationally convenient.  The confusion matrix is used to see the accuracy and kappa.

Cross validation can be used to reduce overfitting, i.e. not relying on error estimates from the training set.  Because there are a large number of parameters, we increase the number cross validation samples.  Leave one out cross validation uses as many samples are there are observations.



```r
largely.na=sapply(dat, function(x) sum(is.na(x))/length(x) > .5 | class(x)=="factor") |
!grepl("arm|belt|dumbbell", names(dat),T)
largely.na=names(largely.na)[largely.na] 
dat1=dat[, - match(largely.na, names(dat))]

modelFit <-  train(dat$classe ~.,data=dat1, method="lda", trainControl="LOOCV")
predictions = predict(modelFit, newdata=dat1)
confusionMatrix(predictions, dat$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4568  586  341  191  133
##          B  121 2429  333  130  611
##          C  444  455 2254  379  323
##          D  429  148  411 2383  344
##          E   18  179   83  133 2196
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7048          
##                  95% CI : (0.6984, 0.7112)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6264          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8186   0.6397   0.6587   0.7410   0.6088
## Specificity            0.9109   0.9245   0.9012   0.9188   0.9742
## Pos Pred Value         0.7850   0.6703   0.5847   0.6415   0.8417
## Neg Pred Value         0.9267   0.9145   0.9259   0.9476   0.9171
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2328   0.1238   0.1149   0.1214   0.1119
## Detection Prevalence   0.2966   0.1847   0.1965   0.1893   0.1330
## Balanced Accuracy      0.8648   0.7821   0.7799   0.8299   0.7915
```

The accuracy is 70% in sample.

Finally, make predictions on the test data.  Classe is not available, so out of sample performance is not known


```r
predictions = predict(modelFit, newdata=testing)
```
 
The data used was

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 
