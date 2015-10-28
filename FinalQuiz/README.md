# Practical Machine Learning - Quiz 4
Calin Uioreanu  
October 28, 2015  

This is the 4th and final Quiz from the [Practical Machine Learning Coursera Course](https://class.coursera.org/predmachlearn-033/) by Jeff Leek, PhD, Roger D. Peng, PhD, Brian Caffo, PhD, part of the Data Science Specialization.



```
##          used (Mb) gc trigger (Mb) max used (Mb)
## Ncells 306953 16.4     592000 31.7   375530 20.1
## Vcells 503580  3.9    1023718  7.9   786384  6.0
```

#Question 1
Load the vowel.train and vowel.test data sets:

```r
library(ElemStatLearn)
data(vowel.train)
data(vowel.test) 
dim(vowel.train); dim(vowel.test)
```

```
## [1] 528  11
```

```
## [1] 462  11
```
Set the variable y to be a factor variable in both the training and test set. Then set the seed to 33833. Fit (1) a random forest predictor relating the factor variable y to the remaining variables and (2) a boosted predictor using the "gbm" method. Fit these both with the train() command in the caret package. 


```r
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)
# fit a random forest predictor using caret's unified model building interface
vowel.fit.rf <- train(y ~ .,
                      data = vowel.train,
                      method="rf")
set.seed(33833)
# fit a gradient boosting machine model, using verbose to limit the output
vowel.fit.gbm <- train(y ~ .,
                      data = vowel.train,
                      method="gbm", 
                      verbose = F)
```

## Question: What are the accuracies for the two approaches on the test data set? 

We'll look at the Out of Sample error for both random forest and grandient boosting machine models.


```r
# Random Forest: 100% accuracy on the training set
confusionMatrix(predict(vowel.fit.rf), vowel.train$y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  1  2  3  4  5  6  7  8  9 10 11
##         1  48  0  0  0  0  0  0  0  0  0  0
##         2   0 48  0  0  0  0  0  0  0  0  0
##         3   0  0 48  0  0  0  0  0  0  0  0
##         4   0  0  0 48  0  0  0  0  0  0  0
##         5   0  0  0  0 48  0  0  0  0  0  0
##         6   0  0  0  0  0 48  0  0  0  0  0
##         7   0  0  0  0  0  0 48  0  0  0  0
##         8   0  0  0  0  0  0  0 48  0  0  0
##         9   0  0  0  0  0  0  0  0 48  0  0
##         10  0  0  0  0  0  0  0  0  0 48  0
##         11  0  0  0  0  0  0  0  0  0  0 48
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.993, 1)
##     No Information Rate : 0.0909    
##     P-Value [Acc > NIR] : < 2.2e-16 
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
## Sensitivity           1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Specificity           1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Pos Pred Value        1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Neg Pred Value        1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Prevalence            0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Detection Rate        0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Detection Prevalence  0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Balanced Accuracy     1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
## Sensitivity           1.00000  1.00000  1.00000   1.00000   1.00000
## Specificity           1.00000  1.00000  1.00000   1.00000   1.00000
## Pos Pred Value        1.00000  1.00000  1.00000   1.00000   1.00000
## Neg Pred Value        1.00000  1.00000  1.00000   1.00000   1.00000
## Prevalence            0.09091  0.09091  0.09091   0.09091   0.09091
## Detection Rate        0.09091  0.09091  0.09091   0.09091   0.09091
## Detection Prevalence  0.09091  0.09091  0.09091   0.09091   0.09091
## Balanced Accuracy     1.00000  1.00000  1.00000   1.00000   1.00000
```

```r
# but only 60.61% accuracy on the testing set
confusionMatrix(predict(vowel.fit.rf, vowel.test), vowel.test$y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  1  2  3  4  5  6  7  8  9 10 11
##         1  34  3  0  0  0  0  0  0  0  1  0
##         2   7 21  3  0  0  0  0  0  1 13  1
##         3   1 14 32  3  0  2  0  0  0  5  2
##         4   0  0  2 29  3  0  0  0  0  0  2
##         5   0  0  0  0 20  8  8  0  0  0  0
##         6   0  0  4  9 15 22  4  0  0  0  6
##         7   0  0  0  0  3  0 28  6  6  0  3
##         8   0  0  0  0  0  0  0 32  6  0  0
##         9   0  4  0  0  0  0  1  4 22  1 12
##         10  0  0  0  0  0  0  1  0  2 22  0
##         11  0  0  1  1  1 10  0  0  5  0 16
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6017          
##                  95% CI : (0.5555, 0.6467)
##     No Information Rate : 0.0909          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5619          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
## Sensitivity           0.80952  0.50000  0.76190  0.69048  0.47619  0.52381
## Specificity           0.99048  0.94048  0.93571  0.98333  0.96190  0.90952
## Pos Pred Value        0.89474  0.45652  0.54237  0.80556  0.55556  0.36667
## Neg Pred Value        0.98113  0.94952  0.97519  0.96948  0.94836  0.95025
## Prevalence            0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Detection Rate        0.07359  0.04545  0.06926  0.06277  0.04329  0.04762
## Detection Prevalence  0.08225  0.09957  0.12771  0.07792  0.07792  0.12987
## Balanced Accuracy     0.90000  0.72024  0.84881  0.83690  0.71905  0.71667
##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
## Sensitivity           0.66667  0.76190  0.52381   0.52381   0.38095
## Specificity           0.95714  0.98571  0.94762   0.99286   0.95714
## Pos Pred Value        0.60870  0.84211  0.50000   0.88000   0.47059
## Neg Pred Value        0.96635  0.97642  0.95215   0.95423   0.93925
## Prevalence            0.09091  0.09091  0.09091   0.09091   0.09091
## Detection Rate        0.06061  0.06926  0.04762   0.04762   0.03463
## Detection Prevalence  0.09957  0.08225  0.09524   0.05411   0.07359
## Balanced Accuracy     0.81190  0.87381  0.73571   0.75833   0.66905
```

```r
# GBM now: 100% accuracy on training
confusionMatrix(predict(vowel.fit.gbm), vowel.train$y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  1  2  3  4  5  6  7  8  9 10 11
##         1  48  0  0  0  0  0  0  0  0  0  0
##         2   0 48  0  0  0  0  0  0  0  0  0
##         3   0  0 48  0  0  0  0  0  0  0  0
##         4   0  0  0 48  0  0  0  0  0  0  0
##         5   0  0  0  0 48  0  0  0  0  0  0
##         6   0  0  0  0  0 48  0  0  0  0  0
##         7   0  0  0  0  0  0 48  0  0  0  0
##         8   0  0  0  0  0  0  0 48  0  0  0
##         9   0  0  0  0  0  0  0  0 48  0  0
##         10  0  0  0  0  0  0  0  0  0 48  0
##         11  0  0  0  0  0  0  0  0  0  0 48
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.993, 1)
##     No Information Rate : 0.0909    
##     P-Value [Acc > NIR] : < 2.2e-16 
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
## Sensitivity           1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Specificity           1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Pos Pred Value        1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Neg Pred Value        1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
## Prevalence            0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Detection Rate        0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Detection Prevalence  0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Balanced Accuracy     1.00000  1.00000  1.00000  1.00000  1.00000  1.00000
##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
## Sensitivity           1.00000  1.00000  1.00000   1.00000   1.00000
## Specificity           1.00000  1.00000  1.00000   1.00000   1.00000
## Pos Pred Value        1.00000  1.00000  1.00000   1.00000   1.00000
## Neg Pred Value        1.00000  1.00000  1.00000   1.00000   1.00000
## Prevalence            0.09091  0.09091  0.09091   0.09091   0.09091
## Detection Rate        0.09091  0.09091  0.09091   0.09091   0.09091
## Detection Prevalence  0.09091  0.09091  0.09091   0.09091   0.09091
## Balanced Accuracy     1.00000  1.00000  1.00000   1.00000   1.00000
```

```r
# Accuracy: 0.526
confusionMatrix(predict(vowel.fit.gbm, vowel.test), vowel.test$y)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  1  2  3  4  5  6  7  8  9 10 11
##         1  29  1  0  0  0  0  0  0  0  2  0
##         2   9 13  1  0  0  0  1  0  0 15  0
##         3   2 13 12  3  0  0  0  0  0  0  1
##         4   0  1  8 22  3  0  0  0  0  0  0
##         5   0  0  0  4 19  4  0  0  0  0  0
##         6   0  2  8 13 10 30  0  0  0  0  8
##         7   0  0  6  0  6  1 39  9  4  0 13
##         8   0  0  0  0  0  0  2 27 10  0  0
##         9   0  9  0  0  0  0  0  6 28  5 16
##         10  2  0  0  0  0  0  0  0  0 20  0
##         11  0  3  7  0  4  7  0  0  0  0  4
## 
## Overall Statistics
##                                           
##                Accuracy : 0.526           
##                  95% CI : (0.4793, 0.5723)
##     No Information Rate : 0.0909          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4786          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
## Sensitivity           0.69048  0.30952  0.28571  0.52381  0.45238  0.71429
## Specificity           0.99286  0.93810  0.95476  0.97143  0.98095  0.90238
## Pos Pred Value        0.90625  0.33333  0.38710  0.64706  0.70370  0.42254
## Neg Pred Value        0.96977  0.93144  0.93039  0.95327  0.94713  0.96931
## Prevalence            0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
## Detection Rate        0.06277  0.02814  0.02597  0.04762  0.04113  0.06494
## Detection Prevalence  0.06926  0.08442  0.06710  0.07359  0.05844  0.15368
## Balanced Accuracy     0.84167  0.62381  0.62024  0.74762  0.71667  0.80833
##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
## Sensitivity           0.92857  0.64286  0.66667   0.47619  0.095238
## Specificity           0.90714  0.97143  0.91429   0.99524  0.950000
## Pos Pred Value        0.50000  0.69231  0.43750   0.90909  0.160000
## Neg Pred Value        0.99219  0.96454  0.96482   0.95000  0.913043
## Prevalence            0.09091  0.09091  0.09091   0.09091  0.090909
## Detection Rate        0.08442  0.05844  0.06061   0.04329  0.008658
## Detection Prevalence  0.16883  0.08442  0.13853   0.04762  0.054113
## Balanced Accuracy     0.91786  0.80714  0.79048   0.73571  0.522619
```
therefore both are **largely overfitted model**.

## What is the accuracy among the test set samples where the two methods agree?

```r
confusionMatrix(
      predict(vowel.fit.rf, vowel.test), 
      predict(vowel.fit.gbm, vowel.test)
)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction  1  2  3  4  5  6  7  8  9 10 11
##         1  27  6  2  0  0  0  0  0  0  3  0
##         2   3 30  5  0  0  1  0  0  5  0  2
##         3   2  0 23  8  0  6  8  0  1  3  7
##         4   0  0  1 26  1  6  0  0  1  0  1
##         5   0  0  0  0 17  4 13  1  0  0  1
##         6   0  0  0  0  7 46  4  0  0  0  2
##         7   0  1  0  0  2  0 39  2  1  0  0
##         8   0  0  0  0  0  0  3 33  2  0  0
##         9   0  0  0  0  0  0  3  2 41  0  0
##         10  0  2  0  0  0  0  1  1  5 16  0
##         11  0  0  0  0  0  8  7  0  8  0 12
## 
## Overall Statistics
##                                           
##                Accuracy : 0.671           
##                  95% CI : (0.6261, 0.7137)
##     No Information Rate : 0.1688          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6359          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
## Sensitivity           0.84375  0.76923  0.74194  0.76471  0.62963  0.64789
## Specificity           0.97442  0.96217  0.91879  0.97664  0.95632  0.96675
## Pos Pred Value        0.71053  0.65217  0.39655  0.72222  0.47222  0.77966
## Neg Pred Value        0.98821  0.97837  0.98020  0.98122  0.97653  0.93797
## Prevalence            0.06926  0.08442  0.06710  0.07359  0.05844  0.15368
## Detection Rate        0.05844  0.06494  0.04978  0.05628  0.03680  0.09957
## Detection Prevalence  0.08225  0.09957  0.12554  0.07792  0.07792  0.12771
## Balanced Accuracy     0.90908  0.86570  0.83036  0.87067  0.79298  0.80732
##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
## Sensitivity           0.50000  0.84615  0.64062   0.72727   0.48000
## Specificity           0.98438  0.98818  0.98744   0.97955   0.94737
## Pos Pred Value        0.86667  0.86842  0.89130   0.64000   0.34286
## Neg Pred Value        0.90647  0.98585  0.94471   0.98627   0.96956
## Prevalence            0.16883  0.08442  0.13853   0.04762   0.05411
## Detection Rate        0.08442  0.07143  0.08874   0.03463   0.02597
## Detection Prevalence  0.09740  0.08225  0.09957   0.05411   0.07576
## Balanced Accuracy     0.74219  0.91717  0.81403   0.85341   0.71368
```
so the accuracy on the test set where both models converge is 66.88%. 



#Question 2

Load the Alzheimer's data using the following commands

```r
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
```
Set the seed to 62433 and predict diagnosis with all the other variables using a random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model. Stack the predictions together using random forests ("rf"). What is the resulting accuracy on the test set? Is it better or worse than each of the individual predictions?

```r
# rough data estimates
dim(training); dim(testing);
```

```
## [1] 251 131
```

```
## [1]  82 131
```

```r
set.seed(62433)
# fit a Random Forest model
alz.fit.rf = train(diagnosis ~ .,
                   data = training,
                   method = "rf")
set.seed(62433)
# fit a Gradient Boosting Machine model
alz.fit.gbm = train(diagnosis ~ .,
                   data = training,
                   method = "gbm", verbose = FALSE)
set.seed(62433)
# fit a Linear Discriminant Analysis model
alz.fit.lda = train(diagnosis ~ .,
                   data = training,
                   method = "lda")
# now we evaluate the accuracy for all 3 models individualy

# Random Forest
postResample(predict(alz.fit.rf, testing), testing$diagnosis)
```

```
##  Accuracy     Kappa 
## 0.7682927 0.3832146
```

```r
# Gradient Boosting Machine
postResample(predict(alz.fit.gbm, testing), testing$diagnosis)
```

```
##  Accuracy     Kappa 
## 0.7926829 0.4642583
```

```r
# Linear Discriminant Analysis
postResample(predict(alz.fit.lda, testing), testing$diagnosis)
```

```
##  Accuracy     Kappa 
## 0.7682927 0.4638679
```

# Stacking the 3 models using Random Forest
we fit a model that combines predictors

```r
# build a data frame with the combined predictions and the outcome variable
predDF = data.frame(
                  predict(alz.fit.rf, testing),
                  predict(alz.fit.gbm, testing),
                  predict(alz.fit.lda, testing),
                  diagnosis = testing$diagnosis)
dim(predDF); head(predDF)
```

```
## [1] 82  4
```

```
##   predict.alz.fit.rf..testing. predict.alz.fit.gbm..testing.
## 1                      Control                       Control
## 2                      Control                       Control
## 3                      Control                      Impaired
## 4                      Control                       Control
## 5                      Control                       Control
## 6                      Control                       Control
##   predict.alz.fit.lda..testing. diagnosis
## 1                       Control   Control
## 2                       Control   Control
## 3                      Impaired  Impaired
## 4                       Control   Control
## 5                       Control   Control
## 6                       Control  Impaired
```

```r
# training a stacking model using random forest
combModFit <- train(diagnosis ~ ., method = "rf", data=predDF)
```

```
## note: only 2 unique complexity parameters in default grid. Truncating the grid to 2 .
```

```r
# predicting on the combined data frame
combPred <- predict(combModFit, predDF)
# resolve overall agreement rate and Kappa
postResample(combPred, testing$diagnosis)
```

```
##  Accuracy     Kappa 
## 0.8048780 0.4560531
```
So despite the individual accuracy of Random Forest of 78%, Gradient Boosting Machine of 79% and Linear Discriminant Analysis of 77%, the accuracy of the stacked model is 80%, higher than each indidual model. Weighting could improve the accuracy of the stacked model even further.

# Question 3
Load the concrete data with the commands:

```r
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
```
Set the seed to 233 and fit a lasso model to predict Compressive Strength. Which variable is the last coefficient to be set to zero as the penalty increases? (Hint: it may be useful to look up ?plot.enet).

We investigate Regularization Regression (lasso, ridge) for the dataset "concrete".

```r
set.seed(233)
conc.fit.lasso = train(CompressiveStrength ~ .,
                       data = training,
                       method = "lasso")
plot.enet(conc.fit.lasso$finalModel,
          xvar="penalty", use.color=TRUE)
```

![](predmachlearn-033-quiz4_files/figure-html/Answer3-1.png) 

```r
# the chart doesn't say much, however this table reveals the last "zeroed" coefficient
conc.fit.lasso$finalModel$beta.pure
```

```
##        Cement BlastFurnaceSlag     FlyAsh      Water Superplasticizer
## 0  0.00000000       0.00000000 0.00000000  0.0000000        0.0000000
## 1  0.01802441       0.00000000 0.00000000  0.0000000        0.0000000
## 2  0.02729231       0.00000000 0.00000000  0.0000000        0.1559023
## 3  0.03847584       0.00000000 0.00000000  0.0000000        0.4343197
## 4  0.04599070       0.01006844 0.00000000  0.0000000        0.5510774
## 5  0.06499050       0.03893552 0.00000000 -0.1174308        0.5737881
## 6  0.06678431       0.04187677 0.00000000 -0.1401379        0.5739760
## 7  0.08502706       0.06390658 0.03312720 -0.1768114        0.4070855
## 8  0.10150634       0.08388323 0.06377012 -0.2190396        0.2504283
## 9  0.10301008       0.08567182 0.06624407 -0.2188092        0.2387808
## 10 0.11351343       0.09842089 0.08097549 -0.1794448        0.2567532
##    CoarseAggregate FineAggregate        Age
## 0       0.00000000   0.000000000 0.00000000
## 1       0.00000000   0.000000000 0.00000000
## 2       0.00000000   0.000000000 0.00000000
## 3       0.00000000   0.000000000 0.02765634
## 4       0.00000000   0.000000000 0.04042040
## 5       0.00000000   0.000000000 0.07968720
## 6       0.00000000  -0.003483276 0.08499933
## 7       0.00000000   0.000000000 0.09810130
## 8       0.00000000   0.000000000 0.11088807
## 9       0.00000000   0.001412076 0.11162794
## 10      0.01130084   0.014480343 0.11346049
## attr(,"scaled:scale")
## [1] 2831.2770 2416.1259 1767.8101  595.3663  168.3104 2133.9390 2234.6943
## [8] 1756.7394
```


#Question 4

Load the data on the number of visitors to the instructors blog from [here](https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv) 

Using the commands:


```r
library(lubridate)  # For year() function below
#dat = read.csv("~/Desktop/gaData.csv")
dat = read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)
```
Fit a model using the bats() function in the forecast package to the training time series. Then forecast this model for the remaining time points. For how many of the testing points is the true value within the 95% prediction interval bounds?

We will first build a BATS model (Exponential smoothing state space model with Box-Cox transformation, ARMA errors, Trend and Seasonal components)

```r
library(forecast)
# build a bats model based on the original time series
visits.exp.smoothing = bats(tstrain)
# build the forecast with the same range as the testing set (2012)
visits.forecast = forecast(visits.exp.smoothing, nrow(testing))
# plot the forecast
plot(visits.forecast)
```

![](predmachlearn-033-quiz4_files/figure-html/Answer4-1.png) 

```r
# extracting the 95% prediction boundaries
visits.forecast.lower95 = visits.forecast$lower[,2]
visits.forecast.upper95 = visits.forecast$upper[,2]

# see how many of the testing visit counts do actually match
table ( 
  (testing$visitsTumblr>visits.forecast.lower95) & 
  (testing$visitsTumblr<visits.forecast.upper95))
```

```
## 
## FALSE  TRUE 
##     9   226
```

```r
# and in percentages
226/nrow(testing)
```

```
## [1] 0.9617021
```


# Question 5

Load the concrete data with the commands:

```r
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]
```
Set the seed to 325 and fit a support vector machine using the e1071 package to predict Compressive Strength using the default settings. Predict on the testing set. What is the RMSE?


```r
#install.packages("e1071")
library("e1071")
set.seed(325)
conc.fit.svm = svm(CompressiveStrength ~ .,
                     data=training)
# comparing predictions to actual values
conc.pred.svm = predict(conc.fit.svm, newdata = testing)

# Root Mean Squared Error
error = conc.pred.svm - testing$CompressiveStrength
sqrt(mean(error^2))
```

```
## [1] 6.715009
```

```r
# plot the relationship between the forecasted svm values and the actual values, coloured by Age
plot(conc.pred.svm, testing$CompressiveStrength, 
              pch=20, cex=1, 
              col=testing$Age,
              main="Relationship between the svm forecast and actual values")
```

![](predmachlearn-033-quiz4_files/figure-html/Answer5-1.png) 


# EOF
