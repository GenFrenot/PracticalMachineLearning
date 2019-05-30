Executive Summary
-----------------

This human activity recognition exercise is based on data from
<Groupware@LES>:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>

In this project, the goal is to use data from accelerometers on the
belt, forearm, arm, and dumbell of 6 participants which were asked to
perform barbell lifts correctly and incorrectly in 5 different ways.

Our strategy is to split the provided training data in two sets, one for
learning and one for testing our model. Once reaching a satisfying
accuracy, we would actually validate our model (and hopefully our
project assignement) by predicting the exercise classe on the provided
test data.

### Data Preparation

Loading the provided training data

    data<-read.csv( # read directly from provided URL
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
        ,header = TRUE
        ,na.strings = c("NA","NaN","","#DIV/0!")) # mark as NA all invalid entries
    dim(data)

    ## [1] 19622   160

In the appendix you can see some basic data exploration on this provided
training set. These 19622 records are a great amount of data sample that
we will be able to split for cross validation.

We are now loading the provided test data which we will use as a the
final validation set for the project

    validation<-read.csv(
        "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
        , header = TRUE, na.strings = c("NA","NaN","","#DIV/0!"))
    dim(validation)

    ## [1]  20 160

In the appendix you can see some basic data exploration on this provided
testing set. These 20 records are the measures for 20 exercises for
which we are going to predict the respective classes: A, B, C, D or E

Splitting the provided training data into a training and testing data
set for cross validation

    inTrain <- createDataPartition(y=data$classe, p=0.75, list=FALSE)
    training<- data[inTrain,]
    testing <- data[-inTrain,]

Creating simplified data set for the prediction model

    # removes the outcome 
    trainingPredictors<-training[,which(names(training)!="classe")] 
    # Remove columns 1 to 8 which are annotations
    trainingPredictors<-trainingPredictors[,-(1:8)] 
    # Look at the quantity of NA values in the predictors
    ratio.na<-function(v) { round(100*sum(is.na(v))/nrow(trainingPredictors)) }
    NAs <- sapply(trainingPredictors,ratio.na)
    table(NAs)

    ## NAs
    ##   0  98 100 
    ##  51  94   6

We can see that 51 predictor columns points are fully populated, and
that the others are 98% or 100% empty. As such variables could biaise
our analysis, we are removing these colunms from the predictors.

    trainingPredictors <- trainingPredictors[,names(NAs[NAs<98])] 

### Predictor Analysis

Now let's look at the correlations between the variables of this cleaned
set of predictors

    ggcorr(cor(trainingPredictors), size=3, hjust = 1.1, layout.exp = 8
           ,name="Correlation") 

![](HAR_Prediction8_files/figure-markdown_strict/correlations-1.png)

*Correlation of all valid predictors for the Human Activity test data*

We can see that a many variables are correlated and therefore decide to
preprocess the data with Principal Component Analysis

    preProc <- preProcess(trainingPredictors,method="pca",pcaComp=ncol(trainingPredictors))
    trainingPC <- predict(preProc,trainingPredictors)

Just because we are curious, we are visualizing the correlations of this
new set of pre-processed predictors.

    ggcorr(cor(trainingPC), size=3, hjust = 1.1, layout.exp = 8
            ,name="Correlation") 

![](HAR_Prediction8_files/figure-markdown_strict/preprocessed%20data%20correlations-1.png)
*Correlation of the pre-processed predictors for the Human Activity test
data*

We can see that these new predictors are nicely independant the ones
from the others.

### Building our Prediction Model

With our the pre-processed predictors, we are now ready to build our
prediction model. We choose to use Random Forest bootstraping, which
takes some time to process but is recognized for its accuracy

    modelFit <- train(x = trainingPC, y = training$classe,method="rf") 
    modelFit$results[modelFit$finalModel$mtry,] # result of the final model

    ##   mtry  Accuracy     Kappa  AccuracySD     KappaSD
    ## 2   26 0.9672361 0.9585468 0.003323737 0.004204737

The accuracy for this model is 96.7%, which is good to go. In the
appendix will find the full summary of this prediction model.

Let's look in-depth at the final model:

    modelFit$finalModel

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 2
    ## 
    ##         OOB estimate of  error rate: 2.03%
    ## Confusion matrix:
    ##      A    B    C    D    E class.error
    ## A 4175    6    2    1    1 0.002389486
    ## B   70 2757   20    0    1 0.031952247
    ## C    3   35 2517   11    1 0.019477990
    ## D    4    0   96 2301   11 0.046019900
    ## E    1    8   17   11 2669 0.013673319

We can see that the out of sample error is of 2.21%, which is
satisfactory as well.

### Cross Validation

We are now trying our prediction model on the data that we stored
separately as out test data

    testingPredictors<-testing[,which(names(testing) %in% names(trainingPredictors))]
    testingPC <- predict(preProc,testingPredictors)
    confusionMatrix<-confusionMatrix(testing$classe,predict(modelFit,testingPC))
    confusionMatrix$overall[1]

    ## Accuracy 
    ## 0.983279

The Overal Prediction Accuracy is 98,4%, which demonstrated that our
prediction model is satisfactory. In the appendix will find the all
details on this confusion matrix.

See below a visualization of the prediction vs observations:

    qplot(testing$classe,predict(modelFit,testingPC)
          , colour=classe, data=testing, geom = "jitter"
          , xlab = "Observed Classe", ylab = "Predicted Classe")

![](HAR_Prediction8_files/figure-markdown_strict/visualize%20the%20predicted%20vs%20observed%20values-1.png)

*Predicted vs. observed in testing data*

As the cross validation demonstrated that the prediction model is
satisfactory, we now can use the validation data.

### Final prediction on the validation data

First pre-processing:

    validationPredictors<-validation[,which(names(validation) %in% names(trainingPredictors))]
    validationPC<-predict(preProc,validationPredictors)

Now predicting the outcomes based on the pre-processed data:

    prediction<-predict(modelFit,validationPC)
    as.data.frame(prediction)

    ##    prediction
    ## 1           B
    ## 2           A
    ## 3           B
    ## 4           A
    ## 5           A
    ## 6           E
    ## 7           D
    ## 8           B
    ## 9           A
    ## 10          A
    ## 11          B
    ## 12          C
    ## 13          B
    ## 14          A
    ## 15          E
    ## 16          E
    ## 17          A
    ## 18          B
    ## 19          B
    ## 20          B

The 20 values above are our prediction to this current human activity
recognition exercise. By submitting, we will get the final evaluation of
its accuracy.

Appendix
========

Basic data exploration on the provided training set
---------------------------------------------------

    dim(data)

    ## [1] 19622   160

    str(data[,1:12]) # look at the structure of the 10 first columns 

    ## 'data.frame':    19622 obs. of  12 variables:
    ##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ kurtosis_roll_belt  : num  NA NA NA NA NA NA NA NA NA NA ...

    str(data[,(ncol(data)-6):(ncol(data))]) # look at the 6 last columns

    ## 'data.frame':    19622 obs. of  7 variables:
    ##  $ accel_forearm_x : int  192 192 196 189 189 193 195 193 193 190 ...
    ##  $ accel_forearm_y : int  203 203 204 206 206 203 205 205 204 205 ...
    ##  $ accel_forearm_z : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
    ##  $ magnet_forearm_x: int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
    ##  $ magnet_forearm_y: num  654 661 658 658 655 660 659 660 653 656 ...
    ##  $ magnet_forearm_z: num  476 473 469 469 473 478 470 474 476 473 ...
    ##  $ classe          : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...

Basic data exploration on the provided testing set
--------------------------------------------------

    dim(validation)

    ## [1]  20 160

    str(data[,1:10]) # look at the structure of the 10 first columns 

    ## 'data.frame':    19622 obs. of  10 variables:
    ##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
    ##  $ raw_timestamp_part_2: int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
    ##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
    ##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...

    str(validation[,(ncol(data)-5):(ncol(data))]) # look at the 5 last columns

    ## 'data.frame':    20 obs. of  6 variables:
    ##  $ accel_forearm_y : int  267 297 271 406 -93 322 170 -331 204 98 ...
    ##  $ accel_forearm_z : int  -149 -118 -129 -39 172 -144 -175 -282 -217 -7 ...
    ##  $ magnet_forearm_x: int  -714 -237 -51 -233 375 -300 -678 -109 0 -403 ...
    ##  $ magnet_forearm_y: int  419 791 698 783 -787 800 284 -619 652 723 ...
    ##  $ magnet_forearm_z: int  617 873 783 521 91 884 585 -32 469 512 ...
    ##  $ problem_id      : int  1 2 3 4 5 6 7 8 9 10 ...

Prediction model summary
------------------------

    modelFit

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    51 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 14718, 14718, 14718, 14718, 14718, 14718, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9709545  0.9632422
    ##   26    0.9672361  0.9585468
    ##   51    0.9539367  0.9417220
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.

Confusion matrix
----------------

    confusionMatrix

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1393    0    0    1    1
    ##          B   21  925    3    0    0
    ##          C    1   10  841    3    0
    ##          D    1    1   31  766    5
    ##          E    0    0    4    0  897
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9833          
    ##                  95% CI : (0.9793, 0.9867)
    ##     No Information Rate : 0.2887          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9788          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9838   0.9882   0.9568   0.9948   0.9934
    ## Specificity            0.9994   0.9940   0.9965   0.9908   0.9990
    ## Pos Pred Value         0.9986   0.9747   0.9836   0.9527   0.9956
    ## Neg Pred Value         0.9934   0.9972   0.9906   0.9990   0.9985
    ## Prevalence             0.2887   0.1909   0.1792   0.1570   0.1841
    ## Detection Rate         0.2841   0.1886   0.1715   0.1562   0.1829
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9916   0.9911   0.9766   0.9928   0.9962
