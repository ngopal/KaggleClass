Creating a logistic regression model from the Kaggle data
========================================================

Modeling the approach detailed here: http://www.ats.ucla.edu/stat/r/dae/logit.htm

Dataset from here: http://www.kaggle.com/c/predict-who-is-more-influential-in-a-social-network



Let's load in the data:


```r
library(aod)
```

```
## Warning: package 'aod' was built under R version 3.0.2
```

```r
library(ggplot2)
```

```
## Warning: package 'ggplot2' was built under R version 3.0.2
```

```r
train <- read.csv("~/Code/kaggle_class_code/data/train.csv")
```

```
## Warning: cannot open file
## '/Users/nikhilgopal/Code/kaggle_class_code/data/train.csv': No such file
## or directory
```

```
## Error: cannot open the connection
```

```r
test <- read.csv("~/Code/kaggle_class_code/data/test.csv")
```

```
## Warning: cannot open file
## '/Users/nikhilgopal/Code/kaggle_class_code/data/test.csv': No such file or
## directory
```

```
## Error: cannot open the connection
```


Let's get a summary of our training set data:


```r
summary(train)
```

```
## Error: error in evaluating the argument 'object' in selecting a method for
## function 'summary': Error: object 'train' not found
```


There seem to be an even amount of influencers and non-influencers in this data. This may suggest that the data has been curated to be proportional. The max values are really high for some of the variables, suggesting that either people have too much time to spend on Twitter, or that they've automated their Twitter activities.


```r
sapply(train, sd)
```

```
## Error: object 'train' not found
```


Of course, the standard deviation is pretty high for all variables. We definitely have outliers in this dataset. So maybe an algorithm more sensitive to outliers is best suited for a model based on this data.

Let's make a contingency table:


```r
xtabs(~Choice + A_network_feature_1, data = train)
```

```
## Error: object 'train' not found
```


After playing with some formulas, my initial poking around makes me think the unnamed variables have the biggest impact on this data. Unfortunately, I don't know what those are...

But anyhow, let's make a logistic regression model (and generate a baseline for us to compare our future models to). Everything in our dataset seems to be numeric and continuous:


```r
mylogit <- glm(Choice ~ A_follower_count + A_following_count + A_listed_count + 
    A_mentions_received + A_retweets_received + A_mentions_sent + A_retweets_sent + 
    A_posts + A_network_feature_1 + A_network_feature_2 + A_network_feature_3 + 
    B_follower_count + B_following_count + B_listed_count + B_mentions_received + 
    B_retweets_received + B_mentions_sent + B_retweets_sent + B_posts + B_network_feature_1 + 
    B_network_feature_2 + B_network_feature_3, data = train, family = "binomial")
```

```
## Error: object 'train' not found
```

```r

summary(mylogit)
```

```
## Error: error in evaluating the argument 'object' in selecting a method for
## function 'summary': Error: object 'mylogit' not found
```


Off of the bat, some variables seem more important than others (in terms of weightage in the model). Specifically, A_listed_count, A_mentions_sent, B_retweets_received, etc. The magnitude of the data seems to differ among variables--we should probably use log to make things less turbulent. However, none of this means anything yet...

Let's make some confidence intervals:

```r
# confint(mylogit)
confint.default(mylogit)  #this will make CIs using standard error
```

```
## Error: object 'mylogit' not found
```


Can we figure out the overall effect of rank using a wald test?:


```r
wald.test(b = coef(mylogit), Sigma = vcov(mylogit), Terms = 1:23)
```

```
## Error: error in evaluating the argument 'object' in selecting a method for
## function 'coef': Error: object 'mylogit' not found
```


Wow. Pretty significant. Maybe I'm doing this wrong. But let's power through...

Let's get some odds ratios (and their 95% CIs):

```r
exp(cbind(OR = coef(mylogit), confint(mylogit)))
```

```
## Error: error in evaluating the argument 'object' in selecting a method for
## function 'coef': Error: object 'mylogit' not found
```


All of the odds are around 1. For A_follower_count, a 1 unit increase will decrease the odds of being an influencer by a factor of 0.9999~

Let's see how our training and test set compare:

```r
plot(mylogit)  # The model. The QQ plot is terrible--seems to predict influencers aren't as effective when they actually are and are effctive when they actually aren't
```

```
## Error: object 'mylogit' not found
```

```r

predresults <- predict(mylogit, newdata = test, type = "response")
```

```
## Error: error in evaluating the argument 'object' in selecting a method for
## function 'predict': Error: object 'mylogit' not found
```



Let's look at the model and prediction of training set results side-by-side:


```r

sidebyside <- cbind(train$Choice, predict(mylogit, newdata = train, type = "response"), 
    round(predict(mylogit, newdata = train, type = "response")))
```

```
## Error: object 'train' not found
```

```r

sum(sidebyside[, 1] == sidebyside[, 3])  # number of correct predictions
```

```
## Error: object 'sidebyside' not found
```

```r
dim(sidebyside)[1]  # total number of training points
```

```
## Error: object 'sidebyside' not found
```

```r
sum(sidebyside[, 1] == sidebyside[, 3])/dim(sidebyside)[1] * 100  # percent correctly classified
```

```
## Error: object 'sidebyside' not found
```


