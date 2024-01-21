#Installing Library
#install.packages("randomForest") 
#install.packages("caTools")  
#install.packages('e1071') 
#install.packages("ISLR")
#install.packages("DataExplorer")

library("caTools")
library(randomForest) 
library(dplyr)
library(caret)
library(DataExplorer)
library(tidyr)


library(e1071) 
library(ISLR)

options(warn=-1)

# Read Input File
df <- read.csv("D:/loanCredit1.csv")
print("Number of Records Before Data Cleaning")
print(dim(df))

#Cleaning Data
#Replacing NA for missing Values
df[df == ''] <- NA
#Remove rows with NA's using na.omit()
df <- na.omit(df)
print("Number of Records  Data AfterCleaning")
print(dim(df))

#Exploratory Data Analysis (EDA)
#DESCRIPTIVE ANALYSIS
print("Summary Statistics")
print(summary(df))
#calculate summary statistics for each numeric variable in data frame
create_report(df,output_file = "D:/EDA.html",report_title = "Exploratory Data Analysis")


# Splitting data in train and test data 
#df=head(df,100000)
split <- sample.split(df, SplitRatio = 0.8) 
train <- subset(df, split == "TRUE") 
test <- subset(df, split == "FALSE") 

df$loan_status <- as.factor(df$loan_status)    
train$loan_status <- as.factor(train$loan_status)

#bestmtry <- tuneRF(train,train$loan_status,stepFactor = 1.2, improve = 0.01, trace=T, plot= T) 

#model <- randomForest(loan_status~.,data= train)
rf <- randomForest(loan_status~loan_amount+term+grade+home_ownership+annual_income
                      +verification_status+int_rate+dti+application_type+tot_cur_bal, data=train)
print(rf) 

print(importance(rf))  
varImpPlot(rf)

#Make predictions on test data
pred_test <- predict(rf, newdata = test, type= "class")
print('The confusion matrix for test data is :')
table(pred_test, test$loan_status)
print(confusionMatrix(table(pred_test,test$loan_status)))


#Support Vector Machine (SVM)
print("Support Vector Machine")
classifier = svm(formula = loan_status ~ ., 
                 data = train, 
                 type = 'C-classification', 
                 kernel = 'linear') 

# Predicting the Test set results 
y_pred = predict(classifier, newdata = test) 

# Making the Confusion Matrix 
cm = table(test$loan_status, y_pred) 
print(confusionMatrix(table(y_pred,test$loan_status)))

#Logistic Regression
print("Logistic Regression")
# Training model
logistic_model <- glm(loan_status ~ ., data = train, family = "binomial")
logistic_model
# Summary
print(summary(logistic_model))

#use model to predict probability of default
predicted <- predict(logistic_model, test, type="response")

#convert defaults from "Yes" and "No" to 1's and 0's
#test$default <- ifelse(test$loan_status=="Yes", 1, 0)

#find optimal cutoff probability to use to maximize accuracy
#optimal <- optimalCutoff(test$loan_status, predicted)[1]

#create confusion matrix
#table(test$loan_status, predicted)
print('Confusion Matrix : ')
print(table(test$loan_status, predicted > 0.5))
#confusionMatrix(test$loan_status, predicted)