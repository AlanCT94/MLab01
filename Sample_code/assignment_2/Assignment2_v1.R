#------------------------------------------------------------------------------#
#- Filename: Assignment2
#- Date: 2022-11-07
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#- Set libraries
#------------------------------------------------------------------------------#
library(tidyverse)
library(caret)
library(dplyr)
library(ggplot2)
#------------------------------------------------------------------------------#
#- 1) Read in data and divide into training, validation and test
#------------------------------------------------------------------------------#

file_in <- "C:/Users/kerstin/Documents/LiU/ML/Labs/Lab01/Data/parkinsons.csv"
data_in <- read_csv(file_in, col_names = TRUE)

spec(data_in)

#- Exclude variables subject and test time, total_UPDRS

df <- data_in %>% dplyr::select(-`subject#`,-test_time, -total_UPDRS)

#age, sex
#data <- dplyr::select(data, motor_UPDRS,`Jitter:RAP`,`Jitter:PPQ5`,`Jitter:DDP`) 
                      
#- Summarise data
summary(df)

#- Plot data
library(GGally)
set.seed(12345)
#select(starts_with("Petal")
d_plot <- sample_n(df,100)
# a <- d_plot %>% dplyr::select(starts_with("Jitter"))
#        
# p1 <- ggpairs(dplyr::select(d_plot, starts_with("Jitter")))
# p1
# p2 <- ggpairs(dplyr::select(d_plot, starts_with("Shimmer")))
# p2
# #sex
# p3 <- ggpairs(dplyr::select(d_plot, age, `Jitter(%)`, Shimmer, NHR:PPE))
# p3
cr <- cor(d_plot)
heatmap(cr,Colv=NA,Rowv=NA)
heatmap(cr)

#- Divide into training and test set
set.seed(12345)
n <- nrow(df)
id <- sample(1:n, floor(n*0.6)) 
train <- df[id,] 
test <- df[-id,] 

#- Scale data
#train_x <- dplyr::select(train, - motor_UPDRS)
#test_x <- dplyr::select(test, - motor_UPDRS)

#scaler <- preProcess(train_x)
#train_x_sc <- predict(scaler, train_x)
#test_x_sc <- predict(scaler, test_x)
#train_sc <- add_column(train_x_sc,motor_UPDRS=train$motor_UPDRS)
#test_sc <- add_column(test_x_sc,motor_UPDRS=test$motor_UPDRS)

scaler <- preProcess(train)
train_sc <- predict(scaler, train)
test_sc <- predict(scaler, test)

# - Remove later
# (c(min_mean=min(colMeans(train_sc_x)),max_mean=max(colMeans(train_sc_x))))
# 
# (c(min_sd=min(apply(train_sc_x,2,sd)),max_sd=max(apply(train_sc_x,2,sd))))

#------------------------------------------------------------------------------#
# 2) Compute a linear regression model from the training data, estimate training 
# and test MSE
#------------------------------------------------------------------------------#
#formula <- motor_UPDRS ~ .
formula <- motor_UPDRS ~ . -1
data = train_sc
  m1 <- lm(formula=formula, data=data)
  #m1 <- lm(formula=motor_UPDRS ~ . -1, data=train_sc)
  summary(m1)
  order_coef <- order(abs(coef(m1)), decreasing = TRUE)
  
  (coef(m1)[order_coef])
  #   scaled y:
  #`Shimmer:DDA`  `Shimmer:APQ3`    `Jitter:DDP`    `Jitter:RAP`
  #-45.896239143    45.544096908     1.568045209    -1.49817479
  
  #- However, standard errors very large, unreliable solution
#------------------------------------------------------------------------------#  
# 3) Implement 4 following functions by using basic R commands only (no 
# external packages):
  
# a) Loglikelihood function
# b) Ridge function
# c) RidgeOpt function
# d) DF function  
#------------------------------------------------------------------------------#

#- a) log likelihood function
Loglikelihood <- function(formula, data, theta, sigma) {
  n <- nrow(data)
  
  X <- stats::model.matrix(formula,data)
  y <- data[,all.vars(formula)[1]] 
  
  c1 <- (n/2)*(log(2*pi*sigma^2))
  c2 <- (1/(2*sigma^2))
  ll <- -c1 - c2*sum((t(theta)%*%t(X)-y)^2) 
  return(ll)
}

#- Calculate log likelihood
theta <- as.matrix(coef(m1))
sigma <- summary(m1)$sigma

ll <- Loglikelihood(formula=formula,data=train_sc,theta=theta, sigma=sigma)

print(ll) 
#- Compare with R function
(logLik(m1))

#- b) ridge function
# params: numeric vector that should contain theta for the first positions, 
# then sigma in the last position
Ridge <- function(params, data, lambda, formula) {

  theta <- params[-length(params)]
  sigma <- params[length(params)] 
  
  ll_lm <- -Loglikelihood(formula=formula, data=data, theta=theta, sigma=sigma)
  #ll_w_ridge <- ll_lm+lambda*sum(theta[-1]^2)
  ll_w_ridge <- ll_lm + lambda*sum(theta^2)
  return(ll_w_ridge)
}

#- c) ridge opt function
# params: numeric vector that should contain theta for the first positions, 
# then sigma in the last position
RidgeOpt <- function(params, fn, formula, data, lambda, method) {
  res <- optim(par=params, fn=fn, formula=formula, data=data, lambda=lambda,
               method=method) 
  return(res)
}

# #- Compare RidgeOpt function with solution obtained from lm.ridge
# method <- "BFGS"
# lambda <- 2
# fn <- Ridge
# theta_start <- numeric(dim(train_sc)[2]-1) 
# names(theta_start) <- names(dplyr::select(data,-motor_UPDRS))
# sigma_start <- 2 
# 
# params <- c(theta=theta_start, sigma=sigma_start)
# 
# res_optim <- RidgeOpt(params, fn, formula, data, lambda, method)
# 
# res_optim$par

#library(MASS)
#m1_ridge <- lm.ridge(formula=formula,data=data,lambda=lambda)
#(theta_ridge <- coef(m1_ridge))

#order_coef_ridge <- order(abs(coef(m1_ridge)), decreasing = TRUE)
#(coef(m1_ridge)[order_coef_ridge])

#- d) ridge opt function

DF <- function(formula, data, lambda) {
  
  X <- stats::model.matrix(formula,data)
  y <- data[,all.vars(formula)[1]] 
  p <- dim(X)[2]
  
  #B_ridge = inv(XTX+lambdaI)XTy
  #y_hat = X%*%B_ridge = X*inv(XTX+lambdaI)XTy =>
  #y_hat = S(X)*Y, where S(X) = X*inv(XTX+lambdaI)XT
  S <- X%*%solve(t(X)%*%X+lambda*diag(p))%*%t(X)
  
  S_trace <- sum(diag(S))
  
  return(S_trace)
}

# #- Test function
# lambda <- 2
# (trace <- DF(formula=formula, data=data, lambda=lambda))

#------------------------------------------------------------------------------#  
# 4) Compute optimal theta parameters för lambda=1, 10 and 1000 respectively.
# Use the estimated parameters to predict the motor_UPDRS values for training 
# and test data and report the training and test MSE values.
#------------------------------------------------------------------------------#  
method <- "BFGS"
fn <- Ridge
theta_start <- numeric(dim(train_sc)[2]-1) 
names(theta_start) <- names(dplyr::select(data,-motor_UPDRS))
sigma_start <- 2 

params <- c(theta=theta_start, sigma=sigma_start)

#- lambda = 1
lambda <- 1
res_optim_l_1 <- RidgeOpt(params, fn, formula, data, lambda, method)

res_optim_l_1$par

#- lambda = 100
lambda <- 100
res_optim_l_100 <- RidgeOpt(params, fn, formula, data, lambda, method)

res_optim_l_100$par

#- lambda = 1000
lambda <- 1000
res_optim_l_1000 <- RidgeOpt(params, fn, formula, data, lambda, method)

res_optim_l_1000$par

#- Predict
train_sc_X <- as.matrix(select(train_sc,- motor_UPDRS))
test_sc_X <- as.matrix(select(test_sc,- motor_UPDRS))

#- Lambda = 1 error calc
theta_l_1 <- as.matrix(res_optim_l_1$par[-length(res_optim_l_1$par)])

y_pred_train_l_1 <- t(theta_l_1)%*%t(train_sc_X)
error_train_l_1 <- train_sc$motor_UPDRS-y_pred_train_l_1
y_pred_test_l_1 <- t(theta_l_1)%*%t(test_sc_X)
error_test_l_1 <- test_sc$motor_UPDRS-y_pred_test_l_1

#- Lambda = 100 error calc
theta_l_100 <- as.matrix(res_optim_l_100$par[-length(res_optim_l_100$par)])

y_pred_train_l_100 <- t(theta_l_100)%*%t(train_sc_X)
error_train_l_100 <- train_sc$motor_UPDRS-y_pred_train_l_100
y_pred_test_l_100 <- t(theta_l_100)%*%t(test_sc_X)
error_test_l_100 <- test_sc$motor_UPDRS-y_pred_test_l_100

#- Lambda = 1000 error calc
theta_l_1000 <- as.matrix(res_optim_l_1000$par[-length(res_optim_l_1000$par)])

y_pred_train_l_1000 <- t(theta_l_1000)%*%t(train_sc_X)
error_train_l_1000 <- train_sc$motor_UPDRS-y_pred_train_l_1000

y_pred_test_l_1000 <- t(theta_l_1000)%*%t(test_sc_X)
error_test_l_1000 <- test_sc$motor_UPDRS-y_pred_test_l_1000

#- RMSE calc
# Lambda = 1
(rmse_train_l_1 <- sqrt(mean(error_train_l_1^2)))
(rmse_test_l_1 <- sqrt(mean(error_test_l_1^2)))
# Lambda = 100
(rmse_train_l_100 <- sqrt(mean(error_train_l_100^2)))
(rmse_test_l_100 <- sqrt(mean(error_test_l_100^2)))
# Lambda = 1000
(rmse_train_l_1000 <- sqrt(mean(error_train_l_1000^2)))
(rmse_test_l_1000 <- sqrt(mean(error_test_l_1000^2)))

#=> lambda 100 has lowest error for the test set

#- Calculate DF for all lambda
lambda <- 1
(trace_l_1 <- DF(formula=formula, data=data, lambda=lambda))

lambda <- 100
(trace_l_100 <- DF(formula=formula, data=data, lambda=lambda))

lambda <- 1000
(trace_l_1000 <- DF(formula=formula, data=data, lambda=lambda))

#=> decreasing DF with increasing lambda

# #- Test with lambda = 0
# 
# lambda <- 0
# res_optim_l_0 <- RidgeOpt(params, fn, formula, data, lambda, method)
# 
# res_optim_l_0$par
# 
# theta_l_0 <- as.matrix(res_optim_l_0$par[-length(res_optim_l_0$par)])
# 
# y_pred_train_l_0 <- t(theta_l_0)%*%t(train_sc_X)
# error_train_l_0 <- train_sc$motor_UPDRS-y_pred_train_l_0
# y_pred_test_l_0 <- t(theta_l_0)%*%t(test_sc_X)
# error_test_l_0 <- test_sc$motor_UPDRS-y_pred_test_l_0
# 
# (rmse_train_l_0 <- sqrt(mean(error_train_l_0^2)))
# (rmse_test_l_0 <- sqrt(mean(error_test_l_0^2)))
# 
# sigma(m1)
