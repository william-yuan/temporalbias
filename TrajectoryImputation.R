library(rriskDistributions)
library(dplyr)
library(glmnet)
#take centile data from supplementary figures,  chinese sample

#get.exp.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.3,7.5,14.5,21.8,33.16,65.38))
#get.gamma.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.3,7.5,14.5,21.8,33.16,65.38))
#get.norm.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.3,7.5,14.5,21.8,33.16,65.38))
#get.tnorm.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.3,7.5,14.5,21.8,33.16,65.38))
#get.weibull.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.3,7.5,14.5,21.8,33.16,65.38))
control_distribution = get.lnorm.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.3,7.5,14.5,21.8,33.16,65.38)) 


#get.tnorm.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.4,8.2,16.7,25,38.8,74.25))
#get.weibull.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.4,8.2,16.7,25,38.8,74.25))
#get.exp.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.4,8.2,16.7,25,38.8,74.25))
case_distribution = get.lnorm.par(p=c(0.025,0.33,0.66,0.8,0.9,0.975),q=c(3.4,8.2,16.7,25,38.8,74.25)) 

#qualitatively, this distribution looks like the one in the supplementary figures
plot(seq(150),dlnorm(seq(150),meanlog = control_distribution[1], sdlog = control_distribution[2]) , type = 'l')
lines(seq(150),dlnorm(seq(150),meanlog = case_distribution[1], sdlog = case_distribution[2]) , type = 'l', col = 2)

#making cases/controls
cases = rlnorm(15000,meanlog =case_distribution[1], sdlog = case_distribution[2]) 
controls = rlnorm(15000,meanlog =control_distribution[1], sdlog = control_distribution[2]) 
median(cases)
median(controls)

#baseline
df = as.data.frame(matrix(nrow = 30000, ncol = 2))
df$V1[1:15000] <- cases
df$V1[15001:30000] <- controls
df$V2[1:15000] <- 1
df$V2[15001:30000] <- 0
df$V1 = log(df$V1)
lpa_model <- glm(df, formula = V2 ~ V1, family = 'binomial')
summary(lpa_model)

###define case starting points

##sample from control distribution, smaller starting value
controlPool = rlnorm(150000,meanlog =control_distribution[1], sdlog = control_distribution[2]) 
while (min(controlPool) > min(cases)){
  controlPool = rlnorm(150000,meanlog =control_distribution[1], sdlog = control_distribution[2]) 
}

randomSmallerControls <- function(x){
  miniPool = sample(controlPool[controlPool < x])
  miniPool = miniPool[1:7]
  return(max(miniPool[is.finite(miniPool)]))
}
randomStarting = lapply(cases, randomSmallerControls)
randomStarting <- unlist(randomStarting)
table(randomStarting < cases)


##same percentile 
caseRef = rlnorm(150000,meanlog =case_distribution[1], sdlog = case_distribution[2]) 
percentile <- ecdf(caseRef)
percentileCases <- unlist(lapply(cases, percentile))

control_percentile <- function(percentile){
  return(qlnorm(p = min(percentile), meanlog =control_distribution[1], sdlog = control_distribution[2]) )
}
percentileCasesStarting <- unlist(lapply(percentileCases, control_percentile))

table(percentileCasesStarting < cases)
percentileCasesStarting[percentileCasesStarting > cases] <- 0.9 * cases[percentileCasesStarting > cases]
table(percentileCasesStarting < cases)

##minusNpercent 
percentReduction <- function(val, percent){
  return(val/(1+percent))
}

minus15percentStarting <- unlist(lapply(cases, percentReduction, percent = 0.15))
minus10percentStarting <- unlist(lapply(cases, percentReduction, percent = 0.10))





### interpolation strategies/sampling
#step function, impulse in first 0.1%
LateStep_interpolation <- function(start, end){
  sm <- min(c(start,end))
  bg <- max(c(start,end))
  
  return(sample(c(rep(bg,999), sm),1))
}
minus15percentLateStep <- mapply(LateStep_interpolation,minus15percentStarting,cases)
percentileCasesLateStep <- mapply(LateStep_interpolation,percentileCasesStarting,cases)
randomLateStep <- mapply(LateStep_interpolation,randomStarting,cases)


#step function, impulse in first 10%
EarlyStep_interpolation <- function(start, end){
  sm <- min(c(start,end))
  bg <- max(c(start,end))
  return(sample(c(rep(bg,9), sm),1))
}
minus15percentEarlyStep <- mapply(EarlyStep_interpolation,minus15percentStarting,cases)
percentileCasesEarlyStep <- mapply(EarlyStep_interpolation,percentileCasesStarting,cases)
randomEarlyStep <- mapply(EarlyStep_interpolation,randomStarting,cases)

#step function, impulse in first 1%
MidStep_interpolation <- function(start, end){
  sm <- min(c(start,end))
  bg <- max(c(start,end))
  return(sample(c(rep(bg,99), sm),1))
}
minus15percentMidStep <- mapply(MidStep_interpolation,minus15percentStarting,cases)
percentileCasesMidStep <- mapply(MidStep_interpolation,percentileCasesStarting,cases)
randomMidStep <- mapply(MidStep_interpolation,randomStarting,cases)


##linear
linear_interpolation <- function(start, end){
  sm <- min(c(start,end))
  bg <- max(c(start,end))
  return(runif(1, sm, bg))
}

minus15percentLinear <- mapply(linear_interpolation,minus15percentStarting,cases)
percentileCasesLinear <- mapply(linear_interpolation,percentileCasesStarting,cases)
randomLinear <- mapply(linear_interpolation,randomStarting,cases)
minus10percentLinear <- mapply(linear_interpolation,minus10percentStarting,cases)

table(minus15percentLinear < cases)
table(minus15percentLinear < minus15percentStarting)
table(percentileCasesLinear < cases)
table(percentileCasesLinear < percentileCasesStarting)
table(randomLinear < cases)
table(randomLinear < randomStarting)


##logistic
logisticf <- function(bg, sm, x, x0){
  return(bg/(1+exp(-1*x-x0)) + sm)
}


logistic_interpolation <- function(start, end){
  sm <- min(c(start,end))
  bg <- max(c(start,end))
  x = seq(-3,3,0.1)
  x0 = runif(1, -1.5, 1.5)
  logseq <- unlist(lapply(x, logisticf, bg = (bg-sm), sm = sm, x0 = x0))
  return(sample(logseq)[1])
}

minus15percentLogistic <- mapply(logistic_interpolation,minus15percentStarting,cases)
percentileCasesLogistic <- mapply(logistic_interpolation,percentileCasesStarting,cases)
randomLogistic <- mapply(logistic_interpolation,randomStarting,cases)
minus10percentLogistic <- mapply(logistic_interpolation,minus10percentStarting,cases)


table(minus15percentLogistic < cases)
table(minus15percentLogistic < minus15percentStarting)
table(percentileCasesLogistic < cases)
table(percentileCasesLogistic < percentileCasesStarting)
table(randomLogistic < cases)
table(randomLogistic < randomStarting)


##log
log_interpolation <- function(start, end){
  sm <- min(c(start,end))
  bg <- max(c(start,end))
  x = seq(1,6.5,0.1)
  x1 = 1
  x2 = 7
  y1 = 0
  y2 = (bg-sm)
  a = (y1-y2)/log(x1/x2)
  b = exp((y2*log(x1)-y1*log(x2))/(y1-y2))
  logseq <- a * log(b * x)+sm
  return(sample(logseq)[1])
}
minus15percentLog <- mapply(log_interpolation,minus15percentStarting,cases)
percentileCasesLog <- mapply(log_interpolation,percentileCasesStarting,cases)
randomLog <- mapply(log_interpolation,randomStarting,cases)
minus10percentLog <- mapply(log_interpolation,minus10percentStarting,cases)


table(minus15percentLog < cases)
table(minus15percentLog < minus15percentStarting)
table(percentileCasesLog < cases)
table(percentileCasesLog < percentileCasesStarting)
table(randomLog < cases)
table(randomLog < randomStarting)

#########
#evaluation

oddsratio <- function(cases){
  df = as.data.frame(matrix(nrow = 30000, ncol = 2))
  df$V1[1:15000] <- cases
  df$V1[15001:30000] <- controls
  df$V2[1:15000] <- 1
  df$V2[15001:30000] <- 0
  df$V1 = log(df$V1)
  lpa_model <- glm(df, formula = V2 ~ V1, family = 'binomial')
  return(c(deparse(substitute(cases)),lpa_model$coefficients[2],   coef(summary(lpa_model))[,4][2]))
}

result_df = rbind(c("baseline", lpa_model$coefficients[2],   coef(summary(lpa_model))[,4][2]),
                  oddsratio(randomLinear),
                  oddsratio(randomLogistic),
                  oddsratio(randomLog),
                  
                  oddsratio(percentileCasesLinear),
                  oddsratio(percentileCasesLogistic),
                  oddsratio(percentileCasesLog),
                  
                  oddsratio(minus15percentLinear),
                  oddsratio(minus15percentLogistic),
                  oddsratio(minus15percentLog))

colnames(result_df) <- c('name','val','pval')
result_df <- as.data.frame(result_df)
result_df$pval <- as.numeric(as.character(result_df$pval))
result_df$val <- as.numeric(as.character(result_df$val))
result_df$ratio = result_df$val / result_df$val[1]




step_df = rbind(c("baseline", lpa_model$coefficients[2],   coef(summary(lpa_model))[,4][2]),
                oddsratio(randomEarlyStep),
                oddsratio(randomMidStep),
                oddsratio(randomLateStep),
                
                oddsratio(percentileCasesEarlyStep),
                oddsratio(percentileCasesMidStep),
                oddsratio(percentileCasesLateStep),
                
                oddsratio(minus15percentEarlyStep),
                oddsratio(minus15percentMidStep),
                oddsratio(minus15percentLateStep))

colnames(step_df) <- c('name','val','pval')
step_df <- as.data.frame(step_df)
step_df$pval <- as.numeric(as.character(step_df$pval))
step_df$val <- as.numeric(as.character(step_df$val))
step_df$ratio = step_df$val / step_df$val[1]