
rm(list=ls())

##########################################################################################
# main function: Wave.PCA.Class
#   output "res": 4개 class의 평균 벡터들과 공분산 행렬들
# 
# 새로운 자료 x의 분류: 
#   N(meani, covi)의 pdf에 x를 넣어서 가장 큰 pdf를 가지는 class i로 x를 분류
##########################################################################################
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) # setting the workspace to source file location

source("UsedWavelet.R")

data1 = read.csv("data_250418/000_Et_H_CO_n.csv", header = FALSE)
data2 = read.csv("data_250418/002_Et_H_CO_H.csv", header = FALSE)
data3 = read.csv("data_250418/008_Et_H_CO_L.csv", header = FALSE)
data4 = read.csv("data_250418/028_Et_H_CO_M.csv", header = FALSE)

data1 = data1[,-(1:3)]
data1 = as.matrix(data1)
# data1 = scale(data1)
dim(data1)

data2 = data2[,-(1:3)]
data2 = as.matrix(data2)
# data2 = scale(data2)
dim(data2)

data3 = data3[,-(1:3)]
data3 = as.matrix(data3)
# data3 = scale(data3)
dim(data3)

data4 = data4[,-(1:3)]
data4 = as.matrix(data4)
# data4 = scale(data4)
dim(data4)



n = 100
err.sd = 30

data.arr1 = array(0, dim = c(nrow(data1), ncol(data1), n))
data.arr2 = array(0, dim = c(nrow(data2), ncol(data2), n))
data.arr3 = array(0, dim = c(nrow(data3), ncol(data3), n))
data.arr4 = array(0, dim = c(nrow(data4), ncol(data4), n))
for(i in 1:n){
  set.seed(i)
  data.arr1[,,i] = data1 + matrix(rnorm(nrow(data1)*ncol(data1), sd = err.sd), nrow(data1), ncol(data1))
  data.arr2[,,i] = data2 + matrix(rnorm(nrow(data2)*ncol(data2), sd = err.sd), nrow(data2), ncol(data2))
  data.arr3[,,i] = data3 + matrix(rnorm(nrow(data3)*ncol(data3), sd = err.sd), nrow(data3), ncol(data3))
  data.arr4[,,i] = data4 + matrix(rnorm(nrow(data4)*ncol(data4), sd = err.sd), nrow(data4), ncol(data4))
}

# data_original: 4개 가스의 모든 관측치를 담은 데이터 행렬
data_original = array(0, dim = c(nrow(data1), ncol(data1), 4*n))
data_original[,,1:n] = data.arr1
data_original[,,n+1:n] = data.arr2
data_original[,,2*n+1:n] = data.arr3
data_original[,,3*n+1:n] = data.arr4


Wave.PCA.Class <- function(data_original, n){
  
  n_samples = 4*n
  n_sensors = 8
  T_original = nrow(data1)
  T <- 4096 # 2의 거듭제곱으로 조정 (2^12)
  
  # 웨이블렛 계수 추출
  # 센서당 추출할 계수 개수 (임의로 설정, 필요 시 조정)
  library(wavethresh)
  
  # 데이터 길이 조정 (T_original -> T, 뒤에 0 추가)
  data <- array(0, dim = c(T, n_sensors, n_samples))
  for (i in 1:n_samples) {
    for (j in 1:n_sensors) {
      data[1:T_original, j, i] <- data_original[, j, i] # 원래 데이터
      data[(T_original + 1):T, j, i] <- 0 # 뒤에 0으로 패딩
    }
  }
  
  # 학습 레이블 (4가지 가스, 각 100개 샘플)
  labels <- factor(rep(1:4, each = n))
  
  # 1. 학습 데이터에 웨이블렛 계수 추출
  w_per_sensor <- T_original
  wavelet_features <- matrix(0, nrow = n_samples, ncol = n_sensors * w_per_sensor)
  
  for (i in 1:n_samples) {
    for (j in 1:n_sensors) {
      sensor_data <- data[, j, i]
      dwt_result <- wd(sensor_data, filter.number = 2, family = "DaubExPhase")
      # 모든 레벨의 계수를 추출
      coeffs <- numeric()
      for (level in 0:(dwt_result$nlevels - 1)) {
        level_coeffs <- accessD(dwt_result, level = level)
        coeffs <- c(coeffs, level_coeffs)
      }
      # 근사 계수도 추가 (최종 레벨의 근사 계수)
      approx_coeffs <- accessC(dwt_result, level = dwt_result$nlevels - 1)
      coeffs <- c(coeffs, approx_coeffs)
      # 상위 w_per_sensor 개 계수 선택 (절대값 기준)
      coeffs <- coeffs[order(abs(coeffs), decreasing = TRUE)][1:w_per_sensor]
      idx <- (j - 1) * w_per_sensor + 1
      wavelet_features[i, idx:(idx + w_per_sensor - 1)] <- coeffs
    }
  }
  
  # 2. 학습 데이터 정규화 (파라미터 저장) : 정규화 하지 않는 것이 분류 성능이 좋다.
  # wavelet_features_scaled <- scale(wavelet_features)
  # means <- attr(wavelet_features_scaled, "scaled:center")
  # sds <- attr(wavelet_features_scaled, "scaled:scale")
  wavelet_features_scaled = wavelet_features
  
  # 3. 학습 데이터에 PCA 적용 (변환 행렬 저장)
  pca_result <- prcomp(wavelet_features_scaled, scale. = FALSE)
  explained_variance <- summary(pca_result)$importance[3, ]
  n_pcs <- which(explained_variance >= 0.8)[1]
  cat("Number of PCs explaining >= 80% variance:", n_pcs, "\n")
  
  # PCA로 데이터 축약
  features_reduced <- pca_result$x[, 1:n_pcs]
  
  
  # 분류기 학습 (naive Bayes)
  mean1 = colMeans(features_reduced[1:n,])
  mean2 = colMeans(features_reduced[n+1:n,])
  mean3 = colMeans(features_reduced[2*n+1:n,])
  mean4 = colMeans(features_reduced[3*n+1:n,])
  cov1 = cov(features_reduced[1:n,]) + 0.0001*diag(n_pcs)
  cov2 = cov(features_reduced[n+1:n,]) + 0.0001*diag(n_pcs)
  cov3 = cov(features_reduced[2*n+1:n,]) + 0.0001*diag(n_pcs)
  cov4 = cov(features_reduced[3*n+1:n,]) + 0.0001*diag(n_pcs)
  
  
  # # 분류기에 사용한 자료의 분류 test
  # x = features_reduced[7+n*2,]
  # fit.values = c(as.numeric(- 0.5*log(det(cov1)) - 0.5*t(x-mean1) %*% solve(cov1) %*% as.matrix(x-mean1)),
  #                as.numeric(- 0.5*log(det(cov2)) - 0.5*t(x-mean2) %*% solve(cov2) %*% as.matrix(x-mean2)),
  #                as.numeric(- 0.5*log(det(cov3)) - 0.5*t(x-mean3) %*% solve(cov3) %*% as.matrix(x-mean3)),
  #                as.numeric(- 0.5*log(det(cov4)) - 0.5*t(x-mean4) %*% solve(cov4) %*% as.matrix(x-mean4)))
  # which.max(fit.values); exp(fit.values)/sum(exp(fit.values))
  
  return(list(mean1=mean1, mean2=mean2, mean3=mean3, mean4=mean4,
              cov1=cov1, cov2=cov2, cov3=cov3, cov4=cov4))
}




res = Wave.PCA.Class(data_original, n)

