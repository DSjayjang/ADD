rm(list=ls())
library(MASS)
###################################################
#GAS data : 학습에 필요한 초기값이 충분치 않아 임의로 초기값을 부여하는 과정을 거침
###################################################
gas.data.raw = read.csv("/Users/seul/Desktop/2024/etc/sensor_example/GAS.csv", header =TRUE, row.names = 1)
getwd()
dim(gas.data.raw)
head(gas.data.raw)
##NA인 마지막 열 제외, colname이 포함된 첫번째 행 제외
gas.data1 =gas.data.raw[,-81]
gas.data=gas.data1[-1,]
dim(gas.data)
head(gas.data)

n.noise = 100
# 원본 데이터의 첫 번째 행을 기준으로 노이즈 생성
set.seed(123)
first.row = as.numeric(gas.data[1, ])

#목표 : 1번째 행 기준으로, 그 앞에 100개의 노이즈 생성
noise.data = as.data.frame(sapply(first.row, function(x) rnorm(n.noise, mean = x, sd = 5)))

#timestamp 생성
start.time = as.POSIXct("2024-01-01 09:34:34")  # 1번 행의 날짜 -1
timestamps = sort(seq.POSIXt(from = start.time, by = "-1 sec", length.out = n.noise))

row.names(noise.data) = format(timestamps, "%H:%M:%S")

#bind 위한 컬럼명 통일
colnames(noise.data) = colnames(gas.data)
new.data = rbind(noise.data, gas.data)

head(new.data, n = 10)  # 처음 10개 행을 확인

# export
write.csv(new.data, "gas_v2.csv", row.names = TRUE)


plot(new_data[,15],main="GAS")



###################################################
#CO2 data
###################################################
CO2.data.raw = read.csv("/Users/seul/Desktop/2024/etc/sensor_example/CO2.csv", header =TRUE, row.names = 1)
head(CO2.data.raw)
plot(CO2.data.raw[,1], main="CO2")

