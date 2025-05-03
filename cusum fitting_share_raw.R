###########################################
# 데이터 불러오기 : gas
###########################################

gas.data = read.csv("/Users/seul/Desktop/2024/etc/sensor_example/GAS_v2.csv", header =TRUE, row.names = 1)

gas.data = gas.data+rnorm(346*80,0,sd=30) #noise 추가 : 241015
plot(gas.data[,1])


library(qcc)

###########################################
# cusum
###########################################
setwd("/Users/seul/Desktop/2024/etc/sensor_example")

for (i in 1:4) {
  
  phase1.data=gas.random[c(1:100),i]
  phase2.data = gas.random[c(101:nrow(gas.random)),i]

  phase2.cusum = cusum(phase1.data, newdata = phase2.data)
  
  start.point = min(c(phase2.cusum$violations$lower, phase2.cusum$violations$upper))

  png(paste0("gas_random2_column_", i, ".png"),width = 1200, height = 800,res=150)
  
  plot(gas.random[, i], type = "o", col = "black", xaxt = "n", xlab = "", ylab = "Value",
       main = paste("( sample", i, ")"), pch = 16)
  
  abline(v = 100, col = "black", lty = "dashed")
  
  abline(v = start.point, col = "red", lty = 1)
  
  legend("bottomright", legend = paste("First detection at", time.labels[start.point]), col = "red", lty = 2)
  time.index=seq(1,length(time.labels),by=10)
  axis(1, at = time.index, labels = time.labels[time.index], las = 2)

  dev.off()
}

###########################################
# 데이터 불러오기 : CO2
###########################################

CO2.data.raw = read.csv("/Users/seul/Desktop/2024/etc/sensor_example/CO2.csv", header =TRUE, row.names = 1)
dim(CO2.data.raw)
CO2.data1 =CO2.data.raw[,-81]
CO2.data=CO2.data1[-1,]
dim(CO2.data)

CO2.data=as.data.frame(lapply(CO2.data, function(x) as.numeric(as.character(x))))
CO2.data = CO2.data+rnorm(136*80,0,sd=10) #noise 추가 : 241015
co2.random = CO2.data[, sample(1:ncol(CO2.data), 4)]

time.labels = rownames(gas.data)

###########################################
# cusum
###########################################
setwd("/Users/seul/Desktop/2024/etc/sensor_example")

for (i in 1:4) {

  phase1.co2data=as.numeric(co2.random[c(1:38),i])
  phase2.co2data = as.numeric(co2.random[c(39:nrow(co2.random)),i])
  phase2.co2cusum = cusum(phase1.co2data, newdata=phase2.co2data, se.shift=0.5)

  start.point = min(c(phase2.co2cusum$violations$lower, phase2.co2cusum$violation$upper))
  
  png(paste0("co2_random2_column_", i, ".png"),width = 1200, height = 800,res=150)
  
  plot(as.numeric(co2.random[, i]), type = "o", col = "black", xaxt = "n", xlab = "", ylab = "Value",
       main = paste("( sample", i, ")"), pch = 16)
  abline(v = 38, col = "black", lty = "dashed")
  abline(v = start.point, col = "red", lty = 1)

  legend("topright", legend = paste("First detection at", time.labels[start.point]), col = "red", lty = 2)
  time.index=seq(1,length(time.labels),by=10)
  axis(1, at = time.index, labels = time.labels[time.index], las = 2)
  
  dev.off()
}

