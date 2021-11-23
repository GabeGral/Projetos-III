## Projetos 3 
## Rede Neural
## Previsão de Cancelamento de Reservas de Hotel
## Gabriel Gral, Guilherme Luz e Paulo Meira


## Importar packages
library(readr)
library(dplyr)
library(caret)
library(neuralnet)
library(boot)
library(plyr)
library(matrixStats)
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r') ## Beautiful Plot for NN

## Importar Dataset
data = as.data.frame(read_csv("hotel_booking_clean.csv"))
data$hotel <- ifelse(data$hotel == "Resort Hotel", 1, 0)
data$country <- ifelse(data$country == "PRT", 1, 0)

## Selecionar Variáveis Importantes
data0 <- select(data, c('is_canceled', 'hotel', 'lead_time', 'adults', 'children', 'stays_in_weekend_nights', 'stays_in_week_nights', 'country', 'adr'))

## Selecionar tamanho da amostra
n <- 10000
data0 <- sample_n(data0, n, replace = F)

## Random sampling
samplesize = 0.80 * nrow(data0)
set.seed(80)
index = sample( seq_len ( nrow ( data0 ) ), size = samplesize )

## Criar base de treino e de teste
datatrain = data0[ index, ]
datatest = data0[ -index, ]

## Normalizar os dados 
max = apply(data0, 2 , max)
min = apply(data0, 2 , min)
scaled = as.data.frame(scale(data0, center = min, scale = max - min))


# Criar base de treino e teste normalizados
trainNN = scaled[index , ]
testNN = scaled[-index , ]

# Fit neural network
NN <- neuralnet(is_canceled ~ hotel + lead_time + adults + children + stays_in_weekend_nights + stays_in_week_nights + country + adr,
                trainNN, hidden = c(3,2),
                linear.output = T,
                lifesign = 'full',
                rep = 1, 
                threshold = 0.10,
                stepmax = 50000)

## Plot neural network
# plot(NN)
plot.nnet(NN)

## Previsão usando a rede

predict_testNN = compute(NN, testNN[,c(2:9)])
predict_testNN = (predict_testNN$net.result * (max(data0$is_canceled) - min(data0$is_canceled))) + min(data0$is_canceled)
plot(datatest$is_canceled, predict_testNN, col='blue', pch=16, ylab = "predicted cancelling rate NN", xlab = "real cancelling rate")

## Matriz de Confusão

confusion <- cbind.data.frame(datatest$is_canceled, predict_testNN)
confusion$predict_binary <- ifelse(confusion$predict_testNN > 0.5, 1, 0)
colnames(confusion) <- c('is_canceled', 'predicted_NN', 'predicted_binary')

confusion$is_canceled <- as.factor(confusion$is_canceled)
confusion$predicted_binary <- as.factor(confusion$predicted_binary)

confusionMatrix(confusion$is_canceled, confusion$predicted_binary)

# Calculate Root Mean Square Error (RMSE)
RMSE.NN = (sum((datatest$is_canceled - predict_testNN)^2) / nrow(datatest)) ^ 0.5
RMSE.NN

## Cross validation do modelo

k = 50 #Número de repetições para cada um dos k fold
RMSE.NN = NULL

List = list( )

# Fitar dentro do loop
for(j in 10:150){
  for (i in 1:k) {
    index = sample(1:nrow(data0),j )
    
    trainNN = scaled[index,]
    testNN = scaled[-index,]
    datatest = data0[-index,]
    
    NN <- neuralnet(is_canceled ~ hotel + lead_time + adults + children + stays_in_weekend_nights + stays_in_week_nights + country + adr,
                    trainNN, hidden = 3,
                    linear.output = T,
                    lifesign = 'full',
                    rep = 1, 
                    threshold = 0.10,
                    stepmax = 40000)

    predict_testNN = compute(NN,testNN[,c(2:9)])
    predict_testNN = (predict_testNN$net.result*(max(data0$is_canceled)-min(data0$is_canceled)))+min(data0$is_canceled)
    
    RMSE.NN [i]<- (sum((datatest$is_canceled - predict_testNN)^2)/nrow(datatest))^0.5
  }

  List[[j]] = RMSE.NN
}

Matrix.RMSE = do.call(cbind, List)

## Boxplot
# boxplot(Matrix.RMSE[,141], ylab = "RMSE", main = "RMSE BoxPlot (tamanho do amostra de treino = 80)")

boxplot(Matrix.RMSE[,141], ylab = "RMSE")

median(Matrix.RMSE[,141])
## Variação do RMSE mediano conforme tamanho da amostra 
med = colMedians(Matrix.RMSE)

X = seq(10,150)

# plot (med~X, type = "l", xlab = "Tamanho da amostra de treino", ylab = "RMSE mediano", main = "Variação do RMSE conforme o tamanho da amostra de treino")
plot (med~X, type = "l", xlab = "Tamanho da amostra de treino", ylab = "RMSE mediano")
abline(h=.5, col="red", type = 'dashed')
