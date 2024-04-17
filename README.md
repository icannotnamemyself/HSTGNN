# Heterogeneous Spatial Temporal Graph Neural Netork

This is the offcial repository of paper "Heterogeneous Spatial Temporal Graph Neural Netork For Multivariate Time Series Forecasting"

![HSTGNN](./fig/overview.png)



# prepare datasets

## single-step


ETTh1, ETTh2, ETTm1, ETTm2, ExchangeRate, ILI, Weather, Electricity will be downloaded automatically

for SP500 dataset, download from https://archive.ics.uci.edu/dataset/554/cnnpred+cnn+based+stock+market+prediction+using+a+diverse+set+of+variables, and put into ./data/SP500 as ./data/SP500/sp500.csv file

## multi-step

PEMS04, PEMS07 datasets are collected from https://github.com/Davidham3/STSGCN

PEMS-BAY, METR-LA datasets are collected from https://github.com/liyaguang/DCRNN

# Run baseline&HSTGNN

## 1. install requirements

to run our code, please first install all the requirements, we assume that you have installed torch and pyg according to your environment
```
pip install -r ./requirements.txt
```




## 2. run scripts


Please first source this init.sh script:

```
source ./init.sh 
```

or manually add this directory to your PATHONPATH environment variable

```
export PYTHONPATH=./
```

### single-step experiment

Please change the settings in the following for what you need.

```python
# running HSTGNN with length 3,6,12 on dataset ETTm1, ETTm2
./scripts/single_step.sh "3 6 12" "ETTm1 ETTm2"  "cuda:2" HSTGNN

# running Crossformer with length 3,6,12 on dataset ETTm1, ETTm2
./scripts/single_step.sh "3 6 12" "ETTm1 ETTm2"  "cuda:2" Crossformer
```
### multi-step experiment


Please change the settings in the following for what you need.
```python
# running HSTGNN with length 3,6,12 on dataset METR_LA, PEMS04
./scripts/multi_step.sh "3 6 12" "METR_LA PEMS04" "cuda:0" HSTGNN

# running ESG with length 3,6,12 on dataset METR_LA, PEMS04
./scripts/multi_step.sh "3 6 12" "METR_LA PEMS04" "cuda:0" ESG
```


### long-term multi-step experiment

```python
# running HSTGNN with length 96 168 336 720 on dataset ETTm1, ETTm2
./scripts/long_term_multistep.sh "96 168 336 720" "ETTm1 ETTm2"  "cuda:2" HSTGNN

# running Crossformer with length 96 168 336 720 on dataset ETTm1, ETTm2
./scripts/long_term_multistep.sh "96 168 336 720" "ETTm1 ETTm2"  "cuda:2" Crossformer
```



# 