## 1.2 读取 mean 数据


from dataclasses import asdict
import string
from openpyxl import Workbook
import pandas as pd
import wandb
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

from torch_timeseries.experiments.dlinear_experiment import DLinearExperiment
# 初始化Wandb API
api = wandb.Api()

def convert_dict(dictionary):
    converted_dict = {}
    for key, value in dictionary.items():
        converted_dict[f"config.{key}"] = value
    return converted_dict

# 查询特定配置的运行
project_name = "BiSTGNN"  # 替换为你的项目名称
config_name = "your-config-name"  # 替换为你的配置名称

# single steps report

baselines = ["DLinear", "NLinear", "MTGNN", "Crossformer", "TSMixer", "FiLM", "GRU", "TNTCN"]
datasets = ["ExchangeRate",  "ETTm1", "ETTm2", "ETTh1", "ETTh2"] #"ExchangeRate",
horizons = [3, 6, 12, 24]

all_data = {}
runs = api.runs(path="BiSTGNN")


# 获得带+-的结果
for run in runs:
        if run.state == "finished":
            try:
                if "TNTCN" in str(run.config['model_type']) or  "BiSTGNN" in str(run.config['model_type']):
                    all_data[(run.config['model_type'],run.config['dataset_type'], run.config['horizon'], 'mse')] = run.summary['mse_mean']
                    all_data[(run.config['model_type'],run.config['dataset_type'], run.config['horizon'], 'r2')] = run.summary['r2_mean']
                    all_data[(run.config['model_type'],run.config['dataset_type'], run.config['horizon'], 'r2w')] = run.summary['r2_weighted_mean']
                    all_data[(run.config['model_type'],run.config['dataset_type'], run.config['horizon'], 'mae')] = run.summary['mae_mean']
            except:
                continue
index = pd.MultiIndex.from_tuples(all_data.keys(), names=['model_type','dataset', 'horizon', 'metric'])
mean__df = pd.DataFrame(all_data.values(), index=index)
"""
(Pdb) df
                                               0
dataset horizon metric model_type               
GRU     ETTh1   12     mse         0.3308±0.0058
                       r2          0.5113±0.0118
                       r2w         0.5851±0.0072
                       mae         0.4093±0.0041
DLinear ETTh2   12     mse         0.0747±0.0007
...                                          ...
MTGNN   ETTh2   24     mae         0.2188±0.0054
                12     mse         0.0783±0.0036
                       r2          0.2924±0.0532
                       r2w         0.4056±0.0272
                       mae         0.2052±0.0065

"""
# .loc[:,0] 去掉第一列索引 （0）
mean_df = mean__df.unstack(0).loc[:,0]


# (df_pivot.iloc[:,2] - df_pivot.iloc[:,1] )
df_pivot = mean_df.pivot_table(index=['dataset', 'horizon'], columns='metric')
reverse_metrics = ['r2', 'r2w']

model_types = mean_df.columns

base_model = 'TNTCN'

model_exclude_baseline = set(model_types) - set([base_model])
print(model_exclude_baseline)
total_metrics = 4
df_pivot[base_model]

for model in model_exclude_baseline:
    for metric in ['r2', 'r2w', 'mae', 'mse']:
        baseline_value = df_pivot[base_model][metric]
        model_value = df_pivot[model][metric]
        if metric not in ['r2', 'r2w']:
            df_pivot.loc[:, (model, metric)] =  (baseline_value - model_value)/baseline_value 
        else:
            df_pivot.loc[:, (model, metric)] =  (model_value - baseline_value)/baseline_value 
mean_diff_percent_df = df_pivot.stack().iloc[:,1:]



# 添加 condition formating
import xlsxwriter


writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
mean_diff_percent_df.to_excel(writer, sheet_name='Sheet1')

# Get the xlsxwriter objects from the dataframe writer object.
workbook  = writer.book
worksheet = writer.sheets['Sheet1']



rows = len(mean_diff_percent_df)


start_col = 3
model_types = list(mean_diff_percent_df.columns)
end_col = start_col + len(model_types) - 1

start_col_letter =  xlsxwriter.utility.xl_col_to_name(start_col)
end_col_letter =  xlsxwriter.utility.xl_col_to_name(end_col)



# 调整百分比格式
percent_format = workbook.add_format({'num_format': '0.00%'})
for col_num in range(start_col, end_col+1):
    # 调整列宽以适应内容
    column_width = len(model_types[col_num - start_col])
    worksheet.set_column(col_num, col_num, column_width, percent_format)


for i in range(rows):
    col_letter = xlsxwriter.utility.xl_col_to_name(i)
    worksheet.conditional_format(f'{start_col_letter}{i}:{end_col_letter}{i}', 
                                 {'type': 'data_bar',
                                  'bar_negative_color': 'red',
                                  'bar_color': '#32CD32',#'green',
                                  'bar_direction': 'left',
                                  'bar_axis_position': 'middle',
                                  'bar_solid': True,
                                   'min_type':'min',
                                   'max_type':'max',

#                                  # 'data_bar_2010': True,
                                 }
                                )
    
writer.close()

