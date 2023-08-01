import wandb

# 初始化Wandb API
api = wandb.Api()

# 查询特定配置的运行
project_name = "BiSTGNN"  # 替换为你的项目名称
config_name = "your-config-name"  # 替换为你的配置名称
runs = api.runs(path="BiSTGNN", filters={"config.dataset_type": "ExchangeRate"})

import pdb;pdb.set_trace()
# 检查运行是否存在并且状态为已完成或正在运行
# for run in runs:
#     if run.state in ["finished", "running"]:
#         print("运行存在且状态为已完成或正在运行")
#         break
# else:
#     print("运行不存在或状态不为已完成或正在运行")
