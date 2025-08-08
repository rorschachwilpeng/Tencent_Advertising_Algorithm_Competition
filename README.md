# 腾讯广告算法竞赛

## 项目简介

2025腾讯广告算法竞赛，包含数据处理、模型训练和预测的完整流程。

队伍名称：“随性啦”

## 项目结构

```
Tencent_Advertising_Algorithm_Competition/
├── Scripts/                    # 主要代码文件
│   ├── dataset.py             # 数据处理模块
│   ├── main.py                # 主程序入口
│   ├── model.py               # 模型定义
│   ├── model_rqvae.py         # RQ-VAE多模态语义ID转换模型
│   └── run.sh                 # 运行脚本
├── TencentGR_1k/              # 数据文件（本地，未上传到Git）
│   ├── creative_emb/          # 创意嵌入数据
│   ├── data_analysis/         # 数据分析文件
│   └── ...
├── .gitattributes             # Git LFS配置
├── .gitignore                 # Git忽略文件配置
└── README.md                  # 项目说明
```

## 数据文件说明

由于数据文件较大（约13GB），这些文件没有被上传到GitHub仓库中。数据文件包括：

- `TencentGR_1k/creative_emb/` - 创意嵌入数据
- `TencentGR_1k/data_analysis/` - 数据分析文件
- `TencentGR_1k/*.pkl` - 序列化数据文件
- `TencentGR_1k/*.json` - JSON格式数据文件

## 模型说明

### Baseline模型 (`model.py`)
基础的推荐系统模型，包含用户-物品交互建模和特征处理。

### RQ-VAE模型 (`model_rqvae.py`)
用于多模态数据的语义ID转换框架，主要功能包括：

- **多模态嵌入处理**：将不同特征ID的多模态嵌入数据转换为语义ID
- **残差量化**：使用RQ-VAE方法进行向量量化，保留语义信息
- **语义对齐**：通过码本离散化打通多模态语义空间和推荐协同空间
- **平衡聚类**：支持K-means和平衡K-means两种聚类方法

## 注意事项

- 数据文件较大，请确保有足够的存储空间
- 首次运行可能需要较长时间来处理数据
- 建议在GPU环境下运行以获得更好的性能
- RQ-VAE模型训练需要额外的计算资源，建议使用多GPU环境
- 语义ID转换过程会生成额外的中间文件，请确保有足够的磁盘空间

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。