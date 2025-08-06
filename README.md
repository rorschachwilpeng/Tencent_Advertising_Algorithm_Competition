# 腾讯广告算法竞赛

## 项目简介

这是一个参加腾讯广告算法竞赛的项目，包含数据处理、模型训练和预测的完整流程。

## 项目结构

```
Tencent_Advertising_Algorithm_Competition/
├── Scripts/                    # 主要代码文件
│   ├── dataset.py             # 数据处理模块
│   ├── main.py                # 主程序入口
│   ├── model.py               # 模型定义
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

## 环境要求

- Python 3.8+
- 相关依赖包（见requirements.txt）

## 使用方法

1. 克隆仓库：
```bash
git clone https://github.com/rorschachwilpeng/Tencent_Advertising_Algorithm_Competition.git
cd Tencent_Advertising_Algorithm_Competition
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 准备数据文件：
   - 将数据文件放置在 `TencentGR_1k/` 目录下
   - 确保数据文件结构与项目结构一致

4. 运行程序：
```bash
cd Scripts
python main.py
```

或者使用提供的脚本：
```bash
bash run.sh
```

## 注意事项

- 数据文件较大，请确保有足够的存储空间
- 首次运行可能需要较长时间来处理数据
- 建议在GPU环境下运行以获得更好的性能

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。