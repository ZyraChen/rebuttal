# ArgCheck - 论证图事实核查系统

基于论证图的事实核查系统，使用对抗性辩论和抽象论证框架进行多轮证据搜索和判决。

## 系统要求

- Python 3.8+
- 阿里云通义千问API密钥
- Jina Search API密钥

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置API密钥：

有两种方式配置API密钥：

**方式1：环境变量（推荐）**
```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export JINA_API_KEY="your-jina-api-key"
```

**方式2：直接修改config.py**
编辑 `config.py` 文件，填写您的API密钥：
```python
DASHSCOPE_API_KEY = "your-dashscope-api-key-here"
JINA_API_KEY = "your-jina-api-key-here"
```

## 项目结构

```
ArgCheck_v1/
├── chains/              # LLM调用链
│   ├── claim_decomposer.py    # Claim分解
│   ├── pro_chain.py            # 正方查询生成
│   ├── con_chain.py            # 反方查询生成
│   └── judge_chain.py          # 判决生成
├── core/                # 核心模块
│   ├── claim_graph.py          # Claim论证图
│   └── evidence_pool.py        # 证据池管理
├── llm/                 # LLM客户端
│   └── qwen_client.py          # 通义千问客户端
├── tools/               # 工具模块
│   ├── jina_search.py          # Jina搜索API
│   ├── claim_argument_extractor.py  # 论点提取
│   ├── argument_merger.py      # 论点合并
│   └── claim_attack_detector.py     # 攻击关系检测
├── utils/               # 工具函数
├── step1_evidence_collection.py    # 步骤1: 证据收集
├── step2_argumentation_graph.py    # 步骤2: 论证图构建
├── step3_judge.py                   # 步骤3: 判决生成
└── config.py            # 配置文件
```

## 使用方法

### 单个Claim核查

```bash
# 步骤1: 证据收集
python step1_evidence_collection.py --claim "你的claim内容" --output output_dir

# 步骤2: 论证图构建
python step2_argumentation_graph.py --step1-file output_dir/single_step1_evidence.json --output output_dir

# 步骤3: 判决生成
python step3_judge.py --step2-file output_dir/single_step2_graph.json --output output_dir
```

### 批量处理

```bash
# 步骤1: 证据收集（批量）
python step1_evidence_collection.py --dataset data/dataset.json --output output_dir --max-parallel 8

# 步骤2: 论证图构建（批量）
python step2_argumentation_graph.py --step1-dir output_dir --output output_dir --max-parallel 10

# 步骤3: 判决生成（批量）
python step3_judge.py --step2-dir output_dir --output output_dir --max-parallel 10
```

## 主要功能

1. **Claim分解**：将复杂claim分解为可验证的子claim
2. **对抗性搜索**：正反双方进行多轮对抗性证据搜索
3. **论点提取**：从搜索结果中提取结构化论点
4. **攻击关系检测**：检测论点之间的攻击关系
5. **论证图计算**：使用Grounded Extension算法计算被接受的论点
6. **最终判决**：基于被接受的论点生成判决（Supported/Refuted/Not Enough Evidence）

## 注意事项

- 确保API密钥配置正确
- 建议使用环境变量配置API密钥，避免泄露
- 批量处理时注意API调用频率限制
- 输出文件包含完整的推理过程和中间结果

## 许可证

本项目仅供学术研究使用。


