# 角色设定
你是一位精通图学习与大语言模型的专家。请为我创建一个完整的 Python 项目，项目名称是“面向大模型推理的离散图表征”。我会分 5 个模块详细描述。请将每个模块实现在单独的 Python 文件中，并提供一个主流程脚本来运行完整的流水线。
# 项目背景
本项目旨在实现一个图结构化数据到大模型文本序列的完整 Pipeline。我们将图节点的文本与结构信息离散化为 Token，通过偏置欧拉游走提取局部子图结构，最后序列化为树状文本供 LLM（如 LLaMA、Qwen）或 Transformer 微调使用。
# 技术栈要求
- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric (PyG)
- NetworkX (用于图游走和树结构处理)
- HuggingFace Transformers & Sentence-Transformers (用于文本嵌入)
- scikit-learn

## 项目结构
project/
├── data/ # 存放数据集文件
├── tokenizers/ # 文本 tokenbook 准备
├── models/
│ ├── structural_codebook.py # 模块1：训练结构码本
│ ├── subgraph_extraction.py # 模块2：子图提取
│ ├── node_representation.py # 模块3：节点文本/结构词元化
│ ├── serialization.py # 模块4：偏置欧拉游走与序列化
│ └── llm_finetune.py # 模块5：大模型微调
├── utils.py # 通用工具函数
├── config.py # 超参数配置文件
├── train_codebook.py # 训练结构码本的入口
├── preprocess_data.py # 生成大模型微调训练数据
├── finetune_llm.py # 大模型微调入
└── README.md
## 数据集
我们将使用引文网络数据集，例如 Cora、PubMed。节点包含文本属性（论文标题/摘要）和类别标签。图数据由边列表表示。你可以使用 PyG 或 DGL 来加载数据集。对于原始文本，假设我们有一个 CSV 文件将节点 ID 映射到文本内容。初始节点特征可以使用 Sentence-BERT 嵌入。
## 通用工具模块 (utils.py)
提供以下函数：
- `load_graph_data(dataset_name, root='./data')`：加载图数据集，返回节点特征、边索引、标签、训练/验证/测试掩码，以及节点文本映射字典。
- `load_raw_text(node_text_path)`：从 CSV 加载节点 ID 到文本的映射。
- `extract_sentence_bert_embeddings(text_dict, model_name='all-MiniLM-L6-v2')`：返回每个节点的 Sentence-BERT 嵌入矩阵。
- `compute_pagerank(edge_index, num_nodes)`：返回所有节点的 PageRank 值。
- `normalize_array(arr)`：将数组归一化到 [0,1] 区间。
- `set_seed(seed=42)`：固定随机种子。
## 配置文件 (config.py)
定义所有超参数，包括：
- 数据集名称（如 'Cora'）
- 结构码本大小 M（如 2048）
- 结构码本嵌入维度 d（如 256）
- 语义偏置强度 lambda_semantic（如 0.1）
- 预热 epoch 数 warmup_epochs（如 20）
- EMA 衰减系数 beta（如 0.99）
- 文本 tokenbook 大小 V（如 15062）
- 每个节点选取的文本 token 数量 K（如 5）
- 结构引导筛选强度 lambda_tfidf（如 0.5）
- 偏置欧拉游走权重 alpha, beta, gamma（默认 0.4, 0.3, 0.3）
- 最大游走步数 max_steps（如 50）
- 大模型名称
- 是否使用 QLoRA（是/否）
- LoRA 秩 r（如 16）
- 学习率、批次大小、训练 epoch 等微调参数

请按模块顺序，逐步实现以下核心功能，注意细节要求：
## 模块1：训练结构码本 (structural_codebook.py)
**目标**：在 VQ-VAE 框架下结合结构与语义信息，训练离散 Codebook，并离线统计 TF-IDF。
输入：图数据集（如 PubMed、Cora 等）
输出：训练好的结构码本、GNN 编码器、语义中心向量
**核心逻辑与约束**：
1. **语义偏置的最近邻分配**：
   - 结构距离需进行归一化（因为 L2 距离尺度不一）。
   - 分配公式：`z_i = argmin_k ( ||h_i - e_k||_2_norm - lambda * sim(t_i, \mu_k) )`
   - `t_i`: 节点文本嵌入 (Sentence-BERT 提取)。
   - `\mu_k`: Code k 的语义中心（历史分配到该 Code 的节点文本嵌入的移动平均）。
   - `sim`: 余弦相似度。
2. **预热机制 (Warm-up)**：
   - 前 N 个 epoch (如 N=20) 为预热期，强制 `lambda = 0`，仅使用结构距离分配，同时累积更新 `\mu_k`。
   - 预热结束后，恢复传入的 `lambda` 值（支持 0, 0.1, 1.0 等超参调节）。
3. **离线 TF-IDF 统计 (训练完成后执行)**：
   - 统计矩阵 `count[C][V]` (C=Codebook大小, V=文本词表大小)。
   - `df[t]`：Token t 在多少个不同的 Code 中出现过。
   - 平滑 IDF 计算：`idf = log(C / (df[t] + 1))`，避免除 0 错误。
   - 计算并保存归一化后的 `TF-IDF_norm[c][t]` 到本地（归一化方式：除以该 Code 下的最大 TF-IDF 值）。
具体流程：
1. 基于 VQ-VAE 框架，使用 GNN 编码器（GraphSAGE 或 GCN）将每个节点编码为结构向量 h_i。
2. 与传统方法不同，在量化节点到离散码时，引入**语义偏置**。节点被分配的结构码由结构距离与文本语义相似度共同决定：
   z_i = argmin_{k∈{1,…,M}} ( ||h_i − e_k||_2 − λ · sim(t_i, μ_k) )
   其中 t_i 是节点 i 的文本嵌入（通过预训练 Sentence-BERT 获得），μ_k 是结构码 k 的语义中心（历史上被分配到该码的所有节点的文本嵌入的指数移动平均），sim(·,·) 为余弦相似度，λ >= 0 为控制语义偏置强度的超参数。
3. 训练初期，μ_k 可能是零向量或随机噪声，语义相似度无意义。采用**预热机制**：在前 warmup_epochs 个 epoch 内，仅使用结构距离分配码字，待码分配稳定后，再逐步引入语义偏置。
4. 余弦相似度在 [-1,1] 范围，而结构距离 ||h_i − e_k||_2 的尺度依赖于向量维度，可能从 0 到几十。因此需将结构距离也归一化（例如除以最大值），使两项尺度匹配。
5. λ 控制语义偏置强度：λ=0 退化为原版 VQGraph，λ=0.1 弱偏置，λ=1.0 强偏置。
6. 训练损失包含：重建损失（重建节点特征和图拓扑）、VQ 损失（码本向量靠近编码器输出）、承诺损失（编码器输出靠近码本向量）。可选项：加入对比学习损失（InfoNCE）强制同一码内的节点文本相似。
7. 每个 batch 后，用指数移动平均更新语义中心 μ_k：
   μ_k ← β · μ_k + (1−β) · (本 batch 内分配给码 k 的节点文本嵌入的均值)
8. 训练结束后保存 GNN 编码器、码本向量 E、语义中心 μ_k。

## 模块2：局部子图提取 (subgraph_extraction.py)
**目标**：提取 1-hop 局部子图。
**逻辑**：
- 输入图数据集，基于 PyG 的 `k_hop_subgraph` 或 NetworkX 的 `ego_graph`，为每个节点提取 1-hop 子图，保留节点属性和边结构。
输入：图数据集、目标节点 ID
输出：以目标节点为中心的局部子图（节点集合、边集合、子图邻接信息）

具体流程：
1. 对于给定的图 G 和目标节点 v_t，提取其 k 跳邻居子图（默认 k=1 或 2）。
2. 实现函数 `extract_subgraph(G, v_t, k)`：
   - 使用 BFS 从 v_t 出发，收集所有距离 <= k 的节点。
   - 子图包括这些节点以及它们之间的所有边。
   - 返回子图的节点列表、边列表和邻接关系字典。
3. 如果子图过大，可考虑重要性采样（如基于度或 PageRank 的截断），但本模块先保持简单。

## 模块3：节点表示离散化 (node_representation.py)
**目标**：将节点表示为 Top-K 个文本 Tokens + 1 个结构 Token。
**逻辑与约束**：
1. **文本 Token 筛选**：
   - 节点文本嵌入与 Tokenbook 嵌入算余弦相似度 `text_sim[t]` (归一化到 [0,1])。
   - 读取模块一的 `TF-IDF_norm`。**注意异常处理**：若推理时遇到未见过的结构 Code，退化为纯文本相似度 (即 TF-IDF 权重置0) 或使用全局平均 TF-IDF。
   - 融合打分：`score[t] = text_sim[t] * (1 + lambda_prior * TF-IDF_norm[c][t])`。
   - 选取 Score 最高的 Top-K tokens。
2. **结构 Token 获取**：
   - 用训练好的 GCN/GraphSAGE 编码目标节点，与 Codebook 计算余弦相似度，取最高相似度的 Token。
   - 最终节点表示格式为字典：`{'text_tokens': [t1, t2...], 'struct_token': s_tok}`。
输入：子图（节点集合、边集合）、结构码本、文本 tokenbook
输出：子图中每个节点由 “Top-K 个文本 token + 一个结构 token” 共同表示

### 3.1 文本 token 的选取（结构引导筛选）
1. **事前离线统计**（可在模块 1 完成后进行）：
   - 遍历训练集所有节点，根据节点分配的结构码和原始文本 token，构建矩阵 count[C][V]，记录每个结构码下每个文本 token 的出现次数。
   - 计算每个 token t 的文档频率 df[t]：有多少个不同的结构码至少有一个节点包含该 token。
   - 计算每个结构码 c 下每个 token t 的 TF-IDF 权重：
     tf = count[c][t] / sum(count[c][:])
     idf = log(C / (df[t] + 1))   （加 1 平滑避免除零）
     tfidf[c][t] = tf * idf
   - 对每个结构码 c，将 tfidf[c] 除以最大值以归一化到 [0,1]。
2. **在线推理**：
   - 对于目标节点，先获取其结构码 c。
   - 计算节点文本嵌入与 tokenbook 中所有 token 嵌入的余弦相似度 text_sim（归一化到 [0,1]）。
   - 融合得分：score[t] = text_sim[t] × (1 + λ_tfidf × TF-IDF_norm[c][t])
   - 选择得分最高的 K 个 token 作为该节点的文本表示。
3. 若某个 token 从未在训练集中出现（df=0），idf 计算已经通过加 1 平滑处理。
4. 若某个结构码在训练中仅有极少数节点，导致其 TF-IDF 向量接近 0，则退化为纯文本相似度选择。

### 3.2 结构 token 的选取
1. 将子图所有节点输入训练好的 GNN 编码器，得到结构向量。
2. 对于每个节点，计算其结构向量与结构码本中所有码向量的余弦相似度，选择相似度最高的码作为该节点的结构 token（形如 <S_15>）。

### 3.3 联合表示
最终，子图中的每个节点用一个字典表示，包含：
- node_id
- struct_token (例如 '<S_15>')
- text_tokens (列表，例如 ['graph', 'neural', 'network'])

## 模块4：序列化模块 (serialization.py)
核心：**偏置欧拉回路游走 + 回溯压缩**
**目标**：偏置欧拉回路游走与树状压缩，生成 LLM 可读文本。
**逻辑与约束**：
1. **子图预处理**：将无向边拆分为双向有向边，确保每个节点入度=出度（欧拉回路存在前提）。
2. **偏置函数 `B(u, v)`**：
   - `B(u, v) = \alpha * norm_degree(v) + \beta * norm_pagerank(v) + \gamma * text_sim(u, v)`。
   - `norm_degree` 和 `norm_pagerank` 需归一化到 [0,1]。
3. **偏置欧拉游走 (严格遵守以下步骤)**：
   - 维护 `visited_edges` 和 `S_raw` 序列。
   - 每步选择未访问的出边中 `B(u,v)` 最大的节点；若分数相同选 ID 最小的。
   - **死胡同回退逻辑**：若当前节点无未访问出边，但图内仍有未访问边，则从 `S_raw` 倒序查找最近的有未访问出边的节点 `u'`，用 BFS 找到从当前节点到 `u'` 的最短路径，路径上的节点加入 `S_raw`，并标记经过的边。
4. **回溯压缩为树结构**：
   - 扫描 `S_raw`，判断是“新节点”还是“回溯”（之前出现过即为回溯）。
   - 用栈维护路径，消除 `A-B-A` 回溯，生成多叉树 `TreeNode`。子节点按偏置分数降序排列。
5. **序列化输出**
输入：
- v_t: 目标节点 ID
- G_sub: 局部子图（节点列表、边列表、节点属性字典）
- bias_func: 偏置函数 B(u, v)，返回优先级分数
- max_steps: 最大游走步数（默认为遍历所有边）

输出：序列化文本字符串

算法步骤：

### 步骤1：子图预处理
将子图的每条无向边 (u,v) 拆分为两条有向边 (u→v) 和 (v→u)，以保证欧拉回路存在。

### 步骤2：偏置欧拉游走
生成原始节点访问序列 S_raw。
初始化：
- S_raw = [v_t]
- u_current = v_t
- visited_edges = set()  记录已访问的有向边
- step_count = 0

循环（直到所有有向边都被访问或达到 max_steps）：
1. 获取从 u_current 出发且未访问的有向边对应的邻居候选列表 candidates。
2. 若有候选边：
   a. 对每个候选 v 计算偏置分数 B(u_current, v)。
   b. 选择分数最高的 v*（平局时选节点 ID 最小的）。
   c. 标记边 (u_current, v*) 为已访问。
   d. 将 v* 加入 S_raw，移动到 v*。
3. 若无候选边（当前节点所有出边均已访问）：
   a. 从 S_raw 的末尾向前扫描，找到第一个仍存在未访问出边的节点 u'。
   b. 若找不到 u'，跳出循环。
   c. 使用 BFS 寻找从 u_current 到 u' 的最短路径，将路径上的节点依次加入 S_raw（路径上的边若未访问则标记为已访问）。
   d. 移动到 u'。
4. step_count += 1。

返回 S_raw。

### 步骤3：回溯压缩
将原始游走序列 S_raw 压缩为树状结构，消除回溯引起的节点重复。

数据结构 TreeNode: { node_id, children (按偏置分数降序排列) }

算法：
- root = TreeNode(S_raw[0])
- stack = [root]
- 遍历 S_raw[1:] 中的每个节点 v：
  - 若 v 不在当前栈的节点 ID 集合中：
    - 创建新节点 new_node，加入栈顶的 children，入栈。
  - 否则（v 已在栈中，即回溯）：
    - 从栈顶弹出直到栈顶节点 ID 等于 v。

返回 root。

### 步骤4：序列化输出
将树结构展平为缩进文本：
每行格式：{层级数字} {结构Token} [文本Token列表]
- 层级数字：0 = 目标节点，1 = 直接邻居，2 = 两跳邻居，以此类推
- 子节点层级 = 父节点层级 + 1
- 同一父节点下的同级节点按偏置分数降序排列
- 无节点ID，无属性标注，无目标节点标记
输出示例：
0 <S_15> [graph, neural, network]
1 <S_5> [convolution, layer, deep]
2 <S_3> [attention, mechanism]
1 <S_8> [optimization, gradient]
1 <S_2> [theory, proof]
偏置函数设计：
B(u, v) = α * norm_degree(v) + β * norm_pagerank(v) + γ * text_sim(u, v)
其中 text_sim 是节点 u 和 v 文本嵌入的余弦相似度。

## 模块5：大模型微调 (llm_finetune.py)
分两阶段：QLoRA 快速验证 -> 标准 LoRA 最终训练

### 训练数据生成 (preprocess_data.py)
1. 对于训练集的每个节点 v_t，调用模块 2、3、4 生成序列化文本。
2. 构建指令微调格式的样本：
{

  "instruction": "You are given a subgraph of a citation network, centered at a target node.
Format:
- Each line: <level> <structure_token> [text_tokens]
- level 0 = target node, level 1 = direct neighbor, level 2 = two-hop neighbor
- Nodes under the same parent are sorted by structural importance (degree + PageRank + semantic similarity)",
  "input": "0 <S_15> [graph, neural, network]\n1 <S_5> [convolution, layer, deep]\n2 <S_3> [attention, mechanism]\n1 <S_8> [optimization, gradient]\n1 <S_2> [theory, proof]",
  "output": "Neural_Networks"
}3. 将样本写入 JSONL 文件（train.jsonl, val.jsonl, test.jsonl）。

### QLoRA 微调（验证）
- 使用 transformers 加载基座模型（ Llama3-8B-Instruct），设置 load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"。
- 应用 LoRA：r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"], lora_dropout=0.05。
- 训练参数：learning_rate=2e-4, batch_size=4, gradient_accumulation_steps=2, num_epochs=3, warmup_ratio=0.03, bf16=True, gradient_checkpointing=True。
- 使用 Hugging Face Trainer，数据整理器为 DataCollatorForSeq2Seq。
- 评估指标：验证集准确率。
- 验证通过后保存模型。

### 标准 LoRA 最终训练
- 不量化，以 bf16 全精度加载基座模型。
- LoRA 配置与 QLoRA 相同。
- 适当增加训练 epoch 数（如 5）。
- 保存最终模型，并在测试集上报告准确率。

## 主流程脚本
### train_codebook.py
1. 加载数据，提取 Sentence-BERT 嵌入。
2. 训练结构码本（模块1），保存模型及语义中心。
3. （可选）离线统计 TF-IDF 矩阵，保存供后续使用。

### preprocess_data.py
1. 加载已训练的结构码本和文本 tokenbook。
2. 对每个训练/验证/测试节点：
   - 提取子图（模块2）
   - 离散化节点表示（模块3）
   - 偏置欧拉游走序列化（模块4）
   - 生成指令样本
3. 写入 JSONL 文件。

### finetune_llm.py
1. 加载训练/验证 JSONL 数据。
2. 根据命令行参数决定 QLoRA 或标准 LoRA 模式。
3. 启动微调，记录日志和检查点。
4. 评估最终模型并打印测试准确率。
---
# 编程要求
1. **类型提示 (Type Hinting)**：所有函数和类必须包含 Python 类型提示。
2. **面向对象/模块化**：不要写成一个巨长的脚本，每个模块封装成独立的 Class。
3. **注释与日志**：在关键数学公式计算、游走回退逻辑、树压缩逻辑处加上详细的中文注释。使用 `logging` 打印训练进度和处理进度。
4. **设备兼容**：代码需支持 CPU 和 CUDA 自动切换 (`device = 'cuda' if torch.cuda.is_available() else 'cpu'`)。
5. 所有超参数从 config.py 读取。
6. 确保各模块可以独立测试。
7. 使用 PyTorch 和 transformers 库。
8. 图相关操作使用 PyG 或 DGL。