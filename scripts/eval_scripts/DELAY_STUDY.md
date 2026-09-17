# 多模型延迟评估

## 运行

复制 `scripts/eval_scripts/delay_study.py` 为同目录的
`delay_study.local.py`，只修改顶部实验参数。填写真实模型清单，例如：

```python
STUDY = "completed_models_delay_v2"
MODELS = [
    dict(id="model_a", config="results/sacred/<run>/1/config.json",
         checkpoint="results/models/<run>/<step>"),
]
```

路径也可以是绝对路径。`id` 在研究内唯一。模型在各自训练地图上测试；
不尝试跨不同输入/动作/智能体维度加载。当前入口支持现行 BCRBC/GLIDE
checkpoint，不是任意算法或历史架构的通用加载器。

旧配置若缺少 `bcrbc_time_block_every`，核对原训练源码后在模型条目中显式
填写 `legacy_time_block_every=4`（或实际值）。不自动用当前默认配置补齐
网络，也不改变 horizon、solver steps、模型宽度或深度。

```bash
python scripts/eval_scripts/delay_study.local.py
```

默认每条件64局、8个并行环境。模型、条件和评估批次顺序执行；不启动多个
GPU训练/评估进程争资源。末批可以少于8局，因此总局数严格等于配置值。
未填写模型清单不会启动任务。

相同地图、环境初始化配置和并行数共用SMAC进程。切换模型或延迟条件不重启
SC2，只更新延迟采样器并reset；模型历史、KV缓存和诊断状态每批重建。
不同地图或初始化配置需新建进程；不足整批的末批单独复用较小的环境组。
执行顺序按环境配置分组，不保证全局模型顺序。

延迟与模型随机数按每批seed初始化，但SMAC随机状态随reset继续推进。
不同条件不承诺相同初始战局；中断后跳过完成批次，也不承诺复现不中断时的
环境随机轨迹。每批result记录environment_seed、environment_batch_index和
worker_pids。此协议为v3，勿混入旧版逐批重启的研究目录。
根目录session.json记录本次待运行批次，session.log保存整体进度和进程输出；
批次run.log保存该批Python输出。

每个条件沿用相同环境seed序列，独立的 CPU torch Generator 采样延迟。
模型生成噪声不会推进延迟随机流。策略改变仍然会导致环境轨迹分叉。

## 条件

- 高斯 μ = -2,-1,0,1,2；σ = 0,0.5,1,1.5,2。
- 实际延迟为 `min(8, ceil(max(0, sample)))`，坐标不是离散后的实际均值。
- 三个非正均值、零方差格点共用固定0延迟结果。固定1、2复用相应高斯格点。
- 额外固定4、8；离散均匀[0,1]、[0,2]、[0,4]、[0,8]。
- 混合：50% N(0,1)+50% N(2,1)；90% N(0,1)+10% N(4,1)。
- 动态：N(0,1)/N(2,1)每16步交替；两状态Markov保持概率0.9。
- 混合/动态的状态在单个环境的智能体之间共享，条件高斯噪声独立。
  balanced mixture与动态对照具有相同状态组成，但时间相关性不同；有限
  episode的实际频率不必相同。周期模式从低延迟开始，Markov从均匀状态开始。
- 始终保持cap=8，不用均匀分布上界改变控制器可修正的历史范围。
- 总共33个独立条件，64局时每模型2112局。

分布只决定新数据包到达时间，不重新抽取旧包。训练仍走原来零延迟逻辑。
这是一组观测延迟评估，不应标记为通信延迟实验。环境state和合法动作沿用
现有wrapper语义（即时），论文需说明这个信息条件。

## 流程与诊断

实际执行始终采用checkpoint对应的决策流程。真实完整本地观测只进入影子
Reference，使用实际执行动作历史；Reference不是最优策略或性能上界。
诊断不更新模型、不改变执行缓存，也不消费策略随机流。

`MASK_INTERVENTION=False`为默认。启用后额外记录关闭生成的影子动作，
字段名为`intervention_*`、`correction`、`damage`，只解释同模型的推理干预，
不是独立训练的“仅编码基线”，也不产生这种基线的胜率。

重建诊断始终可以比较执行z与MASK编码z；这不需要额外执行一个MASK策略。
Reference z来自完整本地历史的encoder，生成z来自实际决策前向，绝不重新
采样。Decoder比较如下：

- Reference observation：完整z与完整decoder历史。
- MASK observation：当前已修正MASK历史的decoder输出。
- Generated observation：用同一个MASK decoder历史条件解码实际生成z。

默认输出latent/observation MSE、MAE、cosine及补全前后latent MSE差值。
latent只建议同模型内部对照。cosine对零向量采用PyTorch的epsilon约定。
`FEATURE_GROUPS={map_name: {group: [start, end]}}`可按真实观测布局增加分组MSE；
不能盲目复用不同地图的切片。不配置时只报告整体误差，不伪造语义分组。

动作指标：greedy agreement、Q-softmax KL（temperature=1）、参考Q差距。
KL不是epsilon-greedy行为分布的KL，参考Q差距也不是环境regret。
动作及重建主汇总仅统计“当前缺失且至少两个合法动作”的位置，保存分子分母；
计数为0时CSV留空。完整观测自身的`reference_obs_all_mse`另按全部位置统计，
无延迟也可检查tokenizer重建水平。

数据还包括当前缺失率、从未收到率、最新观测年龄、连续缺失长度、真实采样
延迟直方图、P95、截顶比例及动态状态/状态持续步数。`age=null`表示从未
收到；分桶表编码成-1，绘图单独标记，不接入普通age曲线。

胜率区间采用95% Wilson；诊断比值区间按episode重采样1000次（加权分子/
分母）。这些区间不是训练seed方差。误差分桶与动态图是描述性均值，不是
因果效应或显著性检验。动态图包括初始状态，横轴明确标注。

动作选择时间在CUDA同步后计时，不包含评分分支；它是整个环境batch的延迟，
不是单智能体推理时间。吞吐与峰值显存包含诊断，不能作为部署成本。
诊断decoder会重新计算已修正历史，有额外开销，不要将其算入训练性能。

## 数据与复跑

```text
results/evaluate/<STUDY>/
  manifest.json
  runs/<model_id>/<condition_id>/batch_<episode_offset>/
    job.json, effective_config.json, run.log
    episodes.jsonl, decisions.jsonl.gz, result.json
  tables/episodes.csv, summary.csv, quality_buckets.csv
  figures/*.png, *.pdf
```

每个decision保存标量误差和动作，未默认保存高维向量。终局支持的dead_allies/
dead_enemies写入episode；环境不提供时为空。模型配置/checkpoint哈希、评估
源码哈希及git提交记录在清单中。结果不是只有TensorBoard均值。

仅`result.json`完成的批次参与汇总；中断批次重新执行，完成批次跳过。
同STUDY不允许混入改变后的协议或checkpoint；修改模型、条件、并行数或
局数请使用新STUDY名称。数据不会自动删除或合并成无法追溯的跨版本结果。

```bash
python scripts/eval_scripts/summarize_study.py results/evaluate/<STUDY>
```

## 独立画图

汇总/绘图离线运行，只需numpy、pandas、matplotlib，不需要SC2或PyTorch。
评估端使用训练环境；绘图端可用单独环境。每个脚本都接受研究根目录：

```bash
python scripts/eval_scripts/plot_winrate_heatmap.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_distribution_robustness.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_reconstruction_quality.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_policy_consistency.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_error_action_relation.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_dynamic_response.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_efficiency.py results/evaluate/<STUDY>
python scripts/eval_scripts/plot_episode_case.py results/evaluate/<STUDY> <model_id> <condition_id> <episode> <agent>
```

地图分开画；热力图统一0–1色阶；未评估单元为空，不插值。
模型不同训练结构与seed的解释来自manifest，不把测试episodes当成训练seed。
病例应预先指定或明确选择规则，不能只挑成功案例。
