# Surrogate Benchmark for Model Merging Optimization
This repository provides surrogate benchmarks for model merging optimization. We provide two types of benchmarks: SMM-Bench-PS and SMM-Bench-DFS. The description of the benchmarks can be found in the paper: <https://arxiv.org/abs/2509.02555>.

## Requirements

- Our surrogate models require `lightgbm`

```shell
pip install -r requirements.txt
```

## Prepare surrogate model files
- Download surrogate model files from <https://drive.google.com/drive/folders/1aeLvUzOwVKqL7vSIdDBMDryJLisxQSXw?usp=sharing>
- Unzip the downloaded files and place them in the `surrogate/` directory.
- Example: `surrogate/smm-bench-dfs` and `surrogate/smm-bench-ps`

## Simple Example of SMM-Bench-PS

```python
import numpy as np
from smmbench import SMMBenchPS

# Load surrogate
dir_name = 'surrogate/smm-bench-ps'
f_ps = SMMBenchPS(dir_name)

# 64 dimensional continuous variable within the range [0.0, 1.0]
rng = np.random.default_rng()
x = rng.uniform(0.0, 1.0, f_ps.D)

# Evaluation (train_acc_ja)
feval = f_ps(x)

# Evaluation (test_acc_ja) for testing merge model
feval_test = f_ps(x, mode='test_acc')

print('feval (train): ', feval, ', feval (test): ', feval_test)
```

## Simple Example of SMM-Bench-DFS

```python
import numpy as np
from smmbench import SMMBenchDFS

# Load surrogate
dir_name = 'surrogate/smm-bench-dfs'
f_dfs = SMMBenchDFS(dir_name)

rng = np.random.default_rng()

# 63 dimensional continuous variable within the range [0.4, 1.5]
x = rng.uniform(0.4, 1.5, f_dfs.D_x)
# dimensional categorical variable with 3 choices (0, 1, 2)
c = rng.integers(low=0, high=3, size=f_dfs.D_c)

# Evaluation (train_acc_ja)
feval = f_dfs(c, x)

# Evaluation (test_acc_ja) for testing merge model
feval_test = f_dfs(c, x, mode='test_acc')

print('feval (train): ', feval, ', feval (test): ', feval_test)
```

## Example
* Please see `example.ipynb` for sample usage of SMM-Bench

## Reference
Rio Akizuki, Yuya Kudo, Nozomu Yoshinari, Yoichi Hirose, Toshiyuki Nishimoto, Kento Uchida, and Shinichi Shirakawa, "Surrogate Benchmarks for Model Merging Optimization," International Conference on Automated Machine Learning (AutoML 2025), Non-Archival Content Track, 2025. [[Link](https://openreview.net/forum?id=Yv3tRT8olz)] [[arXiv](https://doi.org/10.48550/arXiv.2509.02555)]

```
@inproceedings{akizuki2025,
    title={Surrogate Benchmarks for Model Merging Optimization},
    author={Rio Akizuki and Yuya Kudo and Nozomu Yoshinari and Yoichi Hirose and Toshiyuki Nishimoto and Kento Uchida and Shinichi Shirakawa},
    booktitle={AutoML 2025 Non-Archival Content Track},
    year={2025},
    url={https://openreview.net/forum?id=Yv3tRT8olz}
}
```