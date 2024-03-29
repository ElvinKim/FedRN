# [CIKM 2022]FedRN: Exploiting k-Reliable Neighbors Towards Robust Federated Learning


## Abstract
Robustness is becoming another important challenge of federated learning in that the data collection process in each client is naturally accompanied by noisy labels. However, it is far more complex and challenging owing to varying levels of data heterogeneity and noise over clients, which exacerbates the client-to-client performance discrepancy. In this work, we propose a robust federated learning method called FedRN, which exploits k-reliable neighbors with high data expertise and similarity. Our method helps mitigate the gap between low- and high-performance clients by training with only a selected set of clean examples, identified by emsembled mixture models. We demonstrate the superiority of FedRN via extensive evaluations on three real-world or synthetic benchmark datasets. Compared with existing robust training methods, the results show that FedRN significantly improves the test accuracy in the presence of noisy labels.

![overview](https://user-images.githubusercontent.com/12638561/132161397-d433a036-0757-4ae0-8c19-aa8a13e339f8.png)


Our main contributions are summarized as follows:
* To the best of our knowledge, this is the first work to present the problem of client-to-client performance discrepancy, which worsens considerably in the presence of noisy labels for federated learning. 
* FedRN remarkably improves the overall robustness without much performance discrepancy owing to the use of k-reliable neighbors.
* FedRN significantly outperforms state-of-the art methods on three real-world or synthetic benchmark datasets with varying levels of data heterogeneity and label noise.


## Installation
Please check below requirements and install packages from `requirements.txt`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Usage
The following command is an `symmetric 0.0-0.4` and `mixed 0.0-0.4` example of running the code.

```
# symmetric 0.0-0.4
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric \
--noise_group_num 100  \
--group_noise_rate 0.0 0.4 \
--num_neighbors 2 \
--method fedrn
```

```
# mixed 0.0-0.4
python main_fed_LNL.py \
--dataset cifar10 \
--model cnn4conv \
--epochs 500 \
--noise_type_lst symmetric pairflip\
--noise_group_num 50 50  \
--group_noise_rate 0.0 0.4 \
--num_neighbors 2 \
--method fedrn
```

### Parameters for learning
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model` | The model architecture. default = `cnn4conv`. |
| `dataset`      | Dataset to use. Options:  `cifar10`, `cifar100`. default = `cifar10`. |
| `lr` | Learning rate for the local models, default = `0.01`. |
| `momentum` | SGD momentum, default = `0.5`. |
| `epochs` | The total number of communication roudns, default = `500`. |

### Parameters for federated learning
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `local_bs` | Local batch size, default = `50`. |
| `loca_ep` | Number of local update epochs, default = `5`. |
| `num_users` | Number of users, Default = `100`. |
| `frac` | The fraction of participating cleints, default = `0.1`. |
| `partition`    | The partition way for Non-IID. Options: `shard`, `dirichlet`, default = `shard` |
| `num_shards` | The number of total shards, default = `200`. |
| `dd_alpha` | The concentration parameter alpha for Dirichlet distribution, default = `0.5`. |


### Parameters for noisy label
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `noise_type_lst` |  Noisy type list. |
| `noisy_group_num`  | Number of clients corresponding to noisy type. |
| `group_noise_rate` | The noise rate corresponding to the noisy group. It increases linearly from 0.0 to noise rate for each group. |

### Parameters for FedRN
| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `num_neighbors` |  The number of reliable neighbors, default = `2` |
| `w_alpha` | The hyperparameter controlling the contribution of expertise and data similiarity.|  |

Please check `run.sh` for commands for various data and noisy label scenarios.


## Experimental Result
![experiment](https://user-images.githubusercontent.com/12638561/138391786-715b1590-7690-4336-b39e-21d4a7677d82.png)
