# 横向训练任务

rong_homo



0. **进入容器（双方）**

host方：

```bash
docker exec -it confs_10000_python bash
cd fate_flow/
```

guest方：

```bash
docker exec -it confs_9999_python bash
cd fate_flow/
```



1. **准备数据与配置文件**

```bash
ls rong_homo/
```

host方：

```
(venv) [root@e0ae2e3b5362 fate_flow]# ls rong_homo/
rong_host.csv  rong_test.csv  upload_host.json  upload_test.json
```

guest方：

```
(venv) [root@1c3f9949fd8e fate_flow]# ls rong_homo/
homo_lr_binary_conf.json  homo_secureboost_binary_conf.json  rong_test.csv
homo_lr_dsl.json          homo_secureboost_dsl.json          upload_guest.json
homo_nn_binary_conf.json  homo_secureboost_dsl_1.json        upload_test.json
homo_nn_dsl.json          rong_guest.csv
```



文件分为数据文件和**三种**配置文件（json格式）

其中host方拥有本方数据和测试数据（\_host.csv和\_test.csv）以及上传这两张数据的配置文件（upload_host.json和upload_test.json）

guest方除了拥有本方数据、测试数据（\_guest.csv和\_test.csv）、上传这两张数据的配置文件（upload_guest.json和upload_test.json）之外，还有一个定义了任务流程DAG的配置文件（\_dsl.json），以及一个运行时参数配置文件（\_conf.json）。



2. **上传数据（双方）**

上传数据配置文件

以upload_host.json为例

```json
{
    "file":"fate_flow/rong_homo/rong_host.csv", // 需要上传的数据位置
    "head":1,
    "partition":10,
    "work_mode":1, // 这三行都不用改
    "table_name":"rong_host", //上传之后的表名，建议命名为：原表名_{host/guest/test}
    "namespace":"homo" // 上传之后的namespace，建议指定为homo或者hetero即可
}    
```



通用格式

```bash
python fate_flow_client.py -f upload -c ${上传数据配置文件}
```

host方

```bash
python fate_flow_client.py -f upload -c rong_homo/upload_host.json
python fate_flow_client.py -f upload -c rong_homo/upload_test.json
```

guest方

```bash
python fate_flow_client.py -f upload -c rong_homo/upload_guest.json
python fate_flow_client.py -f upload -c rong_homo/upload_test.json
```



3. **执行任务(guest方)**

通用格式

```bash
python fate_flow_client.py -f submit_job -d ${dsl文件} -c ${conf文件}
```

运行时只需指定conf文件与dsl文件两个配置文件即可。



如我想执行一个横向boost模型任务

```bash
python fate_flow_client.py -f submit_job -d rong_homo/homo_secureboost_dsl.json -c rong_homo/homo_secureboost_conf.json
```



如果执行其他任务，则只需把-c和-d参数换成对应的文件



4. **参数调整与修改**

dsl文件在一个任务中可以不用修改

优化模型我们只需在conf文件中调整"algorithm_parameters"，除此之外，可调整的参数在下方用注释标出了。

```json
{
    "initiator": {
        "role": "guest",
        "party_id": 9999
    },
    "job_parameters": {
        "work_mode": 1,
        "processors_per_node": 2
    },
    "role": {
        "guest": [9999],
        "host": [10000],
        "arbiter": [10000]
    },
    "role_parameters": {
        "guest": {
            "args": {
                "data": {
                    // 指定要训练的数据（guest方）
                    "train_data": [{"name": "rong_guest", "namespace": "homo"}],
                    "eval_data": [{"name": "rong_test", "namespace": "homo"}]
                }
            },
            "dataio_0":{
                // 指定标签列（guest方）
                "with_label": [true],
                "label_name": ["y"],
                "label_type": ["int"],
                "output_format": ["dense"]
            },
            "feature_scale_0": {
                "method": ["standard_scale"]
            },
            "feature_scale_1": {
                "method": ["standard_scale"]
            }
        },
        "host": {
            "args": {
                "data": {
                    // 指定要训练的数据（host方）
                    "train_data": [{"name": "rong_host", "namespace": "homo"}],
                    "eval_data": [{"name": "rong_test", "namespace": "homo"}]
                }
            },
            "dataio_0":{
                // 指定标签列（host方，在纵向任务中host方无标签）
                "with_label": [true],
                "label_name": ["y"],
                "label_type": ["int"],
                "output_format": ["dense"]
            },
            "feature_scale_0": {
                "method": ["standard_scale"]
            },
            "feature_scale_1": {
                "method": ["standard_scale"]
            }
        }
    },
    // 算法参数可调
    "algorithm_parameters": {
        "secureboost_0": {
            "task_type": "classification",
            "learning_rate": 0.01,
            "num_trees": 3,
            "subsample_feature_rate": 0.6,
            "n_iter_no_change": false,
            "tol": 0.0001,
            "bin_num": 50,
            "validation_freqs": 1,
            "tree_param": {
                "max_depth": 3
            },
            "objective_param": {
                "objective": "cross_entropy"
            },
            "encrypt_param":{
                "method": "IterativeAffine"
            },
            "predict_param": {
                "with_proba": true,
                "threshold": 0.5
            },
            "cv_param": {
                "n_splits": 5,
                "shuffle": false,
                "random_seed": 103,
                "need_cv": false
             }
        },
        "evaluation_0": {
            "eval_type": "binary"
        },
        "evaluation_1": {
            "eval_type": "binary"
        }
    }
}
```

