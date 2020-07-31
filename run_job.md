# A Federated Learning Task



**data**

We put the dataset we used on google drive:

For horizontal federal learning: https://drive.google.com/file/d/1WqNODomhVV9xmTM7kzeqtBwSyMdQFeBj/view?usp=sharing

For vertical federal learning: https://drive.google.com/file/d/1n9mMukIeUyaW57tL4C4KmDu1myZ02nfK/view?usp=sharing


**Here we take a Horizontal FL task for example**, and Vertical FL task process is similar.



0. **Enter into Docker (both parties)**

host:

```bash
docker exec -it confs_10000_python bash
cd fate_flow/
```

guest

```bash
docker exec -it confs_9999_python bash
cd fate_flow/
```



1. **Get the Data Files and Config Files Ready**

```bash
ls rong_homo/
```

host:

```bash
(venv) [root@e0ae2e3b5362 fate_flow]# ls rong_homo/
rong_host.csv  rong_test.csv  upload_host.json  upload_test.json
```

guest:

```bash
(venv) [root@1c3f9949fd8e fate_flow]# ls rong_homo/
homo_lr_binary_conf.json  homo_secureboost_binary_conf.json  rong_test.csv
homo_lr_dsl.json          homo_secureboost_dsl.json          upload_guest.json
homo_nn_binary_conf.json  homo_secureboost_dsl_1.json        upload_test.json
homo_nn_dsl.json          rong_guest.csv
```

Files are divided into data files (csv format) and config files (json format).

Host keeps its own training data and test data (\_host.csv和\_test.csv), and upload config files which is to upload these two datasets.

guest方除了拥有本方数据、测试数据（\_guest.csv和\_test.csv）、上传这两张数据的配置文件（upload_guest.json和upload_test.json）之外，还有一个定义了任务流程DAG的配置文件（\_dsl.json），以及一个运行时参数配置文件（\_conf.json）。

Guest not only keeps its own training data, test data, upload config files, but also two more config files: a _dsl.json which defines the task flow (DAG) and a _conf.json which is the runtime config file.



2. **Upload Data Files (both parties)**

the *upload_host.json* file

```
{
    "file":"fate_flow/rong_homo/rong_host.csv", // data to upload
    "head":1,
    "partition":10,
    "work_mode":1, 
    "table_name":"rong_host", //table name in FATE. recommended：origin_{host/guest/test}
    "namespace":"homo" // namespace in FATE，recommended:{homo/hetero}
}    
```



General Command

```bash
python fate_flow_client.py -f upload -c ${Upload json File}
```

host

```bash
python fate_flow_client.py -f upload -c rong_homo/upload_host.json
python fate_flow_client.py -f upload -c rong_homo/upload_test.json
```

guest

```bash
python fate_flow_client.py -f upload -c rong_homo/upload_guest.json
python fate_flow_client.py -f upload -c rong_homo/upload_test.json
```



3. **Submit Jon (only guest)**

General Command

```bash
python fate_flow_client.py -f submit_job -d ${dsl json file} -c ${runtime config json file}
```

Here if I'd like to run a Homo Secureboost task:

```bash
python fate_flow_client.py -f submit_job -d rong_homo/homo_secureboost_dsl.json -c rong_homo/homo_secureboost_conf.json
```



4. **Paramter Tunning**

The dsl file could remain unchanged during a job.

To optimize the model we need to change the "algorithm_parameters" and other parameters in the runtime config file, which is marked below:

```
{
    "initiator": {
        "role": "guest",
        "party_id": 9999
    },
    "job_parameters": {
        "work_mode": 1,
        "processors_per_node": 2 // processors to user
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
                    // specified datasets (guest)
                    "train_data": [{"name": "rong_guest", "namespace": "homo"}],
                    "eval_data": [{"name": "rong_test", "namespace": "homo"}]
                }
            },
            "dataio_0":{
                // specified label/the class to predict (guest)
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
                    // specified datasets (host)
                    "train_data": [{"name": "rong_host", "namespace": "homo"}],
                    "eval_data": [{"name": "rong_test", "namespace": "homo"}]
                }
            },
            "dataio_0":{
                // specified label/the class to predict (host. in Vertical FL task it is false)
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
    // algorithms parameters tunning
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
