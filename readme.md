# 基本信息

**Links for VMs**

party1(host): 

https://drive.google.com/file/d/1CTGmE6zcwXqAuCvn5Xr-PH-PW2QWFi7E/view?usp=sharing 

party2(guest): 

https://drive.google.com/file/d/1uLMsLhvtaFaqL8RH8EqwMJ03FZmVGxj-/view?usp=sharing




**集群信息**

两台主机名: 

| 主机名 | 又名  | cluster ID |
| ------ | ----- | ---------- |
| party1 | host  | 10000      |
| party2 | guest | 9999       |



**工作用户**

root

```bash
sudo su -
password: root
```



**终端工作目录（host）**

/root/docker-deploy/

```bash
cd docker-deploy/
```



# 部署

**检查 private IP**

```bash
ifconfig
```

不出意外应该是名叫ens\*\*的那个网卡



**修改hosts以及配置文件**

```bash
vim /etc/hosts
```

把party1, party2 对应的IP改成上面查到的private IP, Ubuntu64_1对应party1, Ubuntu64_2对应party2

```bash
cd docker-deploy/
vim parties.conf
```

把两个iplist变量中的party1, party2替换成对应的private IP



**部署集群（只用在host上执行，只需第一次执行，之后重启只需启动集群）**

```bash
cd docker-deploy/
bash generate_config.sh
bash docker_deploy.sh all
```



**启动集群（部署集群之后，每次重启只需启动集群）**

这个也可以用来启动已经关闭的组件（当你发现某个组件比如fateboard 停止了，也可以用这个命令启动）

与部署集群不同，单纯启动集群会保留之前的文件和配置，部署集群会忽略之前的配置，格式化

```bash
cd /data/projects/fate/confs-<container-ID>
docker-compose up -d
```



**查看组件状态**

```bash
docker ps
```



**验证部署**

```bash
docker exec -it confs-10000_python_1 bash     #进入python组件容器内部
cd /data/projects/fate/python/examples/toy_example
python run_toy_example.py 10000 9999 1 
```



**停止集群，删除部署**

```bash
bash docker_deploy.sh --delete all
```

```bash
cd /data/projects/fate/confs-<id>/  # id of party (10000, 9999)
docker-compose down
rm -rf ../confs-<id>/
```



# 训练

**进入容器（双方）**

```bash
docker exec -it confs-10000_python_1 bash (host)
docker exec -it confs-9999_python_1 bash (guest)
cd fate_flow/
```



建议把一个项目的所有数据文件、配置文件放在一个文件夹

这里以examples/文件夹为例



**准备数据文件（双方）**

host方:

examples/data/breast_a.csv

guest方:

examples/data/breast_b.csv



**配置上传选项（双方）**

host方

```bash
vi examples/upload_host.json
```

```
{
  "file": "examples/data/breast_a.csv", # 数据文件
  "head": 1,
  "partition": 10,
  "work_mode": 1, # 修改成1，表示集群模式
  "namespace": "fate_flow_test_breast",
  "table_name": "breast" # 上传上去的表名两方要一致
}
```



guest方

```bash
vi examples/upload_guest.json
```

```
{
  "file": "examples/data/breast_b.csv", # 数据文件
  "head": 1,
  "partition": 10,
  "work_mode": 1, # 修改成1，表示集群模式
  "namespace": "fate_flow_test_breast",
  "table_name": "breast" # 上传上去的表名两方要一致
}
```



**上传数据文件（双方）**

通用格式

```bash
python fate_flow_client.py -f upload -c $数据文件
```

这里是

host方：

```
python fate_flow_client.py -f upload -c examples/upload_host.json 
```

guest方：

```bash
python fate_flow_client.py -f upload -c examples/upload_guest.json 
```



**配置训练参数文件（guest方）**

以examples/文件夹下的纵向LR为例

```bash
vi examples/test_hetero_lr_job_conf.json
```

```
{
    "initiator": {
        "role": "guest", # 自己
        "party_id": 9999 # 自己的id
    },
    "job_parameters": {
        "work_mode": 1, # 集群模式
        "processors_per_node": 1
    },
    "role": {
    	# 修改成对应的party id
        "guest": [9999],
        "host": [10000],
        "arbiter": [10000]
    },
    "role_parameters": {
    	# guest方参数
        "guest": {
            "args": {
                "data": {
                	# name是在upload文件里改好的上传名
                    "train_data": [{"name": "breast", "namespace": "fate_flow_test_breast"}]
                }
            },
            "dataio_0":{
                "with_label": [true],
                "label_name": ["y"],
                "label_type": ["int"],
                "output_format": ["dense"]
            }
        },
        # host方参数
        "host": {
            "args": {
                "data": {
                	# name是在upload文件里改好的上传名
                    "train_data": [{"name": "breast", "namespace": "fate_flow_test_breast"}]
                }
            },
             "dataio_0":{
                "with_label": [false],
                "output_format": ["dense"]
            }
        }
    },
    "algorithm_parameters": {
    	# 算法的参数
        "hetero_lr_0": {
            "penalty": "L2",
            "optimizer": "rmsprop",
            "eps": 1e-5,
            "alpha": 0.01,
            "max_iter": 3,
            "converge_func": "diff",
            "batch_size": 320,
            "learning_rate": 0.15,
            "init_param": {
				"init_method": "random_uniform"
            }
        }
    }
}
```



**提交训练任务（guest方）**

通用格式

```bash
python fate_flow_client.py -f submit_job -d $DSL文件 -c $数据文件
```

这里是

```
python fate_flow_client.py -f submit_job -d examples/test_hetero_lr_job_dsl.json -c examples/test_hetero_lr_job_conf.json
```



.dsl文件定义了这个任务各个组件之间的关系（如谁的输出是谁的输入等），类似一个DAG，可自定义

如本examples/test_hetero_lr_job_dsl.json定义了

```
{
    "components" : {
        "dataio_0": {
            "module": "DataIO",
            "input": {
                "data": {
                    "data": [
                        "args.train_data"
                    ]
                }
            },
            "output": {
                "data": ["train"],
                "model": ["dataio"]
            },
			"need_deploy": true
         },
        "hetero_feature_binning_0": {
            "module": "HeteroFeatureBinning",
            "input": {
                "data": {
                    "data": [
                        "dataio_0.train"
                    ]
                }
            },
            "output": {
                "data": ["train"],
                "model": ["hetero_feature_binning"]
            }
        },
        "hetero_feature_selection_0": {
            "module": "HeteroFeatureSelection",
            "input": {
                "data": {
                    "data": [
                        "hetero_feature_binning_0.train"
                    ]
                },
                "isometric_model": [
                    "hetero_feature_binning_0.hetero_feature_binning"
                ]
            },
            "output": {
                "data": ["train"],
                "model": ["selected"]
            }
        },
        "hetero_lr_0": {
            "module": "HeteroLR",
            "input": {
                "data": {
                    "train_data": ["hetero_feature_selection_0.train"]
                }
            },
            "output": {
                "data": ["train"],
                "model": ["hetero_lr"]
            }
        },
        "evaluation_0": {
            "module": "Evaluation",
            "input": {
                "data": {
                    "data": ["hetero_lr_0.train"]
                }
            },
            "output": {
                "data": ["evaluate"]
            }
        }
    }
}
```

一套线性的拥有5个组件（dataio_0, hetero_feature_binning_0, hetero_feature_selection_0, hetero_lr_0, evaluation_0）的任务，每个组件有3个属性：module, input, output，分别定义了本组件使用的模块、使用的输入数据和模型，输出的数据和模型。

提交任务后，在给出的board_url（fate board模块）上能够可视化查看任务的进行

```
{
    "data": {
        "board_url": "http://fateboard:8080/index.html#/dashboard?job_id=202006241043279578162&role=guest&party_id=9999",
        "job_dsl_path": "/data/projects/fate/python/jobs/202006241043279578162/job_dsl.json",
        "job_runtime_conf_path": "/data/projects/fate/python/jobs/202006241043279578162/job_runtime_conf.json",
        "logs_directory": "/data/projects/fate/python/logs/202006241043279578162",
        "model_info": {
            "model_id": "arbiter-10000#guest-9999#host-10000#model",
            "model_version": "202006241043279578162"
        }
    },
    "jobId": "202006241043279578162",
    "retcode": 0,
    "retmsg": "success"
}
```

如果没问题，retcode和retmsg分别应为0和success

model_id和model_version 会在加载模型和在线推理中用到

jobId和model_version等同，可以用jobId来查看训练任务的状态

```bash
python fate_flow_client.py -f query_task -j $JOBID | grep f_status
```



**加载训练好的模型（guest方）**

修改加载模型的配置文件

```bash
vi examples/publish_load_model.json
```

```
{
    "initiator": {
    	# 对应改
        "party_id": "9999",
        "role": "guest"
    },
    "role": {
    	# 对应改
        "guest": ["9999"],
        "host": ["10000"],
        "arbiter": ["10000"]
    },
    "job_parameters": {
        "work_mode": 1, # 1表示集群模式
        # model_id和model_version 改成提交任务后返回的模型信息
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202006241043279578162"
    }
}
```

加载模型

通用格式

```bash
python fate_flow_client.py -f load -c $加载模型配置文件
```

这里是

```bash
python fate_flow_client.py -f load -c examples/publish_load_model.json
```

返回

```
{
    "data": {
        "guest": {
            "9999": 0
        },
        "host": {
            "10000": 0
        }
    },
    "jobId": "202006241043279578162",
    "retcode": 0,
    "retmsg": "success"
}
```



**绑定加载好的模型到serving（guest方）**

修改绑定模型的配置文件

```bash
vi examples/bind_model_service.json
```

```
{
    "service_id": "001", # 给一个唯一的ID
    "initiator": {
    	# 对应改
        "party_id": "9999",
        "role": "guest"
    },
    "role": {
    	# 对应改
        "guest": ["9999"],
        "host": ["10000"],
        "arbiter": ["10000"]
    },
    "job_parameters": {
        "work_mode": 1, # 1表示集群模式
        # model_id和model_version 改成提交任务后返回的模型信息
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202006241043279578162"
    }
}
```

绑定模型

通用格式

```bash
python fate_flow_client.py -f bind -c $绑定模型配置文件
```

这里是

```bash
python fate_flow_client.py -f bind -c examples/bind_model_service.json
```

返回

```
{
    "retcode": 0,
    "retmsg": "service id is 001"
}
```



**在线测试（guest方）**

此时模型已经训练完并且加载绑定到serving，我们guest可以POST发送新数据在线测试

第一种方法：curl

(其中${SERVING_SERVICE_IP}一般就是localhost)

```bash
curl -X POST -H 'Content-Type: application/json' -i 'http://${SERVING_SERVICE_IP}:8059/federation/v1/inference' --data '{
  "head": {
    "serviceId": "001"
  },
  "body": {
    "featureData": {
      "x0": 0.254879,
      "x1": -1.046633,
      "x2": 0.209656,
      "x3": 0.074214,
      "x4": -0.441366,
      "x5": -0.377645,
      "x6": -0.485934,
      "x7": 0.347072,
      "x8": -0.287570,
      "x9": -0.733474,
    },
    "sendToRemoteFeatureData": {
      "id": "123"
    }
  }
}'
```

第二种方法：RESTED(Chrome插件)

返回

```
{"flag":0,"data":{"prob":0.020201574669380547,"retcode":0},"retmsg":"success","retcode":0}
```

