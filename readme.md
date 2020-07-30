# Deployment

https://github.com/FederatedAI/KubeFATE/tree/master/docker-deploy



# Basic Information

**user**

root



# Start Cluster (both parties)

It could be also used to restart closed modules (for example, when you find some module such as FATE Board stopped unexpectedly, you could use these commands to restart it).

```bash
cd /data/projects/fate/confs-<container-ID>
docker-compose up -d
cd /data/projects/fate/serving-<container-ID>
docker-compose up -d
```



**View Modules' status**

```bash
docker ps
```



# Training

**Enter into Docker（both parties）**

```bash
docker exec -it confs-10000_python_1 bash (host)
docker exec -it confs-9999_python_1 bash (guest)
cd fate_flow/
```



You are recommended to put all the data files and config files into one directory of your project.

Here we take the *examples/* directory for example.



**Get Data Files Ready（both parties）**

host:

examples/data/breast_a.csv

guest:

examples/data/breast_b.csv



**Configure Upload Files（both parties）**

host

```bash
vi examples/upload_host.json
```

```
{
  "file": "examples/data/breast_a.csv", # data file to upload
  "head": 1,
  "partition": 10, 
  "work_mode": 1, # 1 for cluster mode
  "namespace": "fate_flow_test_breast",
  "table_name": "breast" # these two identify a dataset in FATE
}
```



guest

```bash
vi examples/upload_guest.json
```

```json
{
  "file": "examples/data/breast_b.csv", 
  "head": 1,
  "partition": 10,
  "work_mode": 1, 
  "namespace": "fate_flow_test_breast",
  "table_name": "breast" 
}
```



**Upload Data（both parties）**

General Command

```bash
python fate_flow_client.py -f upload -c ${upload json file}
```

Here they are

host:

```bash
python fate_flow_client.py -f upload -c examples/upload_host.json 
```

guest：

```bash
python fate_flow_client.py -f upload -c examples/upload_guest.json 
```



**Configure Training Parameters（only guest）**

Take Hetero Logistic Regression in *examples/* directory for example

```bash
vi examples/test_hetero_lr_job_conf.json
```

```
{
    "initiator": {
        "role": "guest", # guest itself
        "party_id": 9999 # own id
    },
    "job_parameters": {
        "work_mode": 1, # 1 for cluster mode
        "processors_per_node": 1 
    },
    "role": {
    	# corresponding to party ids
        "guest": [9999],
        "host": [10000],
        "arbiter": [10000]
    },
    "role_parameters": {
    	# guest's parameters
        "guest": {
            "args": {
                "data": {
                	# specified in upload configs
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
        # host's parameters
        "host": {
            "args": {
                "data": {
                	# specified in upload configs
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



**Submit Training Job（guest）**

General Command

```bash
python fate_flow_client.py -f submit_job -d ${dsl json file} -c ${runtime json config file}
```

Here it is

```
python fate_flow_client.py -f submit_job -d examples/test_hetero_lr_job_dsl.json -c examples/test_hetero_lr_job_conf.json
```



Dsl file defines the relationships among all modules that this task uses (for example, one's output is another's input)，which is a Directed Acyclic Graph，and could be customized.

*examples/test_hetero_lr_job_dsl.json* here defines:

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

a linear task which holds 5 modules (dataio_0, hetero_feature_binning_0, hetero_feature_selection_0, hetero_lr_0, evaluation_0),where each module keeps 3 attributes: module, input and output respectively。

After submitting a job, its process could be monitored on FATE Board using the given URL

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

If nothing went wrong, retcode and retmsg should be 0 and success respectively.

model_id and model_version would be used in binding model and online inference.

Job_id could be used to query job status:

```bash
python fate_flow_client.py -f query_task -j $JOBID | grep f_status
```



**Load Already-trained Model（only guest）**

Configure Load json file

```bash
vi examples/publish_load_model.json
```

```
{
    "initiator": {
        "party_id": "9999",
        "role": "guest"
    },
    "role": {
        "guest": ["9999"],
        "host": ["10000"],
        "arbiter": ["10000"]
    },
    "job_parameters": {
        "work_mode": 1, 
        # model_id model_version should be specified
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202006241043279578162"
    }
}
```

Load model

General Command

```bash
python fate_flow_client.py -f load -c ${load json file}
```

Here it is

```bash
python fate_flow_client.py -f load -c examples/publish_load_model.json
```

Return

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



**Bind loaded Model to Serving（only guest）**

Configure Bind json file

```bash
vi examples/bind_model_service.json
```

```
{
    "service_id": "001", # assign an identified
    "initiator": {
        "party_id": "9999",
        "role": "guest"
    },
    "role": {
        "guest": ["9999"],
        "host": ["10000"],
        "arbiter": ["10000"]
    },
    "job_parameters": {
        "work_mode": 1, 
        # model_id and model_version should be specified
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202006241043279578162"
    }
}
```

Bind model

General Command

```bash
python fate_flow_client.py -f bind -c ${bind json file}
```

Here it is

```bash
python fate_flow_client.py -f bind -c examples/bind_model_service.json
```

Return

```
{
    "retcode": 0,
    "retmsg": "service id is 001"
}
```



**Online Testing（only guest）**

When a model is loaded and bound to Serving, the guest could push an HTTP POST request which contains new data record for online test.

Method 1: using *curl*

(where ${SERVING_SERVICE_IP} is the guest)

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

Method 2: RESTED(Chrome extension)

Return

```
{"flag":0,"data":{"prob":0.020201574669380547,"retcode":0},"retmsg":"success","retcode":0}
```
