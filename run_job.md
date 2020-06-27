### HOMO

本次任务分为测试任务和实际任务

测试任务包括：test_homo_lr, test_homo_boost. 数据集为breast

实际任务包括：homo_lr, homo_boost. 数据集为rong360



1. **准备数据与配置文件**

下载homo_boost, homo_lr, test_homo_boost, test_homo_lr和upload 共5个配置文件夹

以及数据文件rong（包括rong360_host.csv, rong360_guest.csv, rong360_test.csv）: 

https://drive.google.com/file/d/15yZbdDTXw-F8jJ7zhuhd9Nmf8RA7EmdS/view?usp=sharing



2. **将所有文件夹拷贝至docker的python容器中**

其中host只需拷贝rong数据文件夹和upload文件夹中的带**'host'**和**'test'**的文件，

且无需拷贝其他4个配置文件夹；

而guest需拷贝rong数据文件夹和upload文件夹中的带**'guest'**和**'test'**的文件，

且要拷贝其他4个配置文件夹



首先拷贝到虚拟机中，然后通过如下命令拷贝到python容器中

```bash
docker cp ${本机文件(夹)路径} ${containerID}:${容器文件(夹)路径}
```

其中container ID可通过如下命令查看

```bash
docker ps | grep python
```

容器文件夹路径

```
/data/projects/fate/python/fate_flow
```



3. **执行任务**

以test_homo_lr为例



**上传数据（双方）**

host方

```bash
python fate_flow_client.py -f upload -c upload/test_upload_host.json
python fate_flow_client.py -f upload -c upload/test_upload_test.json
```

guest方

```bash
python fate_flow_client.py -f upload -c upload/test_upload_guest.json
python fate_flow_client.py -f upload -c upload/test_upload_test.json
```



**执行任务(guest方)**

```bash
python fate_flow_client.py -f submit_job -d test_homo_lr/homo_lr_dsl.json -c test_homo_lr/homo_lr_conf.json
```



如果执行其他任务，则只需把-c和-d参数换成对应的文件