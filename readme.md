# 基本信息

**URL for VMs**

https://drive.google.com/file/d/1jpAZs_xTIcSfAsuoBkw3rmy2mzVxuQLT/view?usp=sharing


**集群信息**

两台主机名: 

| 主机名 | 又名   | cluster ID |
| ------ | ----- | ---------- |
| party1 | host | 10000      |
| party2 | guest | 9999       |


**工作用户**

root

```
sudo su -
password: root
```


**终端工作目录（host）**

/root/docker-deploy/

```
cd docker-deploy/
```





# 部署

**检查 private IP**
```
ifconfig
```
不出意外应该是名叫ens\*\*的那个网卡

**修改hosts以及配置文件**
```
vim /etc/hosts
```
把party1, party2 对应的IP改成上面查到的private IP, Ubuntu64_1对应party1, Ubuntu64_2对应party2

```
cd docker-deploy/
vim parties.conf
```
把两个iplist变量中的party1, party2替换成对应的private IP

**部署集群（只用在host上执行，重启后需再次部署）**

```
cd docker-deploy/
bash generate_config.sh
bash docker_deploy.sh all
```



**查看组件状态**

```
docker ps
```



**验证部署**

```
docker exec -it confs-10000_python_1 bash     #进入python组件容器内部
cd /data/projects/fate/python/examples/toy_example
python run_toy_example.py 10000 9999 1 
```


# 训练

```
docker exec -it confs-10000_python_1 bash (host)
docker exec -it confs-9999_python_1 bash (guest)
cd fate_flow/
```

建议把一个项目的所有数据文件、配置文件放在一个文件夹
这里以examples/文件夹为例

**准备数据文件**
host方：examples/data/breast_a.csv
guest方：examples/data/breast_b.csv

**配置上传选项**
host方：vi examples/upload_host.json
guest方 > vi examples/upload_guest.json
