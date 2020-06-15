# 基本信息

**URL for VMs**

https://drive.google.com/file/d/1jpAZs_xTIcSfAsuoBkw3rmy2mzVxuQLT/view?usp=sharing


**集群信息**

两台主机名: 

| 主机名 | 又名   | cluster ID |
| ------ | ----- | ---------- |
| party1 | host | 10000      |
| party2 | guest | 9999       |



**终端工作目录（host）**

~/docker-deploy/

```
cd docker-deploy/
```



**工作用户**

root

```
sudo su -
password: root
```



# 部署

**检查 private IP**
```
ifconfig
```
不出意外应该是名叫ens\*\*的那个网卡

**修改hosts**
```
vim /etc/hosts
```
把party1, party2 对应的IP改成上面查到的private IP, Ubuntu64_1对应party1, Ubuntu64_2对应party2

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



