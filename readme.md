# 基本信息

**URL for VMs**

https://drive.google.com/open?id=1FFQ8mkT2jnpjV-fqVwScWeLoCIhILPrG


**集群信息**

两台主机名: 

| 主机名 | 又名  | private IP      | cluster ID |
| ------ | ----- | --------------- | ---------- |
| party1 | host  | 192.168.246.149 | 10000      |
| party2 | guest | 192.168.246.148 | 9999       |



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

**部署集群（只用在host上执行，重启后需再次部署）**

```
cd docker-deploy/
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



