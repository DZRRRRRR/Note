# Dataset

## Cityspaces

https://www.cityscapes-dataset.com/ 

​	账号密码：chuxy C302702216xY！ 邮箱：chuxy@stu.xmu.edu.cn

linux wget下载

```
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=c&password=C302702216xY! &submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

```

deeplabv2-resnet50

<img src="分割DG.assets/image-20220516095657383.png" alt="image-20220516095657383"  />

densecrf没有预期的效果

cityscapes的对应关系
