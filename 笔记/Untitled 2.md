makefile

```makefile
KERNELDIR := /usr/src/linux-headers-4.9.253-tegra-ubuntu18.04_aarch64/kernel-4.9
CURRENT_PATH:=$(shell pwd)
obj-m := task1.o
build:kernel_modules
kernel_modules:
	$(MAKE) -C $(KERNELDIR) M=$(CURRENT_PATH) modules
clean:
	$(MAKE) -C $(KERNELDIR) M=$(CURRENT_PATH) clean
```

## 相关命令

### 加载模块

```
insmod 
modprobe(解决依赖)
```

### 卸载驱动

```
rmmod（推荐）
modprobe -r
```

### 查看加载的模块

```
lsmod
```

