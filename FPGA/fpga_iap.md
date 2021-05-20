# FPGA Multiboot 配置文档

##  实现框图
MultiBoot功能使FPGA能够有选择地从flash存储器中的指定地址加载位流。一个内部生成的脉冲(IPROG)启动配置逻辑，在Golden bitstream WBSTAR(温启动启动地址)寄存器中指定的地址位置跳转到Update bitstream，并尝试加载Update bitstream。如果在Update bitstream加载期间检测到配置错误，则会触发Fallback以加载Golden bitstream。

![1](./img/Multiboot%20and%20Fallback%20Flow.png)

- Fallback 或 Golden bitstream：原image
- Update/MultiBoot image:更新的image
- PROG:
- IPROG
- WBSTAR:温启动启动地址

>若更新到Multiboot image过程中错误或中断，需能够可靠的回退到golden image\

### 两种启动方式：

1. PROG嵌入在位流中。IPROG使用NEXT_CONFIG_ADDR位流选项启用。

> 此方法的IPROG是一个自动化选项，把设置嵌入到stream中，不在用户应用中跳转。(应用中不可控)

2. 使用ICAPE2的IPROG(在本应用程序说明中未涉及):将寄存器写命令应用于ICAPE2原语。

>特定在用户应用程序中，基于某种事件触发Multiboot

## 方式一：PROG嵌入在位流中

WBSTAR（warm boot start address） 和 IPROG（internal PROGRAM）嵌入在位流中




