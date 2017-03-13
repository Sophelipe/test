# 问题记录

1.  自定义的dataLayer如果不在setup中将top reshape的话，在后面接其他层就崩溃了。原因可能是因为分配的内存空间不足：前两个top的大小都是0。
```
I0313 16:59:59.235903 15744 net.cpp:157] Top shape: (0)
I0313 16:59:59.235903 15744 net.cpp:157] Top shape: (0)
I0313 16:59:59.235903 15744 net.cpp:157] Top shape: 1 1 (1)
I0313 16:59:59.235903 15744 net.cpp:165] Memory required for data: 4
```
