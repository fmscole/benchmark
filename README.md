# NMS(in faster-rcnn)    benchmark
1.ubuntu 18
2.python 3.7
3.cuda 9.2
4.gpu 2080ti

run  :
```
python build.py build_ext --inplace
nvcc --shared -Xcompiler -fPIC nms_cuda.cu nms_cuda.cpp  -o nms.so  -I /home/fms/anaconda3/include/python3.7m
```
