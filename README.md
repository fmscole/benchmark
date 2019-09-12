# benchmark
```
python build.py build_ext --inplace
nvcc --shared -Xcompiler -fPIC nms_cuda.cu nms_cuda.cpp  -o nms.so  -I /home/fms/anaconda3/include/python3.7m
```
