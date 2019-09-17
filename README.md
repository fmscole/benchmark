# NMS(in faster-rcnn)    benchmark
1.ubuntu 18    
2.python 3.7    
3.cuda 9.2    
4.gpu 2080ti    

    
build _nms_gpu_post    
```
python build.py build_ext --inplace
```
build nms_cuda(pytorch 1.x)    
```
python setup.py build_ext --inplace
```
run  :    
```
jupyter notebook
```
select:    
benchmark.ipynb
