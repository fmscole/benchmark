import nms_cuda 
import numpy as np
bbox=np.load("bbox.npy")
n_bbox = bbox.shape[0]
keep_out=np.zeros(bbox.shape[0],dtype=np.int32)
n=nms_cuda.nms_cuda(keep_out,bbox)
print(n,keep_out)