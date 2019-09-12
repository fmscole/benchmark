import nms 
import numpy as np

bbox=np.load("bbox.npy")
n_bbox = bbox.shape[0]

keep_out=np.zeros(bbox.shape[0],dtype=np.int32)
n=nms.nms_cuda(keep_out,bbox)
print(keep_out)
print(n)