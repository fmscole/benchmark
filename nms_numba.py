from __future__ import absolute_import
from numba import guvectorize,vectorize,cuda
import numpy as np
import numba as nb
import torch
@cuda.jit(device=True)
def DIVUP(m,n):
     return ((m) // (n) + ((m) % (n) > 0))

@cuda.jit(device=True)   
def devIoU(bbox_a,bbox_b):
    top = max(bbox_a[0], bbox_b[0])
    bottom = min(bbox_a[2], bbox_b[2])
    left = max(bbox_a[1], bbox_b[1])
    right = min(bbox_a[3], bbox_b[3])
    height = max(bottom - top, 0)
    width = max(right - left, 0)
    area_i = height * width
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    return area_i / (area_a + area_b - area_i)

@cuda.jit
def nms_kernel(n_bbox, dev_bbox,dev_mask):
    thresh=0.7
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    threadsPerBlock =64
    row_size =min(n_bbox - row_start * threadsPerBlock, threadsPerBlock)
    col_size =min(n_bbox - col_start * threadsPerBlock, threadsPerBlock)

    block_bbox=cuda.shared.array((256),dtype=nb.float32)
    if  cuda.threadIdx.x < col_size:
        block_bbox[cuda.threadIdx.x * 4 + 0] =dev_bbox[(threadsPerBlock * col_start + cuda.threadIdx.x) * 4 + 0]
        block_bbox[cuda.threadIdx.x * 4 + 1] =dev_bbox[(threadsPerBlock * col_start + cuda.threadIdx.x) * 4 + 1]
        block_bbox[cuda.threadIdx.x * 4 + 2] =dev_bbox[(threadsPerBlock * col_start + cuda.threadIdx.x) * 4 + 2]
        block_bbox[cuda.threadIdx.x * 4 + 3] =dev_bbox[(threadsPerBlock * col_start + cuda.threadIdx.x) * 4 + 3]
    
    cuda.syncthreads()

    if  cuda.threadIdx.x < row_size:
        cur_box_idx = threadsPerBlock * row_start + cuda.threadIdx.x
        cur_box = dev_bbox[cur_box_idx * 4:cur_box_idx * 4+4]
        i = 0
        t = np.int64(0)
        
        start = 0
        if row_start == col_start:
            start = cuda.threadIdx.x + 1
        
        for  i in range( start ,col_size ):
            if devIoU(cur_box, block_bbox [i * 4:i * 4+4]) >= thresh:
                t |= np.int64(1)<< i
            
        col_blocks = DIVUP(n_bbox, threadsPerBlock)
        dev_mask[cur_box_idx * col_blocks + col_start] = t
       

def numba_call_nms_kernel(bbox, thresh):
    n_bbox = bbox.shape[0]
    threads_per_block = 64
    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)
    blocks = (col_blocks, col_blocks, 1)
    threads = (threads_per_block, 1, 1)

    mask_dev = cuda.device_array((n_bbox * col_blocks,), dtype=np.uint64)
    bbox=bbox.reshape(-1)
    bbox =cuda.to_device(bbox)

    nms_kernel[blocks, threads](n_bbox, bbox, mask_dev)
    mask_host=mask_dev.copy_to_host()
    return  mask_host

if __name__ == "__main__":
    bbox=np.load("bbox.npy")
    mask_dev= _call_nms_kernel(bbox,0.7)
    print(mask_dev)
