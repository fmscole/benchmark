#include "nms_cuda.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

int nms_cuda(py::array_t<int> keep_out,py::array_t<float> boxes_host)
          {
                py::buffer_info boxes_host_buf = boxes_host.request();
                int boxes_num=boxes_host_buf.shape[0];
                int boxes_dim=boxes_host_buf.shape[1];
               
                // py::array_t<int>  keep_out=py::array_t<int>(boxes_num);
                py::buffer_info keep_out_buf = keep_out.request();

                float nms_overlap_thresh=0.7;
                int num_out=0;

                nms_cuda_compute((int*)keep_out_buf.ptr,&num_out,(float*)boxes_host_buf.ptr,boxes_num,boxes_dim,nms_overlap_thresh);
                
                return  num_out;
          }

PYBIND11_MODULE(nms_cuda, m) {
    m.doc() = "nms_cuda"; 
    m.def("nms_cuda", &nms_cuda, "A function which nms_cuda");
}