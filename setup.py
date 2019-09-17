from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension,BuildExtension
setup(
    name="nms_cuda",
    ext_modules=[
        CUDAExtension(
            "nms_cuda",
            sources=["nms_cuda_vision.cpp","nms_cuda.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
