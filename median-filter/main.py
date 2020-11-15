import pyopencl as cl
from pyopencl import device_type
import numpy as np
from imageio import imread, imsave
from time import time

imputImage = imread('input.png', as_gray=True).astype(np.float32)

platforms = cl.get_platforms()
devices = platforms[0].get_devices(device_type.GPU)
print("Devices: " + str(devices))

ctx = cl.Context(devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# Kernel function
with open('kernel.cl', 'r') as file:
    src = file.read()

start = time()

# Kernel function instantiation
program = cl.Program(ctx, src).build()

# Allocate memory for variables on the device
img_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=imputImage)
result_g = cl.Buffer(ctx, mf.WRITE_ONLY, imputImage.nbytes)
width_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(imputImage.shape[1]))
height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(imputImage.shape[0]))

# Call Kernel. Automatically takes care of block/grid distribution
program.medianFilter(queue, imputImage.shape, None, img_g, result_g, width_g, height_g)
result = np.empty_like(imputImage)
cl.enqueue_copy(queue, result, result_g)

# Show the blurred image
imsave('output.jpg', result.astype(np.uint8))
end = time()
print("Time taken: " + str((end-start) * 1000) + " ms")
