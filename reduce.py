import numpy as np
import taichi as ti
from time import time

ti.init(arch=ti.cuda, device_memory_GB=2)

N = 100
ARR_SIZE = 100000000
a = np.random.randint(low=1, high=10000, size=ARR_SIZE, dtype=np.int32)
# a = np.array([0,1,2,3,4,5,6,7,8,9], dtype=np.int32)
a_ti = ti.field(dtype=ti.int32, shape=(ARR_SIZE))
a_ti.from_numpy(a)


def timeit(name:str, func, *args):
    t_start = time()
    for i in range(N):
        sum = func(*args)
    spent = time() - t_start
    print(f'{name} spent time = {spent}, sum = {sum}')

# ---------------------------------------------------------------------------- #
#                                     numpy                                    #
# ---------------------------------------------------------------------------- #

timeit('np.sum', np.sum, a)


# ---------------------------------------------------------------------------- #
#                                 taichi atomic                                #
# ---------------------------------------------------------------------------- #
@ti.kernel
def sum_atomic(a_ti: ti.template())->ti.int32:
    sum = 0
    for i in range(ARR_SIZE):
        sum += a_ti[i]
    return sum

timeit('taichi atomic', sum_atomic, a_ti)

# ---------------------------------------------------------------------------- #
#                                 taichi reduce                                #
# ---------------------------------------------------------------------------- #
buffer = ti.field(dtype=ti.int32, shape=ARR_SIZE)
buffer_reduced = ti.field(dtype=ti.int32, shape=ARR_SIZE)
buffer_compact = ti.field(dtype=ti.int32, shape=ARR_SIZE)
@ti.kernel
def reduce_once(dst:ti.template(), src:ti.template(), length:int):
    for i in range(dst.shape[0]):
        dst[i] = 0
    for i in range(length):
        if i%(2)==0:
            dst[i] = src[i] + src[i + 1]

@ti.kernel
def compact(dst:ti.template(), src:ti.template())->ti.int32:
    cnt = 0
    for i in range(dst.shape[0]):
        dst[i] = 0
    for i in range(src.shape[0]):
        if src[i] != 0:
            k = ti.atomic_add(cnt, 1)
            dst[k] = src[i]
    return cnt

@ti.kernel
def deepcopy(dst:ti.template(), src:ti.template()):
    for i in range(src.shape[0]):
        dst[i] = src[i]


def sum_reduce():
    length = ARR_SIZE
    buffer = a_ti
    while length > 1:
        reduce_once(buffer_reduced,buffer,length)
        length = compact(buffer_compact, buffer_reduced)
        deepcopy(buffer, buffer_compact)
    sum = buffer[0]
    return sum


timeit('taichi reduce', sum_reduce)


# ---------------------------------------------------------------------------- #
#                            taichi reduce better                              #
# ---------------------------------------------------------------------------- #
@ti.kernel
def reduce_once_better(dst:ti.template(), src:ti.template(), length:int):
    for i in range(length//2):
        dst[i] = src[i] + src[i + length//2]

def reduce_better():
    length = ARR_SIZE
    buffer = a_ti
    reduce_once_better(buffer, a_ti, length)
    sum = buffer[0]
    return sum

timeit('taichi reduce better', reduce_better)