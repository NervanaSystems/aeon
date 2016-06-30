import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
import numpy as np

class CpuBackend(object):
    def __init__(self):
        self.use_pinned_mem = False

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        if devlist[buf_index] is None:
            devlist[buf_index] = self.empty_like(hostlist[buf_index])
        print devlist[buf_index].shape, devlist[buf_index].dtype
        print hostlist[buf_index].shape, hostlist[buf_index].dtype
        devlist[buf_index][:] = hostlist[buf_index].T

    def empty_like(self, npary):
        return np.empty_like(npary.T)

    def get_ary(self, cpu_array):
        return cpu_array

class GpuBackend(object):
    '''
    Defines the stubs that are necessary for a backend object
    '''
    def __init__(self, device_id=0):
        self.use_pinned_mem = False
        self.device_id = device_id
        drv.init()
        self.ctx = drv.Device(device_id).make_context()
        self.ctx.pop()

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        self.ctx.push()
        if devlist[buf_index] is None:
            shape, dtype = hostlist[buf_index].shape, hostlist[buf_index].dtype
            devlist[buf_index] = GPUArray(shape, dtype)
        devlist[buf_index].set(hostlist[buf_index].T)
        self.ctx.pop()

    def empty_like(self, npary):
        dbuf = GPUArray(nparay.shape[::-1], nparay.dtype)
        return dbuf

    def get_ary(self, gpu_array):
        self.ctx.push()
        res = gpu_array.get()
        self.ctx.pop()
        return res

    def __del__(self):
        try:
            self.ctx.detach()
        except:
            pass


class MultiGpuBackend(object):
    '''
    Defines the stubs that are necessary for a backend object
    '''
    def __init__(self, num_dev=1):
        self.use_pinned_mem = True
        drv.init()
        assert(num_dev <= drv.Device.count())

        self.num_dev = num_dev
        self.device_ids = list(range(num_dev));
        self.ctxs, self.streams = [], []
        for i in self.device_ids:
            ctx = drv.Device(i).make_context()
            self.ctxs.append(ctx)
            ctx.push()
            self.streams.append(drv.Stream())
            ctx.pop()

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        frag_sz = hostlist[buf_index].shape[0] // self.num_dev

        if devlist[buf_index] is None:
            devlist[buf_index] = self.make_fragments(hostlist[buf_index], frag_sz)

        for idx, ctx, dbuf, strm in zip(self.device_ids, devlist[buf_index],
                                        self.ctxs, self.streams):
            ctx.push()
            dbuf.set_async(hostlist[buf_index][idx*frag_sz:(idx+1)*frag_sz, :].T, strm)
            ctx.pop()

    # Make a fragment ary
    def make_fragments(self, npary, frag_sz):
        mgpuary = []
        (ndata, ndims), ndtype = nparay.shape, npary.dtype
        for ctx in self.ctxs:
            ctx.push()
            mgpuary.append(GPUArray((ndims, frag_sz) , ndtype))
            ctx.pop()
        return mgpuary

    def synchronize(self):
        for ctx in self.ctxs:
            ctx.push()
            ctx.synchronize()
            ctx.pop()

    def __del__(self):
        try:
            for ctx in self.ctxs:
                ctx.detach()
        except:
            pass
