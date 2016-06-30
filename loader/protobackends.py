import pycuda.driver as drv
from pycuda.gpuarray import GPUArray
import numpy as np

class CpuBackend(object):
    def __init__(self):
        self.use_pinned_mem = False

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        if devlist[buf_index] is None:
            devlist[buf_index] = np.empty_like(hostlist[buf_index].T)
        print devlist[buf_index].shape, devlist[buf_index].dtype
        print hostlist[buf_index].shape, hostlist[buf_index].dtype
        devlist[buf_index][:] = hostlist[buf_index].T

    def get_ary(self, cpu_array):
        return cpu_array

class GpuBackend(object):
    '''
    Defines the stubs that are necessary for a backend object
    '''
    def __init__(self, device_id=0):
        self.use_pinned_mem = True
        self.device_id = device_id
        drv.init()
        self.ctx = drv.Device(device_id).make_context()

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        self.ctx.push()
        hbuf = hostlist[buf_index]
        if devlist[buf_index] is None:
            shape, dtype = hbuf.shape[::-1], hbuf.dtype
            devlist[buf_index] = GPUArray(shape, dtype)
        devlist[buf_index].set(hbuf.T)
        self.ctx.pop()

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
            self.ctxs.append(drv.Device(i).make_context())
            self.streams.append(drv.Stream())
            self.ctxs[-1].pop()
        self.ctxs[0].push()

    def consume(self, buf_index, hostlist, devlist):
        assert 0 <= buf_index < 2, 'Can only double buffer'
        hbuf = hostlist[buf_index]

        frag_sz, ndims, ndtype = hbuf.shape[0] // self.num_dev, hbuf.shape[1], hbuf.dtype

        # Create fragment array destination if missing
        if devlist[buf_index] is None:
            devlist[buf_index] = []
            for ctx in self.ctxs:
                ctx.push()
                devlist[buf_index].append(GPUArray((ndims, frag_sz), ndtype))
                ctx.pop()

        # Initiate the transfer
        for idx, ctx, dbuf, strm in zip(self.device_ids, devlist[buf_index],
                                        self.ctxs, self.streams):
            ctx.push()
            dbuf.set_async(hbuf[idx*frag_sz:(idx+1)*frag_sz, :].T, strm)
            ctx.pop()

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
