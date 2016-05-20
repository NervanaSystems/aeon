#include "device.hpp"

Device* Device::create(DeviceParams* params) {
#if HAS_GPU
    if (params->_type == CPU) {
        return new Cpu(reinterpret_cast<CpuParams*>(params));
    }
    return new Gpu(reinterpret_cast<GpuParams*>(params));
#else
    assert(params->_type == CPU);
    return new Cpu(reinterpret_cast<CpuParams*>(params));
#endif
}