/*
 Copyright 2015 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <stdlib.h>
#include <cstring>
#include <stdexcept>
#include <memory>
#include "util.hpp"

enum DeviceType { CPU=0, GPU=1 };

class DeviceParams {
public:
    DeviceParams(int type, int id)
    : _type(type), _id(id)
    {}

public:
    int                         _type;
    int                         _id;
    int                         _batchSize;
    nervana::count_size_type    _dtmInfo;
    nervana::count_size_type    _tgtInfo;
};

class CpuParams : public DeviceParams {
public:
    CpuParams(int type, int id, char* data[2], char* targets[2])
    : DeviceParams(type, id) {
        for (int i = 0; i < 2; i++) {
            _data[i] = data[i];
            _targets[i] = targets[i];
        }
    }

public:
    char*                       _data[2];
    char*                       _targets[2];
};

class Device {
public:
    Device(int type) : _type(type) {}
    virtual ~Device() {};
    virtual int init() = 0;
    virtual int copyData(int idx, char* data, int size) = 0;
    virtual int copyLabels(int idx, char* data, int size) = 0;
    virtual int copyDataBack(int idx, char* data, int size) = 0;
    virtual int copyLabelsBack(int idx, char* data, int size) = 0;

    static std::shared_ptr<Device> create(DeviceParams* params, bool alloc);

public:
    int                         _type;
    int                         _dlen;
    int                         _tlen;
};

#if HAS_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#define checkCudaErrors(val) check( (val), cudaSuccess, #val, __FILE__, __LINE__)
#define checkDriverErrors(val) check( (val), CUDA_SUCCESS, #val, __FILE__, __LINE__)

template<typename T>
void check(T err, T sval, const char* const func,
           const char* const file, const int line) {
    if (err == sval) {
        return;
    }
    printf("CUDA error %d at: %s:%d\n", err, file, line);
    throw std::runtime_error("CUDA error\n");
}

class GpuParams : public DeviceParams {
public:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _targets[2];
};

class Gpu : public Device {
public:
    Gpu(GpuParams* params, bool alloc)
    : Device(GPU), _alloc(alloc), _id(params->_id) {
        _dlen = params->_dtmInfo.count * params->_dtmInfo.size * params->_batchSize;
        _tlen = params->_tgtInfo.count * params->_tgtInfo.size * params->_batchSize;
        if (_alloc) {
            init();
            for (int i = 0; i < 2; i++) {
                checkDriverErrors(cuMemAlloc(&_data[i], _dlen));
                checkDriverErrors(cuMemAlloc(&_targets[i], _tlen));
                params->_targets[i] = _targets[i];
                params->_data[i]    = _data[i];
            }
        } else {
            for (int i = 0; i < 2; i++) {
                _data[i] = params->_data[i];
                _targets[i] = params->_targets[i];
            }
        }
    }

    virtual ~Gpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                cuMemFree(_data[i]);
                cuMemFree(_targets[i]);
            }
        }
    }

    int init() {
        try {
            checkCudaErrors(cudaSetDevice(_id));
            checkCudaErrors(cudaFree(0));
        } catch(...) {
            return -1;
        }
        return 0;
    }

    int copyData(int idx, char* data, int size) {
        return copy(_data[idx], data, size);
    }

    int copyLabels(int idx, char* targets, int size) {
        return copy(_targets[idx], targets, size);
    }

    int copyDataBack(int idx, char* data, int size) {
        return copyBack(data, _data[idx], size);
    }

    int copyLabelsBack(int idx, char* targets, int size) {
        return copyBack(targets, _targets[idx], size);
    }

private:
    int copy(CUdeviceptr dst, char* src, int size) {
        try {
            checkDriverErrors(cuMemcpyHtoD(dst, src, size));
        } catch(...) {
            return -1;
        }
        return 0;
    }

    int copyBack(char* dst, CUdeviceptr src, int size) {
        try {
            checkDriverErrors(cuMemcpyDtoH(dst, src, size));
        } catch(...) {
            return -1;
        }
        return 0;
    }

private:
    CUdeviceptr                 _data[2];
    CUdeviceptr                 _targets[2];
    bool                        _alloc;
    int                         _id;
};
#endif

class Cpu : public Device {
public:
    Cpu(CpuParams* params, bool alloc)
    : Device(CPU), _alloc(alloc) {
        _dlen = params->_dtmInfo.count * params->_dtmInfo.size * params->_batchSize;
        _tlen = params->_tgtInfo.count * params->_tgtInfo.size * params->_batchSize;
        if (_alloc) {
            init();
            for (int i = 0; i < 2; i++) {
                _data[i] = new char[_dlen];
                _targets[i] = new char[_tlen];
                params->_targets[i] = _targets[i];
                params->_data[i]    = _data[i];
            }
        } else {
            for (int i = 0; i < 2; i++) {
                _data[i] = params->_data[i];
                _targets[i] = params->_targets[i];
            }
        }
    }

    virtual ~Cpu() {
        if (_alloc == true) {
            for (int i = 0; i < 2; i++) {
                delete[] _data[i];
                delete[] _targets[i];
            }
        }
    }

    int init() {
        return 0;
    }

    int copyData(int idx, char* data, int size) {
        memcpy(_data[idx], data, size);
        return 0;
    }

    int copyLabels(int idx, char* targets, int size) {
        memcpy(_targets[idx], targets, size);
        return 0;
    }

    int copyDataBack(int idx, char* data, int size) {
        memcpy(data, _data[idx], size);
        return 0;
    }

    int copyLabelsBack(int idx, char* targets, int size) {
        memcpy(targets, _targets[idx], size);
        return 0;
    }

private:
    char*                       _data[2];
    char*                       _targets[2];
    bool                        _alloc;
};

