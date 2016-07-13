#pragma once
#include <map>
#include <opencv2/core/core.hpp>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>

namespace nervana {

    static const std::map<std::string, std::tuple<int, int, size_t>> all_outputs {
        {"int8_t",   std::make_tuple<int, int, size_t>(NPY_INT8,    CV_8S,  sizeof(int8_t))},
        {"uint8_t",  std::make_tuple<int, int, size_t>(NPY_UINT8,   CV_8U,  sizeof(uint8_t))},
        {"int16_t",  std::make_tuple<int, int, size_t>(NPY_INT16,   CV_16S, sizeof(int16_t))},
        {"uint16_t", std::make_tuple<int, int, size_t>(NPY_UINT16,  CV_16U, sizeof(uint16_t))},
        {"int32_t",  std::make_tuple<int, int, size_t>(NPY_INT32,   CV_32S, sizeof(int32_t))},
        {"uint32_t", std::make_tuple<int, int, size_t>(NPY_UINT32,  CV_32S, sizeof(uint32_t))},
        {"float",    std::make_tuple<int, int, size_t>(NPY_FLOAT32, CV_32F, sizeof(float))},
        {"double",   std::make_tuple<int, int, size_t>(NPY_FLOAT64, CV_64F, sizeof(double))},
        {"char",     std::make_tuple<int, int, size_t>(NPY_INT8,    CV_8S,  sizeof(char))}
    };

    class output_type {
    public:
        output_type() {}
        output_type(const std::string& r)
        {
            auto tpl_iter = all_outputs.find(r);
            if (tpl_iter != all_outputs.end()) {
                std::tie(np_type, cv_type, size) = tpl_iter->second;
                tp_name = r;
            } else {
                throw std::runtime_error("Unable to map type " + r);
            }
        }
        bool valid() const {
            return tp_name.size() > 0;
        }
        std::string tp_name;
        int np_type;
        int cv_type;
        size_t size;
    };
}


