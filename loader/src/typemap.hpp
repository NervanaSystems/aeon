/*
 Copyright 2016 Nervana Systems Inc.
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

#include <map>
#include <opencv2/core/core.hpp>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>

namespace nervana
{
    class output_type;
    class shape_type;

    static const std::map<std::string, std::tuple<int, int, size_t>> all_outputs{
        {"int8_t", std::make_tuple<int, int, size_t>(NPY_INT8, CV_8S, sizeof(int8_t))},
        {"uint8_t", std::make_tuple<int, int, size_t>(NPY_UINT8, CV_8U, sizeof(uint8_t))},
        {"int16_t", std::make_tuple<int, int, size_t>(NPY_INT16, CV_16S, sizeof(int16_t))},
        {"uint16_t", std::make_tuple<int, int, size_t>(NPY_UINT16, CV_16U, sizeof(uint16_t))},
        {"int32_t", std::make_tuple<int, int, size_t>(NPY_INT32, CV_32S, sizeof(int32_t))},
        {"uint32_t", std::make_tuple<int, int, size_t>(NPY_UINT32, CV_32S, sizeof(uint32_t))},
        {"float", std::make_tuple<int, int, size_t>(NPY_FLOAT32, CV_32F, sizeof(float))},
        {"double", std::make_tuple<int, int, size_t>(NPY_FLOAT64, CV_64F, sizeof(double))},
        {"char", std::make_tuple<int, int, size_t>(NPY_INT8, CV_8S, sizeof(char))}};
}

class nervana::output_type
{
public:
    output_type() {}
    output_type(const std::string& r)
    {
        auto tpl_iter = all_outputs.find(r);
        if (tpl_iter != all_outputs.end())
        {
            std::tie(m_np_type, m_cv_type, m_size) = tpl_iter->second;
            m_tp_name = r;
        }
        else
        {
            throw std::runtime_error("Unable to map output type " + r);
        }
    }
    bool        valid() const { return m_tp_name.size() > 0; }
    int         get_cv_type() const { return m_cv_type; }
    int         get_np_type() const { return m_np_type; }
    size_t      get_size() const { return m_size; }
    static bool is_valid_type(const std::string& s) { return all_outputs.find(s) != all_outputs.end(); }
    std::string                                  m_tp_name;
    int                                          m_np_type;
    int                                          m_cv_type;
    size_t                                       m_size;
};

class nervana::shape_type
{
public:
    shape_type(const std::vector<size_t>& shape, const output_type& otype, const bool flatten = false)
        : m_shape{shape}
        , m_otype{otype}
        , m_flatten_with_batch_size{flatten}
    {
        m_byte_size = static_cast<size_t>(m_otype.get_size() * get_element_count());
    }

    size_t get_element_count() const { return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<uint32_t>()); }
    size_t get_byte_size() const { return m_byte_size; }
    const std::vector<size_t>& get_shape() const { return m_shape; }
    const output_type&         get_otype() const { return m_otype; }
    bool                       flatten_all_dims() const { return m_flatten_with_batch_size; }
private:
    std::vector<size_t> m_shape;
    output_type         m_otype;
    size_t              m_byte_size;
    bool                m_flatten_with_batch_size;
};
