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

#ifdef PYTHON_FOUND
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#else
#define NPY_INT8 0
#define NPY_UINT8 0
#define NPY_INT16 0
#define NPY_UINT16 0
#define NPY_INT32 0
#define NPY_UINT32 0
#define NPY_FLOAT32 0
#define NPY_FLOAT64 0
#endif

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
    static bool is_valid_type(const std::string& s)
    {
        return all_outputs.find(s) != all_outputs.end();
    }

    bool operator==(const output_type& other) const
    {
        if (m_size == other.m_size && m_cv_type == other.m_cv_type &&
            m_np_type == other.m_np_type && m_tp_name == other.m_tp_name)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool operator!=(const output_type& other) const { return !(*this == other); }
    std::string                        m_tp_name;
    int                                m_np_type;
    int                                m_cv_type;
    size_t                             m_size;
};

class nervana::shape_type
{
public:
    shape_type(const std::vector<size_t>& shape, const output_type& otype)
        : m_shape{shape}
        , m_otype{otype}
    {
        m_byte_size = static_cast<size_t>(m_otype.get_size() * get_element_count());
    }

    size_t get_element_count() const
    {
        return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<uint32_t>());
    }
    size_t                     get_byte_size() const { return m_byte_size; }
    const std::vector<size_t>& get_shape() const { return m_shape; }
    const output_type&         get_otype() const { return m_otype; }
    void set_names(const std::vector<std::string>& names)
    {
        if (m_shape.size() != names.size())
        {
            throw std::runtime_error(
                "naming shape dimensions: number of names does not match number of dimensions");
        }
        else
        {
            m_names = names;
        }
    }

    std::vector<std::string> get_names() const
    {
        if (m_names.size() == 0)
        {
            std::vector<std::string> res;
            for (int i = 0; i < m_shape.size(); ++i)
            {
                res.push_back(std::to_string(i));
            }
            return res;
        }
        else
        {
            return m_names;
        }
    }

    bool operator==(const shape_type& other) const
    {
        if (m_byte_size == other.m_byte_size && m_otype == other.m_otype &&
            m_shape == other.m_shape && m_names == other.m_names)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    bool operator!=(const shape_type& other) const { return !(*this == other); }
private:
    std::vector<size_t>      m_shape;
    output_type              m_otype;
    size_t                   m_byte_size;
    std::vector<std::string> m_names;
};
