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
#include "buffer_batch.hpp"
#include <algorithm>

using namespace std;
using namespace nervana;

buffer_fixed_size_elements::buffer_fixed_size_elements(const shape_type& shp_tp,
                                                       size_t            batch_size,
                                                       bool              pinned)
    : m_pinned{pinned}
    , m_shape_type{shp_tp}
{
    m_size       = m_shape_type.get_byte_size() * batch_size;
    m_batch_size = batch_size;
    m_stride     = m_shape_type.get_byte_size();
#if HAS_GPU
    if (m_pinned)
    {
        CUresult status = cuMemAllocHost((void**)&m_data, m_size);
        if (status != CUDA_SUCCESS)
        {
            throw std::bad_alloc();
        }
    }
    else
    {
        m_data = new char[m_size];
    }
#else
    m_data = new char[m_size];
#endif
}

char* buffer_fixed_size_elements::get_item(size_t index)
{
    size_t offset = index * m_stride;
    if (index >= (int)m_batch_size)
    {
        throw invalid_argument("buffer_fixed_size: index out-of-range");
    }
    return &m_data[offset];
}

cv::Mat buffer_fixed_size_elements::get_item_as_mat(size_t index)
{
    std::vector<int> sizes;
    for (auto& d : m_shape_type.get_shape())
    {
        sizes.push_back(static_cast<int>(d));
    }
    int ndims = static_cast<int>(sizes.size());

    cv::Mat ret(
        ndims, &sizes[0], m_shape_type.get_otype().get_cv_type(), (void*)&m_data[index * m_stride]);
    return ret;
}

const char* buffer_fixed_size_elements::get_item(size_t index) const
{
    size_t offset = index * m_stride;
    if (index >= (int)m_batch_size)
    {
        throw invalid_argument("buffer_fixed_size: index out-of-range");
    }
    return &m_data[offset];
}

void buffer_fixed_size_elements::allocate(const shape_type& shp_tp, size_t batch_size, bool pinned)
{
    m_size       = m_shape_type.get_byte_size() * batch_size;
    m_batch_size = batch_size;
    m_stride     = m_shape_type.get_byte_size();
#if HAS_GPU
    if (m_pinned)
    {
        CUresult status = cuMemAllocHost((void**)&m_data, m_size);
        if (status != CUDA_SUCCESS)
        {
            throw std::bad_alloc();
        }
    }
    else
    {
        m_data = new char[m_size];
    }
#else
    m_data = new char[m_size];
#endif
}

buffer_fixed_size_elements::~buffer_fixed_size_elements()
{
#if HAS_GPU
    if (m_pinned)
    {
        cuMemFreeHost(m_data);
    }
    else
    {
        delete[] m_data;
    }
#else
    delete[] m_data;
#endif
}
