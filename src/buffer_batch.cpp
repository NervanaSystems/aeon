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

#include <algorithm>
#include <stdexcept>

#include "buffer_batch.hpp"
#include "log.hpp"

using namespace std;
using namespace nervana;

buffer_fixed_size_elements::buffer_fixed_size_elements(const shape_type& shp_tp,
                                                       size_t            batch_size,
                                                       bool              pinned)
    : m_shape_type{shp_tp}
    , m_size{m_shape_type.get_byte_size() * batch_size}
    , m_batch_size{batch_size}
    , m_stride{m_shape_type.get_byte_size()}
    , m_pinned{pinned}
{
    allocate();
}

buffer_fixed_size_elements::buffer_fixed_size_elements(const buffer_fixed_size_elements& rhs)
    : m_data{nullptr}
    , m_shape_type{rhs.m_shape_type}
    , m_size{rhs.m_size}
    , m_batch_size{rhs.m_batch_size}
    , m_stride{rhs.m_stride}
    , m_pinned{rhs.m_pinned}
{
    allocate();
    memcpy(m_data, rhs.m_data, m_size);
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

cv::Mat buffer_fixed_size_elements::get_item_as_mat(size_t index, bool channel_major) const
{
    std::vector<int> sizes;
    size_t           channels;
    for (auto& d : m_shape_type.get_shape())
    {
        sizes.push_back(static_cast<int>(d));
    }
    int ndims = static_cast<int>(sizes.size());

    if (channel_major)
    {
        channels = sizes[0];
    }
    else
    {
        ndims -= 1;
        channels = sizes.back();
        sizes.pop_back();
    }

    cv::Mat ret(ndims,
                &sizes[0],
                CV_MAKETYPE(m_shape_type.get_otype().get_cv_type(), channels),
                (void*)&m_data[index * m_stride]);
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

void buffer_fixed_size_elements::allocate()
{
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

void fixed_buffer_map::copy(fixed_buffer_map& src, size_t src_index, size_t dst_index, size_t count)
{
    for (auto name: m_names)
    {
        buffer_fixed_size_elements* src_fbm =    src[name];
        buffer_fixed_size_elements* dst_fbm = m_data[name];
        char* p_src = src_fbm->get_item(src_index);
        char* p_dst = dst_fbm->get_item(dst_index);

        if ((count + src_index > src_fbm->get_item_count())
          ||(count + dst_index > dst_fbm->get_item_count()))
           throw invalid_argument("buffer_fixed_size: count out-of-range");

        memcpy(p_dst, p_src, count * src_fbm->get_stride());
    }
}
