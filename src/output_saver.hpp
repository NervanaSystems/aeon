/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <atomic>

#include "opencv2/opencv.hpp"

#include "etl_image.hpp"
#include "buffer_batch.hpp"
#include "json.hpp"

namespace nervana
{
    class output_saver;
};

class nervana::output_saver final
{
public:
    output_saver() = default;
    output_saver(const std::string& output_dir)
        : m_output_dir(output_dir)
    {
    }
    void save(const nervana::fixed_buffer_map* batch);
    void save(const cv::Mat& image, std::shared_ptr<augment::image::params> img_xform);

private:
    std::string get_debug_file_id();
    void save(const cv::Mat& image);
    std::string get_filename();
    std::string get_filename(const std::string& directory);

    std::string      m_output_dir;
    std::atomic_uint m_index{0};
};
