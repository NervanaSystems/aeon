/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "manifest_builder.hpp"
#include "manifest_file.hpp"
#include "file_util.hpp"
#include "gen_image.hpp"
#include "base64.hpp"

using namespace std;
using namespace nervana;

stringstream& manifest_builder::create()
{
    m_stream.str("");
    if (m_sizes.size() > 0)
    {
        // all exements are text elements
        m_stream << manifest_file::get_metadata_char();
        for (size_t i = 0; i < m_sizes.size(); i++)
        {
            if (i > 0)
            {
                m_stream << manifest_file::get_delimiter();
            }
            m_stream << manifest_file::get_string_type_id();
        }
        m_stream << "\n";

        // now add the records
        for (size_t record_number = 0; record_number < m_record_count; record_number++)
        {
            for (size_t element_number = 0; element_number < m_sizes.size(); element_number++)
            {
                if (element_number > 0)
                {
                    m_stream << manifest_file::get_delimiter();
                }
                m_stream << record_number << ":" << element_number;
            }
            m_stream << "\n";
        }
    }
    else if (m_image_width > 0 && m_image_height > 0)
    {
        // first element image, second element label
        size_t rows = 8;
        size_t cols = 8;
        m_stream << manifest_file::get_metadata_char() << manifest_file::get_binary_type_id();
        m_stream << manifest_file::get_delimiter() << manifest_file::get_string_type_id();
        m_stream << endl;
        for (size_t record_number = 0; record_number < m_record_count; record_number++)
        {
            cv::Mat         mat = embedded_id_image::generate_image(rows, cols, record_number);
            vector<uint8_t> result;
            cv::imencode(".png", mat, result);
            vector<char> image_data = base64::encode((const char*)result.data(), result.size());
            m_stream << vector2string(image_data);
            m_stream << manifest_file::get_delimiter();
            m_stream << record_number;
            m_stream << "\n";
        }
    }
    else
    {
        throw invalid_argument("must set either sizes or image dimensions");
    }

    return m_stream;
}

manifest_builder& manifest_builder::record_count(size_t value)
{
    m_record_count = value;
    return *this;
}

manifest_builder& manifest_builder::sizes(const std::vector<size_t>& sizes)
{
    m_sizes = sizes;
    return *this;
}

manifest_builder& manifest_builder::image_width(size_t value)
{
    m_image_width = value;
    return *this;
}

manifest_builder& manifest_builder::image_height(size_t value)
{
    m_image_height = value;
    return *this;
}
