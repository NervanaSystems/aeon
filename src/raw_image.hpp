/*
 Copyright 2017 Intel(R) Nervana(TM)
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

#include <iostream>
#include <string>
#include <memory>

#include <opencv2/core/core.hpp>

namespace nervana
{
    class raw_image;
}

class nervana::raw_image
{
public:
    raw_image(const std::string& filename);
    raw_image(std::istream& in_stream);

    void write(const std::string& filename);
    void write(std::ostream& out_stream);

    static raw_image from_cvmat(cv::Mat&);
    cv::Mat          to_cvmat();

    size_t size() const;

private:
    raw_image();
    void read(std::istream& in);

    bool to_bool(const std::string&) const;

    std::shared_ptr<char> m_data;
    size_t                m_width;
    size_t                m_height;
    size_t                m_channels;
    size_t                m_bitwidth;
    bool                  m_is_float;
    bool                  m_is_big_endian;
};
