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

#pragma once

#include <string>
#include <vector>
#include <sstream>

class manifest_builder
{
public:
    manifest_builder& record_count(size_t value);
    manifest_builder& sizes(const std::vector<size_t>& sizes);
    manifest_builder& image_width(size_t value);
    manifest_builder& image_height(size_t value);

    std::stringstream& create();

private:
    std::vector<size_t> m_sizes;
    size_t              m_image_width  = 0;
    size_t              m_image_height = 0;
    size_t              m_record_count = 0;
    std::stringstream   m_stream;
};
