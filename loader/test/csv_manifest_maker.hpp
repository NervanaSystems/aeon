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

#include <string>
#include <vector>

class manifest_maker
{
public:
    manifest_maker(uint32_t num_records, std::vector<uint32_t> sizes);
    manifest_maker();
    ~manifest_maker();

    std::string get_manifest_name();

    std::string tmp_filename();
    std::string tmp_zero_file(uint32_t size);
    std::string tmp_manifest_file(uint32_t num_records, std::vector<uint32_t> sizes);
    std::string tmp_manifest_file_with_invalid_filename();
    std::string tmp_manifest_file_with_ragged_fields();
    std::string tmp_file_repeating(uint32_t size, uint32_t x);
    void remove_files();

    std::string manifest_name;

    std::vector<std::string> tmp_filenames;
};
