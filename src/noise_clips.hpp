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

#include <opencv2/core/core.hpp>
#include "util.hpp"

namespace nervana
{
    class noise_clips;
}

class nervana::noise_clips
{
public:
    noise_clips(const std::string noiseIndexFile, const std::string noiseRoot);
    virtual ~noise_clips();
    void addNoise(cv::Mat& wav_mat,
                  bool     add_noise,
                  uint32_t noise_index,
                  float    noise_offset_fraction,
                  float    noise_level);

private:
    void load_index(const std::string& index_file, const std::string& root_dir);
    void load_data();
    void read_noise(std::string& noise_file, int* dataLen);

private:
    std::vector<cv::Mat>     _noise_data;
    std::vector<std::string> _noise_files;
    char*                    _buf    = 0;
    int                      _bufLen = 0;
};
