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

#include <opencv2/core/core.hpp>

#include "codec.hpp"
#include "params.hpp"

class Codec;

class NoiseConfig : public nervana::json_config_parser {
public:
    std::string              noise_dir   {""};
    std::vector<std::string> noise_files {};
    NoiseConfig() {}

    NoiseConfig(nlohmann::json js)
    {
        parse_value(noise_dir,   "noise_dir",   js, mode::OPTIONAL);
        parse_value(noise_files, "noise_files", js, mode::OPTIONAL);
    }
};

class NoiseClips {
public:
    NoiseClips(const std::string noiseIndexFile);
    virtual ~NoiseClips();
    void addNoise(std::shared_ptr<RawMedia> media,
                  bool add_noise,
                  uint32_t noise_index,
                  float noise_offset_fraction,
                  float noise_level);


private:
    void load_index(const std::string& index_file);
    void load_data(std::shared_ptr<Codec> codec);
    void read_noise(std::string& noise_file, int* dataLen);

private:
    NoiseConfig _cfg;
    std::vector<std::shared_ptr<RawMedia>> _noise_data;
    char*                                  _buf = 0;
    int                                    _bufLen = 0;
};
