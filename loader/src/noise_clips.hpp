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

class NoiseClips {
public:
    NoiseClips(const std::string noiseIndexFile);
    virtual ~NoiseClips();
    void addNoise(std::shared_ptr<RawMedia> media, std::shared_ptr<nervana::audio::params> prm);

private:
    void load_index(std::string& index_file);
    void load_data(std::shared_ptr<Codec> codec);
    void read_noise(std::string& noise_file, int* dataLen);

private:
    std::string                            _noise_dir {""};
    std::vector<std::string>               _noise_files;
    std::vector<std::shared_ptr<RawMedia>> _noise_data;
    char*                                  _buf = 0;
    int                                    _bufLen = 0;
};
