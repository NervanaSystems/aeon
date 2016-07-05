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

#include "audio_params.hpp"
#include "codec.hpp"
#include "noise_clips.hpp"

class Codec;

class IndexElement {
public:
    IndexElement();

public:
    std::string                      _fileName;
    std::vector<std::string>         _targets;
};

class Index {
public:
    Index();
    virtual ~Index();

    void load(std::string& fileName, bool shuf = false);
    IndexElement* operator[] (int idx);
    uint size();

private:
    void addElement(std::string& line);
    void shuffle();

public:
    vector<IndexElement*>       _elements;
    uint                        _maxTargetSize;
};

class NoiseClips {
public:
    NoiseClips(char* _noiseIndexFile, char* _noiseDir, Codec* codec);
    virtual ~NoiseClips();

    void addNoise(std::shared_ptr<RawMedia> media, cv::RNG& rng);

private:
    void next(cv::RNG rng);
    void loadIndex(std::string& indexFile);
    void loadData(Codec* codec);
    void readFile(std::string& fileName, int* dataLen);
    void resize(int newLen);

private:
    std::string                 _indexFile;
    std::string                 _indexDir;
    vector<std::shared_ptr<RawMedia>> _data;
    Index                       _index;
    char*                       _buf;
    int                         _bufLen;
    uint                        _clipIndex;
    int                         _clipOffset;
};
