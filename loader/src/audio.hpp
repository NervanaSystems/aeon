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

#include <sys/stat.h>
#include <libgen.h>

#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include <opencv2/core/core.hpp>

#include "media.hpp"
#include "codec.hpp"
#include "specgram.hpp"

using std::vector;
using std::ifstream;
using cv::Mat;


class IndexElement {
public:
    IndexElement() {
    }

public:
    std::string                      _fileName;
    std::vector<std::string>         _targets;
};

class Index {
public:
    Index() : _maxTargetSize(0) {
    }

    virtual ~Index() {
        for (auto elem : _elements) {
            delete elem;
        }
    }

    void load(std::string& fileName, bool shuf = false) {
        ifstream ifs(fileName);
        if (!ifs) {
            stringstream ss;
            ss << "Could not open " << fileName;
            throw std::ios_base::failure(ss.str());
        }

        std::string line;
        // Ignore the header line.
        std::getline(ifs, line);
        while (std::getline(ifs, line)) {
            if (line[0] == '#') {
                // Ignore comments.
                continue;
            }
            addElement(line);
        }

        if (shuf == true) {
            shuffle();
        }

        if (_elements.size() == 0) {
            stringstream ss;
            ss << "Could not load index from " << fileName;
            throw std::runtime_error(ss.str());
        }
    }

    IndexElement* operator[] (int idx) {
        return _elements[idx];
    }

    uint size() {
        return _elements.size();
    }

private:
    void addElement(std::string& line) {
        IndexElement* elem = new IndexElement();
        std::istringstream ss(line);
        std::string token;
        std::getline(ss, token, ',');
        elem->_fileName = token;
        while (std::getline(ss, token, ',')) {
            elem->_targets.push_back(token);
        }

        // For now, restrict to a single target.
        assert(elem->_targets.size() <= 1);
        _elements.push_back(elem);
        if (elem->_targets.size() == 0) {
            return;
        }
        if (elem->_targets[0].size() > _maxTargetSize) {
            _maxTargetSize = elem->_targets[0].size();
        }
    }

    void shuffle() {
        std::srand(0);
        std::random_shuffle(_elements.begin(), _elements.end());
    }

public:
    vector<IndexElement*>       _elements;
    uint                        _maxTargetSize;
};

class AudioParams : public SignalParams {
};

class NoiseClips {
public:
    NoiseClips(AudioParams* params, Codec* codec)
    : _indexFile(params->_noiseIndexFile), _indexDir(params->_noiseDir),
      _buf(0), _bufLen(0),
      _clipIndex(0), _clipOffset(0) {
        loadIndex(_indexFile);
        loadData(codec);
    }

    virtual ~NoiseClips() {
        delete[] _buf;
        for (auto elem : _data) {
            delete elem;
        }
    }

    void addNoise(RawMedia* media, cv::RNG& rng) {
        if (rng(2) == 0) {
            // Augment half of the data examples.
            return;
        }
        // Assume a single channel with 16 bit samples for now.
        assert(media->size() == 1);
        assert(media->sampleSize() == 2);
        int sampleSize = media->sampleSize();
        int numSamples = media->numSamples();
        Mat data(1, numSamples, CV_16S, media->getBuf(0));
        Mat noise(1, numSamples, CV_16S);
        int left = numSamples;
        int offset = 0;
        // Collect enough noise data to cover the entire input clip.
        while (left > 0) {
            RawMedia* clipData = _data[_clipIndex];
            assert(clipData->sampleSize() == sampleSize);
            int clipSize = clipData->numSamples() - _clipOffset;
            Mat clip(1, clipSize , CV_16S,
                     clipData->getBuf(0) + sampleSize * _clipOffset);
            if (clipSize > left) {
                const Mat& src = clip(Range::all(), Range(0, left));
                const Mat& dst = noise(Range::all(), Range(offset, offset + left));
                src.copyTo(dst);
                left = 0;
                _clipOffset += left;
            } else {
                const Mat& dst = noise(Range::all(),
                                       Range(offset, offset + clipSize));
                clip.copyTo(dst);
                left -= clipSize;
                offset += clipSize;
                next(rng);
            }
        }
        // Superimpose noise without overflowing.
        Mat convData;
        data.convertTo(convData, CV_32F);
        Mat convNoise;
        noise.convertTo(convNoise, CV_32F);
        float noiseLevel = rng.uniform(0.f, 1.0f);
        convNoise *= noiseLevel;
        convData += convNoise;
        double min, max;
        cv::minMaxLoc(convData, &min, &max);
        if (-min > 0x8000) {
            convData *= 0x8000 / -min;
            cv::minMaxLoc(convData, &min, &max);
        }
        if (max > 0x7FFF) {
            convData *= 0x7FFF / max;
        }
        convData.convertTo(data, CV_16S);
    }

private:
    void next(cv::RNG rng) {
        _clipIndex++;
        if (_clipIndex != _data.size()) {
            _clipOffset = 0;
        } else {
            // Wrap around.
            _clipIndex = 0;
            // Start at a random offset.
            _clipOffset = rng(_data[0]->numSamples());
        }
    }

    void loadIndex(std::string& indexFile) {
        _index.load(indexFile, true);
    }

    void loadData(Codec* codec) {
        for (uint i = 0; i < _index.size(); i++) {
            std::string& fileName = _index[i]->_fileName;
            int len = 0;
            readFile(fileName, &len);
            if (len == 0) {
                stringstream ss;
                ss << "Could not read " << fileName;
                throw std::runtime_error(ss.str());
            }
            RawMedia* raw = codec->decode(_buf, len);
            _data.push_back(new RawMedia(*raw));
        }
    }

    void readFile(std::string& fileName, int* dataLen) {
        std::string path;
        if (fileName[0] == '/') {
            path = fileName;
        } else {
            path = _indexDir + '/' + fileName;
        }
        struct stat stats;
        int result = stat(path.c_str(), &stats);
        if (result == -1) {
            stringstream ss;
            ss << "Could not find " << path;
            throw std::runtime_error(ss.str());
        }
        off_t size = stats.st_size;
        if (_bufLen < size) {
            resize(size + size / 8);
        }
        std::ifstream ifs(path, std::ios::binary);
        ifs.read(_buf, size);
        *dataLen = size;
    }

    void resize(int newLen) {
        delete[] _buf;
        _buf = new char[newLen];
        _bufLen = newLen;
    }

private:
    std::string                 _indexFile;
    std::string                 _indexDir;
    vector<RawMedia*>           _data;
    Index                       _index;
    char*                       _buf;
    int                         _bufLen;
    uint                        _clipIndex;
    int                         _clipOffset;
};

class Audio : public Media {
public:
    Audio(AudioParams *params, int id)
    : _params(params), _noiseClips(0), _loadedNoise(false) {
        _rng.state = 0;
        _codec = new Codec(params);
        _specgram = new Specgram(params, id);
        if (params->_noiseIndexFile != 0) {
            if ((id == 0) && (params->_noiseClips == 0)) {
                params->_noiseClips = new NoiseClips(params, _codec);
                _loadedNoise = true;
            }
            _noiseClips = reinterpret_cast<NoiseClips*>(params->_noiseClips);
            assert(_noiseClips != 0);
        }
    }

    virtual ~Audio() {
        if (_loadedNoise == true) {
            delete _noiseClips;
        }
        delete _specgram;
        delete _codec;
    }

    void dumpToBin(char* filename, RawMedia* audio, int idx) {
        // dump audio file in `audio` at buffer index `idx` into bin
        // file at `filename`.  Assumes buffer is already scaled to
        // int16.  A simple python script can convert this file into
        // a wav file: loader/test/raw_to_wav.py
        FILE *file = fopen(filename, "wb");
        fwrite(audio->getBuf(idx), audio->numSamples(), 2, file);
        fclose(file);
    }

    void transform(char* item, int itemSize, char* buf, int bufSize, int* meta) {
        RawMedia* raw = _codec->decode(item, itemSize);
        if (_noiseClips != 0) {
            _noiseClips->addNoise(raw, _rng);
        }
        int len = _specgram->generate(raw, buf, bufSize);
        if (meta != 0) {
            *meta = len;
        }
    }

    void ingest(char** dataBuf, int* dataBufLen, int* dataLen) {
    }

private:
    AudioParams*                _params;
    Codec*                      _codec;
    Specgram*                   _specgram;
    NoiseClips*                 _noiseClips;
    bool                        _loadedNoise;
    cv::RNG                     _rng;
};
