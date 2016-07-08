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

#include "media.hpp"
#include "etl_audio.hpp"

#include <sstream>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cmath>
#include <vector>

namespace nervana {
    namespace audio {
        class config;
    }
}

static_assert(sizeof(short) == 2, "Unsupported platform");

class Specgram {
public:
    Specgram(std::shared_ptr<const nervana::audio::config> params, int id);
    virtual ~Specgram();
    int generate(std::shared_ptr<RawMedia> in_time, cv::Mat& out_freq);

private:
    bool powerOfTwo(int num);
    void create_window(const std::string& window_type, uint32_t frame_length_tn);
    void applyWindow(cv::Mat& signal);
    int stridedSignal(std::shared_ptr<RawMedia> raw);
    double hzToMel(double freqInHz);
    double melToHz(double freqInMels);
    void linspace(double a, double b, int n, std::vector<double>& intervals);
    void fill_filterbanks(int filts, int ffts, double samplingRate, cv::Mat& filterbanks);
    void extractFeatures(cv::Mat& spectrogram, cv::Mat& features);

private:
    int                         _feature;
    // Maximum duration in milliseconds.
    int                         _clipDuration;
    // Window size and stride are in terms of samples.
    int                         _windowSize;
    int                         _stride;
    int                         _width;
    int                         _numFreqs;
    int                         _height;
    int                         _maxSignalSize;
    int                         _numFilts;
    int                         _numCepstra;

    char*                       _buf;
    int                         _bufSize;
    cv::Mat                     _window {0, 0, CV_32FC1};
    cv::Mat                     _fbank;
    constexpr static int        MAX_BYTES_PER_SAMPLE = 4;
};
