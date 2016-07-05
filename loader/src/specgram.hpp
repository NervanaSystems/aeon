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

enum FeatureType {
    SPECGRAM    = 0,
    MFSC        = 1,
    MFCC        = 2,
};

static_assert(sizeof(short) == 2, "Unsupported platform");

class Specgram {
public:
    Specgram(std::shared_ptr<const nervana::audio::config> params, int id);
    virtual ~Specgram();
    int generate(std::shared_ptr<RawMedia> raw, char* buf, int bufSize);

private:
    void randomize(cv::Mat& img);
    void resize(cv::Mat& img, float fx);
    bool powerOfTwo(int num);
    void none(int);
    void hann(int steps);
    void blackman(int steps);
    void hamming(int steps);
    void bartlett(int steps);
    void createWindow(int windowType);
    void applyWindow(cv::Mat& signal);
    int stridedSignal(std::shared_ptr<RawMedia> raw);
    double hzToMel(double freqInHz);
    double melToHz(double freqInMels);
    std::vector<double> linspace(double a, double b, int n);
    cv::Mat getFilterbank(int filts, int ffts, double samplingRate);
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
    int                         _samplingFreq;
    int                         _maxSignalSize;
    int                         _numFilts;
    int                         _numCepstra;
    float                       _scaleBy;
    float                       _scaleMin;
    float                       _scaleMax;
    char*                       _buf;
    int                         _bufSize;
    cv::Mat*                    _image;
    cv::Mat*                    _window;
    cv::Mat                     _fbank;
    cv::RNG                     _rng;
    constexpr static int        MAX_SAMPLE_SIZE = 4;
};
