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

#include <sstream>
#include <math.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>

static_assert(sizeof(short) == 2, "Unsupported platform");

namespace nervana
{
    class specgram;
}

class nervana::specgram
{
public:
    specgram()          = delete;
    virtual ~specgram() = delete;
    static void wav_to_specgram(const cv::Mat& wav_mat,
                                const int      frame_length_tn,
                                const int      frame_stride_tn,
                                const int      max_time_steps,
                                const cv::Mat& window,
                                cv::Mat&       specgram);

    static void specgram_to_cepsgram(const cv::Mat& specgram,
                                     const cv::Mat& filter_bank,
                                     cv::Mat&       cepsgram);

    static void cepsgram_to_mfcc(const cv::Mat& cepsgram, const int num_cepstra, cv::Mat& mfcc);

    static void create_window(const std::string& window_type, const int n, cv::Mat& win);

    static void create_filterbanks(const int num_filters,
                                   const int fftsz,
                                   const int sample_freq_hz,
                                   cv::Mat&  fbank);

private:
    static inline double hz_to_mel(double freq_hz)
    {
        return 2595 * std::log10(1 + freq_hz / 700.0);
    }
    static inline double mel_to_hz(double freq_mel)
    {
        return 700 * (std::pow(10, freq_mel / 2595.0) - 1);
    }
};
