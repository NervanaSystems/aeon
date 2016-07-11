/*
 * Copyright (c) 2001 Fabrice Bellard
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */
#pragma once

#include <string>
#include <vector>
#include <random>

#include <opencv2/core/core.hpp>

#include "dataset.hpp"
#include "util.hpp"
class signal_generator {
public:
    virtual int16_t operator() (float) = 0;
};

class sinewave_generator : public signal_generator {
public:
    sinewave_generator(float frequency, int16_t amplitude) :
    frequency(frequency), amplitude(amplitude)
    {}

    int16_t operator () (float t) override
    {
        return static_cast<int16_t>(sin(frequency * t) * amplitude);
    }

private:
    float frequency;
    int16_t amplitude;
};

class wav_data {
public:
    wav_data(signal_generator signal, int duration_ss, int sample_rate, bool is_stereo) :
    _sample_rate(sample_rate)
    {
        data.create(duration_ss * sample_rate, is_stereo ? 2 : 1, CV_16SC1);
        for (int n = 0; n < data.rows; ++n) {
            for (int c = 0; c < data.cols; ++c) {
                float t = n / static_cast<float>(sample_rate);
                data.at<int16_t>(n, c) = signal(t);
            }
        }
    }

    void write_to_file(std::string filename);

    inline int32_t sample_rate() { return _sample_rate; }
    inline int32_t channels() { return data.cols; }
    inline int16_t bit_depth() { return data.elemSize(); }
    inline int32_t nbytes() { return data.total() * bit_depth(); }
    inline int16_t block_align() {return bit_depth() * data.cols ;}
    inline int32_t bytes_per_second() {return sample_rate() * block_align() ;}

private:
    void write_header();
    void write_data();

    std::ofstream _ofs;
    cv::Mat       data;
    int32_t       _sample_rate;
};

class gen_audio : public dataset<gen_audio> {
public:
    gen_audio();

    std::vector<unsigned char> encode(float frequencyHz, int duration);
    void encode(const std::string& filename, float frequencyHz, int duration);
    void decode(const std::string& outfilename, const std::string& filename);

    static std::vector<std::string> get_codec_list();

private:
    std::vector<unsigned char> render_target( int datumNumber ) override;
    std::vector<unsigned char> render_datum( int datumNumber ) override;

    static const char* Encoder_GetNextCodecName();
    static const char* Encoder_GetFirstCodecName();

    const std::vector<std::string> vocab = {"a","and","the","quick","fox","cow","dog","blue",
        "black","brown","happy","lazy","skip","jumped","run","under","over","around"};

    std::minstd_rand0 r;
};


