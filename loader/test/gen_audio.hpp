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

namespace nervana {
#define MKFOURCC(a, b, c, d) ((uint32_t)(a) | (b) << 8 | (c) << 16 | (d) << 24)

    constexpr uint32_t FOURCC ( char a, char b, char c, char d )
    {
        return (a |  (b << 8) | (c << 16) | (d << 24));
    }
}
#pragma pack(1)
struct RiffMainHeader
{
    uint32_t dwRiffCC;
    uint32_t dwRiffLen;
    uint32_t dwWaveID;
};
struct FmtHeader
{
    uint32_t dwFmtCC;
    uint32_t dwFmtLen;
    uint16_t hwFmtTag;
    uint16_t hwChannels;
    uint32_t dwSampleRate;
    uint32_t dwBytesPerSec;
    uint16_t hwBlockAlign;
    uint16_t hwBitDepth;
};
struct DataHeader
{
    uint32_t dwDataCC;
    uint32_t dwDataLen;
};
#pragma pack()

class wavefile_exception: public std::runtime_error {
public:
    wavefile_exception (const std::string& msg) :
    runtime_error(msg.c_str())
    {}
};

class signal_generator {
public:
    virtual ~signal_generator() {}
    virtual int16_t operator() (float t) = 0;
};

class sinewave_generator : public signal_generator {
public:
    sinewave_generator(float frequency, int16_t amplitude) :
    frequency(frequency), amplitude(amplitude)
    {}

    int16_t operator() (float t) override
    {
        return static_cast<int16_t>(sin(frequency * t) * amplitude);
    }

private:
    float frequency;
    int16_t amplitude;
};

class wav_data {
public:
    wav_data(std::shared_ptr<signal_generator> sigptr,
             int duration_ss, int rate, bool is_stereo) :
    _sample_rate(rate)
    {
        data.create(duration_ss * rate, is_stereo ? 2 : 1, CV_16SC1);
        for (int n = 0; n < data.rows; ++n) {
            float t = 2.0 * CV_PI * n / static_cast<float>(rate);
            for (int c = 0; c < data.cols; ++c) {
                data.at<int16_t>(n, c) = (*sigptr)(t);
            }
        }
    }

    wav_data(char *buf, uint32_t bufsize);

    void dump(std::ostream & ostr = std::cout);
    void write_to_file(std::string filename);
    void write_to_buffer(char *buf, uint32_t bufsize);

    inline uint32_t nbytes() { return data.total() * data.elemSize(); }

    static constexpr size_t HEADER_SIZE = sizeof(RiffMainHeader) + sizeof(FmtHeader) + sizeof(DataHeader);

    static constexpr int WAVE_FORMAT_PCM = 0x0001;
    static constexpr int WAVE_FORMAT_IEEE_FLOAT = 0x0003;
    static constexpr int WAVE_FORMAT_EXTENSIBLE = 0xfffe;

private:
    void wav_assert(bool cond, const std::string &msg)
    {
        if (!cond)
        {
            throw wavefile_exception(msg);
        }
    }

    void write_header(char* buf, uint32_t bufsize);
    void write_data(char* buf, uint32_t bufsize);

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


