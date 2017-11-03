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

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "util.hpp"

namespace nervana
{
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

    constexpr uint32_t FOURCC(char a, char b, char c, char d)
    {
        return (a | (b << 8) | (c << 16) | (d << 24));
    }
    class wavefile_exception : public std::runtime_error
    {
    public:
        wavefile_exception(const std::string& msg)
            : runtime_error(msg.c_str())
        {
        }
    };

    class signal_generator
    {
    public:
        virtual ~signal_generator() {}
        virtual int16_t operator()(float t) const = 0;
    };

    class sinewave_generator : public signal_generator
    {
    public:
        sinewave_generator(float frequency, int16_t amplitude = INT16_MAX)
            : m_frequency(frequency)
            , m_amplitude(amplitude)
        {
        }

        int16_t operator()(float t) const override
        {
            return static_cast<int16_t>(sin(m_frequency * t) * m_amplitude);
        }

    private:
        float   m_frequency;
        int16_t m_amplitude;
    };

    class wav_data
    {
    public:
        wav_data(const signal_generator& sigptr, int duration_ss, int rate, bool is_stereo)
            : _sample_rate(rate)
        {
            data.create(duration_ss * rate, is_stereo ? 2 : 1, CV_16SC1);
            for (int n = 0; n < data.rows; ++n)
            {
                float t = 2.0 * CV_PI * n / static_cast<float>(rate);
                for (int c = 0; c < data.cols; ++c)
                {
                    data.at<int16_t>(n, c) = sigptr(t);
                }
            }
        }

        wav_data(const char* buf, uint32_t bufsize);

        void dump(std::ostream& ostr = std::cout);
        void write_to_file(std::string filename);
        void write_to_buffer(char* buf, uint32_t bufsize);

        cv::Mat&                get_data() { return data; }
        char**                  get_raw_data() { return (char**)&(data.data); }
        inline uint32_t         nbytes() { return data.total() * data.elemSize(); }
        inline uint32_t         nsamples() { return data.rows; }
        inline int32_t          sample_rate() { return _sample_rate; }
        static constexpr size_t HEADER_SIZE =
            sizeof(RiffMainHeader) + sizeof(FmtHeader) + sizeof(DataHeader);

        static constexpr int WAVE_FORMAT_PCM        = 0x0001;
        static constexpr int WAVE_FORMAT_IEEE_FLOAT = 0x0003;
        static constexpr int WAVE_FORMAT_EXTENSIBLE = 0xfffe;

    private:
        void wav_assert(bool cond, const std::string& msg)
        {
            if (!cond)
            {
                throw wavefile_exception(msg);
            }
        }

        void write_header(char* buf, uint32_t bufsize);
        void write_data(char* buf, uint32_t bufsize);

        cv::Mat data;
        int32_t _sample_rate;
    };
}
