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

#include <sstream>
#include <fstream>
#include <sys/stat.h>

#include "noise_clips.hpp"
#include "file_util.hpp"

using namespace std;
using namespace nervana;

noise_clips::noise_clips(const std::string noiseIndexFile, const std::string noiseRoot)
{
    if (!noiseIndexFile.empty())
    {
        load_index(noiseIndexFile, noiseRoot);
        load_data();
    }
}

noise_clips::~noise_clips()
{
    if (_bufLen != 0)
    {
        delete[] _buf;
    }
}

void noise_clips::load_index(const std::string& index_file, const std::string& root_dir)
{
    ifstream ifs(index_file);

    if (!ifs)
    {
        throw std::ios_base::failure("Could not open " + index_file);
    }

    string line;
    while (getline(ifs, line))
    {
        if (!root_dir.empty())
            line = file_util::path_join(root_dir, line);
        _noise_files.push_back(line);
    }

    if (_noise_files.empty())
    {
        throw std::runtime_error("No noise files provided in " + index_file);
    }
}

// From Factory, get add_noise, offset (frac), noise index, noise level
/** \brief Add noise to a sound waveform
*
*
*/
void noise_clips::addNoise(cv::Mat& wav_mat,
                           bool     add_noise,
                           uint32_t noise_index,
                           float    noise_offset_fraction,
                           float    noise_level)
{
    // No-op if we have no noise files or randomly not adding noise on this datum
    if (!add_noise || _noise_data.empty())
    {
        return;
    }

    // Assume a single channel with 16 bit samples for now.
    affirm(wav_mat.cols == 1, "wav samples more than one column");
    affirm(wav_mat.type() == CV_16SC1, "wav not 16 bit signed");

    // Collect enough noise data to cover the entire input clip.
    cv::Mat        noise_dst = cv::Mat::zeros(wav_mat.size(), wav_mat.type());
    const cv::Mat& noise_src = _noise_data[noise_index % _noise_data.size()];

    affirm(noise_src.type() == wav_mat.type(), "noise type does not match wav type");

    uint32_t src_offset = noise_src.rows * noise_offset_fraction;
    uint32_t src_left   = noise_src.rows - src_offset;
    uint32_t dst_offset = 0;
    uint32_t dst_left   = wav_mat.rows;

    while (dst_left > 0)
    {
        uint32_t copy_size = std::min(dst_left, src_left);
        cv::Rect src_roi   = cv::Rect(cv::Point2i(0, src_offset), cv::Size2i(1, copy_size));
        cv::Rect dst_roi   = cv::Rect(cv::Point2i(0, dst_offset), cv::Size2i(1, copy_size));

        noise_src(src_roi).copyTo(noise_dst(dst_roi));

        if (src_left > dst_left)
        {
            dst_left = 0;
        }
        else
        {
            dst_left -= copy_size;
            dst_offset += copy_size;
            src_left   = noise_src.rows;
            src_offset = 0; // loop around
        }
    }
    // Superimpose noise without overflowing (opencv handles saturation cast for non CV_32S)
    cv::addWeighted(wav_mat, 1.0f, noise_dst, noise_level, 0.0f, wav_mat);
}

void noise_clips::load_data()
{
    for (auto nfile : _noise_files)
    {
        int len = 0;
        read_noise(nfile, &len);
        _noise_data.push_back(read_audio_from_mem(_buf, len));
    }
}

void noise_clips::read_noise(std::string& noise_file, int* dataLen)
{
    struct stat stats;
    int         result = stat(noise_file.c_str(), &stats);
    if (result == -1)
    {
        throw std::runtime_error("noise_clips: Could not find " + noise_file);
    }

    off_t size = stats.st_size;
    if (_bufLen < size)
    {
        delete[] _buf;
        _buf    = new char[size + size / 8];
        _bufLen = size + size / 8;
    }

    std::ifstream ifs(noise_file, std::ios::binary);
    ifs.read(_buf, size);

    if (size == 0)
    {
        throw std::runtime_error("noise_clips:  Could not read " + noise_file);
    }
    *dataLen = size;
}
