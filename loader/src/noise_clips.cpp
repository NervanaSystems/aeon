#include "noise_clips.hpp"

#include <sstream>
#include <fstream>

#include <sys/stat.h>

using namespace std;

NoiseClips::NoiseClips(const std::string noiseIndexFile)
{
    if (!noiseIndexFile.empty()) {
        load_index(noiseIndexFile);
        load_data();
    }
}

NoiseClips::~NoiseClips()
{
    if (_bufLen != 0) {
        delete[] _buf;
    }
}

void NoiseClips::load_index(const std::string& index_file)
{
    ifstream ifs(index_file);

    if (!ifs) {
        throw std::ios_base::failure("Could not open " + index_file);
    }

    string line;
    while(getline(ifs, line)) {
        _noise_files.push_back(line);
    }

    if (_noise_files.empty()) {
        throw std::runtime_error("No noise files provided in " + index_file);
    }
}

// From Factory, get add_noise, offset (frac), noise index, noise level
void NoiseClips::addNoise(cv::Mat& wav_mat,
                          bool add_noise,
                          uint32_t noise_index,
                          float noise_offset_fraction,
                          float noise_level)
{
    // No-op if we have no noise files or randomly not adding noise on this datum
    if (!add_noise || _noise_data.empty()) {
        return;
    }

    // Assume a single channel with 16 bit samples for now.
    assert(wav_mat.cols == 1);
    assert(wav_mat.type() == CV_16SC1);
    cv::Mat noise = cv::Mat::zeros(wav_mat.size(), wav_mat.type());

    // Collect enough noise data to cover the entire input clip.
    const cv::Mat& noise_src = _noise_data[ noise_index % _noise_data.size() ]->get_data();

    assert(noise_src.type() == wav_mat.type());

    uint32_t src_offset = noise_src.rows * noise_offset_fraction;
    uint32_t src_left = noise_src.rows - src_offset;
    uint32_t dst_offset = 0;
    uint32_t dst_left = wav_mat.rows;
    while (dst_left > 0) {
        uint32_t copy_size = std::min(dst_left, src_left);

        const cv::Mat& src = noise_src(cv::Range::all(),
                                       cv::Range(src_offset, src_offset + copy_size));
        const cv::Mat& dst = noise(cv::Range::all(),
                                       cv::Range(dst_offset, dst_offset + copy_size));
        src.copyTo(dst);

        if (src_left > dst_left) {
            dst_left = 0;
        } else {
            dst_left -= copy_size;
            dst_offset += copy_size;
            src_left = noise_src.rows;
            src_offset = 0; // loop around
        }
    }
    // Superimpose noise without overflowing (opencv handles saturation cast for non CV_32S)
    cv::addWeighted(wav_mat, 1.0f, noise, noise_level, 0.0f, wav_mat);

}

void NoiseClips::load_data() {
    for(auto nfile: _noise_files) {
        int len = 0;
        read_noise(nfile, &len);
        _noise_data.push_back(make_shared<nervana::wav_data>(_buf, len));
    }
}

void NoiseClips::read_noise(std::string& noise_file, int* dataLen) {

    struct stat stats;
    int result = stat(noise_file.c_str(), &stats);
    if (result == -1) {
        throw std::runtime_error("Could not find " + noise_file);
    }

    off_t size = stats.st_size;
    if (_bufLen < size) {
        delete[] _buf;
        _buf = new char[size + size / 8];
        _bufLen = size + size / 8;
    }

    std::ifstream ifs(noise_file, std::ios::binary);
    ifs.read(_buf, size);

    if (size == 0) {
        throw std::runtime_error("Could not read " + noise_file);
    }
    *dataLen = size;
}
