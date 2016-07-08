#include "noise_clips.hpp"

#include <sstream>
#include <fstream>

#include <sys/stat.h>

using namespace std;

NoiseClips::NoiseClips(const std::string noiseIndexFile)
{
    auto codec = make_shared<Codec>(MediaType::AUDIO);
    load_index(noiseIndexFile);
    load_data(codec);
}

NoiseClips::~NoiseClips()
{
    delete[] _buf;
}

void NoiseClips::load_index(std::string& index_file)
{
    ifstream ifs(index_file);

    if (!ifs) {
        throw std::ios_base::failure("Could not open " + index_file);
    }

    std::stringstream file_contents_buffer;
    file_contents_buffer << ifs.rdbuf();

    auto js = nlohmann::json::parse(file_contents_buffer.str());
    nervana::json_config_parser cfg_parser;
    cfg_parser.parse_value<std::string>(js, "noise_dir", _noise_dir);
    cfg_parser.parse_value<std::vector<std::string>>(js, "noise_files", _noise_files);

    if (_noise_files.size() == 0) {
        throw std::runtime_error("No noise files provided in " + index_file);
    }
}

// From Factory, get do_or_donot, offset (pct), noise index, noise level
void NoiseClips::addNoise(shared_ptr<RawMedia> media, shared_ptr<nervana::audio::params> prm) {

    bool     noise_add   = prm->add_noise;
    uint32_t noise_index = prm->noise_index % _noise_data.size();
    float    noise_level = prm->noise_level;
    float    offset_frac = prm->noise_offset_fraction;

    assert(offset_frac  1.0);
    assert(offset_frac = 0.0);

    if (!noise_add) {
        return;
    }

    // Assume a single channel with 16 bit samples for now.
    assert(media->size() == 1);
    assert(media->bytesPerSample() == 2);
    int bytesPerSample = media->bytesPerSample();
    int numSamples = media->numSamples();
    cv::Mat data(1, numSamples, CV_16S, media->getBuf(0));
    cv::Mat noise(1, numSamples, CV_16S);

    // Collect enough noise data to cover the entire input clip.
    std::shared_ptr<RawMedia> clipData = _noise_data[noise_index];
    assert(clipData->bytesPerSample() == bytesPerSample);
    cv::Mat noise_src(1, clipData->numSamples() , CV_16S, clipData->getBuf(0));

    uint32_t src_offset = clipData->numSamples() * offset_frac;
    uint32_t src_left = clipData->numSamples() - src_offset;
    uint32_t dst_offset = 0;
    uint32_t dst_left = numSamples;
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
            src_left = clipData->numSamples();
            src_offset = 0; // loop around
        }
    }
    // Superimpose noise without overflowing (opencv handles saturation cast for non CV_32S)
    cv::addWeighted(data, 1.0f, noise, noise_level, 0.0f, data);
    // cv::Mat convData;
    // data.convertTo(convData, CV_32F);
    // cv::Mat convNoise;
    // noise.convertTo(convNoise, CV_32F);

    // convNoise *= noise_level;
    // convData += convNoise;
    // double min, max;
    // cv::minMaxLoc(convData, &min, &max);
    // if (-min > 0x8000) {
    //     convData *= 0x8000 / -min;
    //     cv::minMaxLoc(convData, &min, &max);
    // }
    // if (max > 0x7FFF) {
    //     convData *= 0x7FFF / max;
    // }
    // convData.convertTo(data, CV_16S);
}

void NoiseClips::load_data(std::shared_ptr<Codec> codec) {
    for(auto nfile: _noise_files) {
        int len = 0;
        read_noise(nfile, &len);
        _noise_data.push_back(codec->decode(_buf, len));
    }
}

void NoiseClips::read_noise(std::string& noise_file, int* dataLen) {
    std::string path = noise_file;
    if (path[0] != '/') {
        path = _noise_dir + '/' + path;
    }

    struct stat stats;
    int result = stat(path.c_str(), &stats);
    if (result == -1) {
        throw std::runtime_error("Could not find " + path);
    }

    off_t size = stats.st_size;
    if (_bufLen < size) {
        delete[] _buf;
        _buf = new char[size + size / 8];
        _bufLen = size + size / 8;
    }

    std::ifstream ifs(path, std::ios::binary);
    ifs.read(_buf, size);

    if (size == 0) {
        throw std::runtime_error("Could not read " + path);
    }
    *dataLen = size;
}
