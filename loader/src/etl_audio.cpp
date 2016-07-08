#include "etl_audio.hpp"

using namespace std;
using namespace nervana;

shared_ptr<audio::params> audio::param_factory::make_params(std::shared_ptr<const decoded>)
{
    auto audio_stgs = shared_ptr<audio::params>(new audio::params());

    audio_stgs->add_noise             = _cfg->add_noise(_dre);
    audio_stgs->noise_index           = _cfg->noise_index(_dre);
    audio_stgs->noise_level           = _cfg->noise_level(_dre);
    audio_stgs->noise_offset_fraction = _cfg->noise_offset_fraction(_dre);
    audio_stgs->time_scale_fraction   = _cfg->time_scale_fraction(_dre);

    return audio_stgs;
}

std::shared_ptr<audio::decoded> audio::extractor::extract(const char* item, int itemSize)
{
    return make_shared<audio::decoded>(_codec->decode(item, itemSize));
}

audio::transformer::transformer(std::shared_ptr<const audio::config> config)
{
    _specmaker = make_shared<Specgram>(config);

    if (config->noise_index_file.empty()) {
        _noisemaker = make_shared<NoiseClips>(config->noise_index_file);
    }
}

audio::transformer::~transformer()
{
    _specmaker = nullptr;
    _noisemaker = nullptr;  // no-op if it was never initialized
}

std::shared_ptr<audio::decoded> audio::transformer::transform(
                                      std::shared_ptr<audio::params> params,
                                      std::shared_ptr<audio::decoded> decoded)
{
    if (_noisemaker != nullptr) {
        _noisemaker->addNoise(decoded->get_time_data(), params);
    }

    // convert from time domain to frequency domain into the freq mat
    _specmaker->generate(decoded->get_time_data(), decoded->get_freq_data());

    resize(decoded->get_freq_data(), params->time_scale_fraction);

    return decoded;
}


void audio::transformer::resize(Mat& img, float fx) {
    if (fx == 1.0f) {
        return;
    }
    Mat dst;
    cv::resize(img, dst, cv::Size(), fx, 1.0, (fx > 1.0) ? CV_INTER_CUBIC : CV_INTER_AREA);
    assert(img.rows == dst.rows);
    if (img.cols > dst.cols) {
        dst.copyTo(img(Range::all(), Range(0, dst.cols)));
        img(Range::all(), Range(dst.cols, img.cols)) = cv::Scalar::all(0);
    } else {
        dst(Range::all(), Range(0, img.cols)).copyTo(img);
    }
}
