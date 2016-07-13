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

audio::transformer::transformer(std::shared_ptr<audio::config> config) :
_cfg(config)
{
    specgram::create_window(_cfg->window_type, _cfg->frame_length_tn, _window);
    specgram::create_filterbanks(_cfg->num_filters, _cfg->frame_length_tn, _cfg->sample_freq_hz,
                                 _filterbank);
    _noisemaker = make_shared<NoiseClips>(_cfg->noise_index_file);
}

audio::transformer::~transformer()
{
    _noisemaker = nullptr;
    _cfg        = nullptr;
}

std::shared_ptr<audio::decoded> audio::transformer::transform(
                                      std::shared_ptr<audio::params> params,
                                      std::shared_ptr<audio::decoded> decoded)
{
    _noisemaker->addNoise(decoded->get_time_data(),
                          params->add_noise,
                          params->noise_index,
                          params->noise_offset_fraction,
                          params->noise_level); // no-op if no noise files

    // convert from time domain to frequency domain into the freq mat
    specgram::wav_to_specgram(decoded->get_time_data(),
                              _cfg->frame_length_tn,
                              _cfg->frame_stride_tn,
                              _cfg->time_steps,
                              _window,
                              decoded->get_freq_data());
    if (_cfg->feature_type != "specgram") {
        cv::Mat tmpmat;
        specgram::specgram_to_cepsgram(decoded->get_freq_data(), _filterbank, tmpmat);
        if (_cfg->feature_type == "mfcc") {
            specgram::cepsgram_to_mfcc(tmpmat, _cfg->num_cepstra, decoded->get_freq_data());
        } else {
            decoded->get_freq_data() = tmpmat;
        }
    }
    resize(decoded->get_freq_data(), params->time_scale_fraction);

    return decoded;
}


void audio::transformer::resize(cv::Mat& img, float fx) {
    if (fx == 1.0f) {
        return;
    }
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(), fx, 1.0, (fx > 1.0) ? CV_INTER_CUBIC : CV_INTER_AREA);
    assert(img.rows == dst.rows);
    if (img.cols > dst.cols) {
        dst.copyTo(img(cv::Range::all(), cv::Range(0, dst.cols)));
        img(cv::Range::all(), cv::Range(dst.cols, img.cols)) = cv::Scalar::all(0);
    } else {
        dst(cv::Range::all(), cv::Range(0, img.cols)).copyTo(img);
    }
}
