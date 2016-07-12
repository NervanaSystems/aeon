#include "etl_audio.hpp"

using namespace std;
using namespace nervana;

audio::config::config(nlohmann::json js) {
    // for now, all config params are required
    parse_value(_samplingFreq, "samplingFreq", js, mode::REQUIRED);
    parse_value(_clipDuration, "clipDuration", js, mode::REQUIRED);
    parse_value(_randomScalePercent, "randomScalePercent", js, mode::REQUIRED);
    parse_value(_numFilts, "numFilts", js, mode::REQUIRED);
    parse_value(_numCepstra, "numCepstra", js, mode::REQUIRED);
    parse_value(_noiseIndexFile, "noiseIndexFile", js, mode::REQUIRED);
    parse_value(_noiseDir, "noiseDir", js, mode::REQUIRED);
    parse_value(_windowSize, "windowSize", js, mode::REQUIRED);
    parse_value(_stride, "stride", js, mode::REQUIRED);
    parse_value(_width, "width", js, mode::REQUIRED);
    parse_value(_height, "height", js, mode::REQUIRED);
    parse_value(_window, "window", js, mode::REQUIRED);
    parse_value(_feature, "feature", js, mode::REQUIRED);
}

shared_ptr<audio::params> audio::param_factory::make_params(std::shared_ptr<const decoded>) {
    auto params = shared_ptr<audio::params>(new audio::params());

    params->_width = _cfg->_width;
    params->_height = _cfg->_height;

    return params;
}

audio::decoded::decoded(shared_ptr<RawMedia> raw)
    : _raw(raw) {
}

MediaType audio::decoded::get_type() {
    return MediaType::AUDIO;
}

size_t audio::decoded::getSize() {
    return _raw->numSamples();
}

audio::extractor::extractor(std::shared_ptr<const audio::config> config) {
    _codec = new Codec(config);
    avcodec_register_all();
}

audio::extractor::~extractor() {
    delete _codec;
}

std::shared_ptr<audio::decoded> audio::extractor::extract(const char* item, int itemSize) {
    return make_shared<audio::decoded>(_codec->decode(item, itemSize));
}

audio::transformer::transformer(std::shared_ptr<const audio::config> config)
    : _codec(0), _noiseClips(0), _state(0), _rng(config->_randomSeed) {
    _specgram = new Specgram(config, config->_randomSeed);

    if (config->_noiseIndexFile.size() != 0) {
        _codec = new Codec(config);
        _noiseClips = new NoiseClips(
            config->_noiseIndexFile, config->_noiseDir, _codec
        );
    }
}

audio::transformer::~transformer() {
    delete _specgram;
    if(_codec != 0) {
        delete _codec;
    }
    if(_noiseClips != 0) {
        delete _noiseClips;
    }
}

std::shared_ptr<audio::decoded> audio::transformer::transform(
      std::shared_ptr<audio::params> params,
      std::shared_ptr<audio::decoded> decoded) {
    if (_noiseClips != 0) {
        _noiseClips->addNoise(decoded->_raw, _state);
    }

    // set up _buf in decoded to accept data from generate
    decoded->_buf.resize(params->_width * params->_height);

    // convert from time domain to frequency domain
    decoded->_len = _specgram->generate(
        decoded->_raw, decoded->_buf.data(), decoded->_buf.size()
    );

    return decoded;
}
