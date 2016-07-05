#include "audio.hpp"

#include "etl_audio.hpp"

using namespace std;
using namespace nervana;

bool audio::config::set_config(nlohmann::json js) {
    // TODO
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
    // TODO: this MediaParams is never freed
    _codec = new Codec(config);
    avcodec_register_all();
}

audio::extractor::~extractor() {
    delete _codec;
}

std::shared_ptr<audio::decoded> audio::extractor::extract(const char* item, int itemSize) {
    return make_shared<audio::decoded>(_codec->decode(item, itemSize));
}

audio::transformer::transformer(std::shared_ptr<const audio::config> config) {
    // TODO: this MediaParams is never freed
    _codec = new Codec(config);
    _specgram = new Specgram(config, config->_randomSeed);

    if (config->_noiseIndexFile != 0) {
        _noiseClips = new NoiseClips(
            config->_noiseIndexFile, config->_noiseDir, _codec
        );
    }
}

audio::transformer::~transformer() {
    delete _specgram;
    delete _codec;
}

std::shared_ptr<audio::decoded> audio::transformer::transform(
      std::shared_ptr<audio::params> params,
      std::shared_ptr<audio::decoded> decoded) {
    if (_noiseClips != 0) {
        _noiseClips->addNoise(decoded->_raw, _rng);
    }

    // set up _buf in decoded to accept data from generate
    decoded->_buf.resize(params->_width * params->_height);

    // convert from time domain to frequency domain
    int len = _specgram->generate(
        decoded->_raw, decoded->_buf.data(), params->_width * params->_height
    );

    // TODO: in private-neon the length of the uterance is used
    // later on down the road somewhere.  Still need to figure out
    // how to pass this through ...

    // if (meta != 0) {
    //     *meta = len;
    // }

    return decoded;
}
