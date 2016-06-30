#include "audio.hpp"

#include "etl_audio.hpp"

using namespace std;
using namespace nervana;

bool audio::config::set_config(nlohmann::json js) {
    // TODO
}

shared_ptr<audio::params> audio::param_factory::make_params(std::shared_ptr<const decoded>) {
}

audio::decoded::decoded(RawMedia* raw)
    : _raw(raw) {
}

MediaType audio::decoded::get_type() {
    return MediaType::AUDIO;
}

size_t audio::decoded::getSize() {
    return _raw->numSamples();
}

audio::extractor::extractor(std::shared_ptr<const audio::config>)
    // TODO: this MediaParams is never freed
    : _codec(new MediaParams(MediaType::AUDIO))
{
    avcodec_register_all();
}

std::shared_ptr<audio::decoded> audio::extractor::extract(const char* item, int itemSize) {
    return make_shared<audio::decoded>(_codec.decode(item, itemSize));
}

audio::transformer::transformer(std::shared_ptr<const audio::config> config) {
    // TODO: this SignalParams is empty and never freed
    _specgram = new Specgram(new SignalParams(MediaType::AUDIO), config->_randomSeed);
}

audio::transformer::~transformer() {
    delete _specgram;
}

std::shared_ptr<audio::decoded> audio::transformer::transform(std::shared_ptr<audio::params> params,
                                                              std::shared_ptr<audio::decoded> decoded) {
    /*
    if (_noiseClips != 0) {
        _noiseClips->addNoise(decoded->_raw, _rng);
    }
    int len = _specgram->generate(decoded->_raw, buf, bufSize);
    if (meta != 0) {
        *meta = len;
    }
    */
    return decoded;
}
