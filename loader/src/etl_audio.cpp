#include "audio.hpp"

#include "etl_audio.hpp"

using namespace std;
using namespace nervana;

bool audio::config::set_config(nlohmann::json js) {
    // TODO
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

