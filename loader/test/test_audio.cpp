/*
 Copyright 2016 Nervana Systems Inc.
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

#include "gtest/gtest.h"

#include "etl_audio.hpp"
#include "gen_audio.hpp"

using namespace std;
using namespace nervana;

shared_ptr<audio::decoded> generate_decoded_audio(float frequencyHz, int duration) {
    // generate a decoded audio by first using gen_audio::encode to generate
    // the encoded audio and then decoding it with an extractor.
    // TODO: refactor audio signal generation from encoding step in
    //       gen_audio::encode
    gen_audio gen;
    vector<unsigned char> encoded_audio = gen.encode(frequencyHz, duration);
    auto config = make_shared<audio::config>();
    config->_mtype = MediaType::AUDIO;
    audio::extractor extractor(config);
    return extractor.extract((char*)encoded_audio.data(), encoded_audio.size());
}

TEST(etl, audio_extract) {
    auto decoded_audio = generate_decoded_audio(1000, 2000);

    // because of the way encoded_audio is being generated, there
    // are 88704 samples instead of 44100 * 2 = 88200 like we would
    // expect ... a fix for another time
    ASSERT_EQ(decoded_audio->getSize(), 88704);
}

TEST(etl, audio_transform) {
    auto decoded_audio = generate_decoded_audio(1000, 2000);

    auto config = make_shared<audio::config>();
    config->_mtype = MediaType::AUDIO;
    config->_windowSize = 1024;
    config->_stride = config->_windowSize / 4;
    config->_clipDuration = 2000;
    config->_samplingFreq = 44100;
    config->_width = (((config->_clipDuration * config->_samplingFreq / 1000) - config->_windowSize) / config->_stride) + 1;
    // TODO: how to compute height ahead of time?
    config->_height = 513;
    config->_numFilts = 64;
    config->_numCepstra = 40;

    audio::transformer _imageTransformer(config);

    std::default_random_engine dre;
    audio::param_factory factory(config, dre);
    auto audioParams = factory.make_params(decoded_audio);

    _imageTransformer.transform(audioParams, decoded_audio);
}
