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

#include <fstream>
#include "gtest/gtest.h"

#include "etl_audio.hpp"
#include "wav_data.hpp"

using namespace std;
using namespace nervana;

// This test is for comparing that a generated wav_data structure can write its data out to
// 16-bit PCM that can be read via extractor (just via buffer rather than actually touching disk)
TEST(wav,compare) {
    sinewave_generator sg{400, 500};
    wav_data wav(sg, 2, 16000, false);

    uint32_t wav_bufsize = wav_data::HEADER_SIZE + wav.nbytes();
    char *wav_buf = new char[wav_bufsize];

    wav.write_to_buffer(wav_buf, wav_bufsize);

    audio::extractor extractor;
    auto d_audio = extractor.extract(wav_buf, wav_bufsize);

    auto extracted_wav = d_audio->get_time_data();
    ASSERT_EQ(extracted_wav->get_data().rows, wav.nsamples());

    uint32_t num_samples = wav.nsamples();
    bool all_eq = true;
    for (int i=0; i<num_samples; i++)
    {
        size_t offset = i * sizeof(uint16_t);
        uint16_t*  waddr = (uint16_t *) (wav.get_raw_data()[0] + offset);
        uint16_t*  daddr = (uint16_t *) (extracted_wav->get_raw_data()[0] + offset);
        if (*waddr != *daddr)
        {
            all_eq = false; break;
        }
    }

    ASSERT_EQ(all_eq, true);
}

TEST(audio, specgram) {
    // This test generates a 1kHz signal and ensures that the spectrogram creates the correct
    // line image
    float signal_freq = 1000;
    sinewave_generator sg{signal_freq};
    int wav_len_sec = 4, sample_freq = 16000;
    bool stereo = false;
    wav_data wav(sg, wav_len_sec, sample_freq, stereo);


    int time_steps = 100;  // This is how wide we want our image to be (number of strides)
    int frame_length_tn = 256; //  16 cycles per frame
    int frame_stride_tn = 16;  //  Shift over 1 period each time
    // int nsamples = (time_steps - 1) * frame_stride_tn + frame_length_tn;

    // Now generate spectrogram
    cv::Mat spec, window;
    specgram::wav_to_specgram(wav.get_data(), frame_length_tn, frame_stride_tn, time_steps, window, spec);

    // Scale back because our original signal was set up to use full 16-bit dynamic range
    spec = spec / INT16_MAX;
    cv::Mat spec_uint8;
    spec.assignTo(spec_uint8, CV_8U);

    // Check the correct shape
    ASSERT_EQ(spec.rows, time_steps);
    ASSERT_EQ(spec.cols, frame_length_tn / 2 + 1);

    // Now create the reference
    cv::Mat ref_spec(spec.rows, spec.cols, CV_8U);
    ref_spec.setTo(cv::Scalar(0));
    int freq_bin = signal_freq / sample_freq * frame_length_tn;
    ref_spec.col(freq_bin).setTo(cv::Scalar(frame_length_tn / 2));

    // Compare pixel-wise sameness
    cv::Mat diff = spec_uint8 != ref_spec;
    ASSERT_EQ(cv::countNonZero(diff), 0);
}


TEST(audio,transform) {

    auto js = R"(
        {
            "max_duration": "2000 milliseconds",
            "frame_length": "1024 samples",
            "frame_stride": "256 samples",
            "sample_freq_hz": 44100,
            "feature_type": "mfcc",
            "num_filters": 64
        }
    )"_json;

    float sine_freq = 400;
    int16_t sine_ampl = 500;
    sinewave_generator sg{sine_freq, sine_ampl};
    int wav_len_sec = 4, sample_freq = 44100;
    bool stereo = false;

    wav_data wav(sg, wav_len_sec, sample_freq, stereo);
    uint32_t bufsize = wav_data::HEADER_SIZE + wav.nbytes();
    char *databuf = new char[bufsize];

    wav.write_to_buffer(databuf, bufsize);

    audio::config config(js);

    audio::extractor extractor;
    audio::transformer _imageTransformer(config);
    audio::param_factory factory(config);

    auto decoded_audio = extractor.extract(databuf, bufsize);
    auto audioParams = factory.make_params(decoded_audio);

    _imageTransformer.transform(audioParams, decoded_audio);
    auto shape = config.get_shape_type();
    ASSERT_EQ(shape.get_shape()[0], 1);
    ASSERT_EQ(shape.get_shape()[1], 40);
    ASSERT_NE(decoded_audio->get_freq_data().rows, 0);
    delete[] databuf;
}

TEST(wav,read) {
    // requires sox : `apt-get install sox` on ubuntu
    auto a = system("sox -r 16000 -b 16 -e s -n output.wav synth 3 sine 400 vol 0.5");
    string test_file = "output.wav";
    if (a == 0)
    {
        basic_ifstream<char> ifs(test_file, ios::binary);
        // read the data:
        vector<char> buf((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());

        wav_data wav(&(buf[0]), buf.size());
        ASSERT_EQ(wav.sample_rate(), 16000);
        ASSERT_EQ(wav.nsamples(), 16000 * 3);
        remove(test_file.c_str());
    }
    else
    {
        cout << "Missing sox for wav read test" << endl;
    }
}


TEST(audio,transform2) {

    auto js = R"(
        {
            "max_duration": "3 seconds",
            "frame_length": "256 samples",
            "frame_stride": "128 samples",
            "sample_freq_hz": 16000
        }
    )"_json;

    float sine_freq = 400;
    int16_t sine_ampl = 500;
    sinewave_generator sg{sine_freq, sine_ampl};
    int wav_len_sec = 2, sample_freq = 16000;
    bool stereo = false;

    wav_data wav(sg, wav_len_sec, sample_freq, stereo);
    uint32_t bufsize = wav_data::HEADER_SIZE + wav.nbytes();
    char *databuf = new char[bufsize];

    wav.write_to_buffer(databuf, bufsize);

    audio::config config(js);

    audio::extractor extractor;
    audio::transformer _imageTransformer(config);
    audio::param_factory factory(config);

    auto decoded_audio = extractor.extract(databuf, bufsize);
    auto audioParams = factory.make_params(decoded_audio);

    _imageTransformer.transform(audioParams, decoded_audio);
    auto shape = config.get_shape_type();
    ASSERT_EQ(shape.get_shape()[0], 1);
    ASSERT_EQ(shape.get_shape()[1], 256/2 + 1);
    ASSERT_NE(decoded_audio->get_freq_data().rows, 0);
    delete[] databuf;
}
