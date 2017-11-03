/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include "etl_audio.hpp"

using namespace std;
using namespace nervana;

/** \brief Extract audio data from a wav file using sox */
std::shared_ptr<audio::decoded> audio::extractor::extract(const void* item, size_t itemSize) const
{
    return make_shared<audio::decoded>(nervana::read_audio_from_mem((const char*)item, itemSize));
}

audio::transformer::transformer(const audio::config& config)
    : _cfg(config)
{
    if (_cfg.feature_type != "samples")
    {
        specgram::create_window(_cfg.window_type, _cfg.frame_length_tn, _window);
        specgram::create_filterbanks(
            _cfg.num_filters, _cfg.frame_length_tn, _cfg.sample_freq_hz, _filterbank);
    }
    _noisemaker = make_shared<noise_clips>(_cfg.noise_index_file, _cfg.noise_root);
}

audio::transformer::~transformer()
{
}

/** \brief Transform the raw sound waveform into the desired feature space,
* possibly after adding noise.
*
* The transformation pipeline is as follows:
* 1. Optionally add noise (controlled by add_noise parameter)
* 2. Convert to spectrogram
* 3. Optionally convert to MFSC (controlled by feature_type parameter)
* 4. Optionally convert to MFCC (controlled by feature_type parameter)
* 5. Optionally time-warp (controlled by time_scale_fraction)
*/
std::shared_ptr<audio::decoded>
    audio::transformer::transform(std::shared_ptr<augment::audio::params> params,
                                  std::shared_ptr<audio::decoded>         decoded) const
{
    cv::Mat& samples_mat = decoded->get_time_data();
    _noisemaker->addNoise(samples_mat,
                          params->add_noise,
                          params->noise_index,
                          params->noise_offset_fraction,
                          params->noise_level); // no-op if no noise files

    if (_cfg.feature_type == "samples")
    {
        decoded->get_freq_data() = samples_mat;
        decoded->valid_frames    = std::min((uint32_t)samples_mat.rows, (uint32_t)_cfg.time_steps);
    }
    else
    {
        // convert from time domain to frequency domain into the freq mat
        specgram::wav_to_specgram(samples_mat,
                                  _cfg.frame_length_tn,
                                  _cfg.frame_stride_tn,
                                  _cfg.time_steps,
                                  _window,
                                  decoded->get_freq_data());
        if (_cfg.feature_type != "specgram")
        {
            cv::Mat tmpmat;
            specgram::specgram_to_cepsgram(decoded->get_freq_data(), _filterbank, tmpmat);
            if (_cfg.feature_type == "mfcc")
            {
                specgram::cepsgram_to_mfcc(tmpmat, _cfg.num_cepstra, decoded->get_freq_data());
            }
            else
            {
                decoded->get_freq_data() = tmpmat;
            }
        }

        // place into a destination with the appropriate time dimensions
        cv::Mat resized;
        cv::resize(decoded->get_freq_data(),
                   resized,
                   cv::Size(),
                   1.0,
                   params->time_scale_fraction,
                   (params->time_scale_fraction > 1.0) ? CV_INTER_CUBIC : CV_INTER_AREA);
        decoded->get_freq_data() = resized;
        decoded->valid_frames    = std::min((uint32_t)resized.rows, (uint32_t)_cfg.time_steps);
    }

    return decoded;
}

void audio::loader::load(const vector<void*>& outbuf, shared_ptr<audio::decoded> input) const
{
    auto nframes = input->valid_frames;
    auto frames  = input->get_freq_data();
    int  cv_type = _cfg.get_shape_type().get_otype().get_cv_type();

    if (_cfg.feature_type != "samples")
    {
        cv::normalize(frames, frames, 0, 255, CV_MINMAX);
    }

    cv::Mat padded_frames(_cfg.time_steps, _cfg.freq_steps, cv_type);

    frames(cv::Range(0, nframes), cv::Range::all())
        .convertTo(padded_frames(cv::Range(0, nframes), cv::Range::all()), cv_type);

    if (nframes < _cfg.time_steps)
    {
        padded_frames(cv::Range(nframes, _cfg.time_steps), cv::Range::all()) = cv::Scalar::all(0);
    }

    cv::Mat dst(_cfg.freq_steps, _cfg.time_steps, cv_type, (void*)outbuf[0]);
    cv::transpose(padded_frames, dst);
    cv::flip(dst, dst, 0);

    if (_cfg.emit_length)
    {
        uint32_t* length_buf = (uint32_t*)outbuf[1];
        *length_buf          = std::min(input->size(), (unsigned long)_cfg.max_duration_tn);
    }
}
