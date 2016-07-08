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

#include "specgram.hpp"

using cv::Mat;
using cv::Range;
using cv::Size;
using namespace std;
using std::stringstream;
using std::vector;

Specgram::Specgram(shared_ptr<const nervana::audio::config> cfg)
: _feature (cfg->_feature),
  _max_duration_tn(cfg->max_duration_tn),
  _frame_length_tn(cfg->frame_length_tn),
  _stride(      cfg->_stride),
  _width(       cfg->_width),
  _numFreqs(    cfg->_frame_length_tn / 2 + 1),
  _height(      cfg->_height),
  _num_cepstra(  cfg->num_cepstra),
{
    assert(_stride != 0);
    create_window(cfg->window_type, _frame_length_tn, _window);

    fill_filterbanks(cfg->num_filters, _frame_length_tn, cfg->sample_freq_hz, _fbank);

    cv::transpose(_fbank);
}

Specgram::~Specgram() {
    delete[] _buf;
}

int Specgram::generate(shared_ptr<RawMedia> time_data, cv::Mat &freq_data) {
    // TODO: get rid of this assumption
    assert(time_data->bytesPerSample() == 2);

    int numWindows = stridedSignal(time_data);

    Mat signal(numWindows, _frame_length_tn, CV_16SC1, (short*) _buf);
    Mat input;
    signal.convertTo(input, CV_32FC1);

    if (_window.cols == _frame_length_tn) {
        signal = signal.mul(cv::repeat(_window, signal.rows, 1));
    }

    Mat planes[] = {input, Mat::zeros(input.size(), CV_32FC1)};
    Mat compx;
    cv::merge(planes, 2, compx);

    cv::dft(compx, compx, cv::DFT_ROWS);
    compx = compx(Range::all(), Range(0, _numFreqs));

    cv::split(compx, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    Mat mag;
    if (_feature == SPECGRAM) {
        mag = planes[0];
    } else {
        extractFeatures(planes[0], mag);
    }

    Mat feats;

    // Rotate by 90 degrees.
    cv::transpose(mag, feats);
    cv::flip(feats, feats, 0);

    cv::normalize(feats, feats, 0, 255, CV_MINMAX, CV_8UC1);
    Mat result(feats.rows, _width, CV_8UC1, buf);

    // assert(feats.rows <= _height);

    feats.copyTo(result(Range(0, feats.rows), Range(0, feats.cols)));

    // Pad the rest with zeros.
    result(Range::all(), Range(feats.cols, result.cols)) = cv::Scalar::all(0);

    // Return the percentage of valid columns.
    return feats.cols * 100 / result.cols;
}


void Specgram::create_window(const std::string& window_type, uint32_t window_length, cv::Mat& win)
{

    if (window_type == "none") {
        return;
    }

    int n = static_cast<int>(window_length);
    win.create(1, n, CV_32FC1)

    float twopi_by_n = 2.0 * CV_PI / (float) n;
    for (int i = 0; i <= n; i++) {
        if (window_type == "hann") {
            win.at<float>(0, i) = 0.5 - 0.5 * cos(twopi_by_n * i);
        } else if (window_type == "blackman") {
            win.at<float>(0, i) = 0.42 - 0.5 * cos(twopi_by_n * i) + 0.08 * cos(2 * twopi_by_n * i);
        } else if (window_type == "hamming") {
            win.at<float>(0, i) = 0.54 - 0.46 * cos(twopi_by_n * i);
        } else if (window_type == "bartlett") {
            win.at<float>(0, i) = 1.0 - 2.0 * fabs(i - n / 2.0) / n;
        } else {
            throw std::runtime_error("Unsupported window function");
        }
    }
}


int Specgram::stridedSignal(shared_ptr<RawMedia> raw) {
    // read frames of length `_frame_length_tn` shifted every `_frame_stride_tn` samples

    // truncate data in raw if larger than _max_duration_tn
    int length_tn = std::min(raw->numSamples(), static_cast<int>(_max_duration_tn));

    // assert that there is more than 1 window of data in raw
    assert(length_tn >= _frame_length_tn);

    // count is the number of windows to capture
    int num_frames = ((length_tn - _frame_length_tn) / _frame_stride_tn) + 1;

    char* src = raw->getBuf(0);

    for (int i = 0; i < num_frames; i++) {
        memcpy(dst, src, windowSizeInBytes);
        dst += windowSizeInBytes;
        src += strideInBytes;
    }

    return num_frames;
}

double Specgram::hzToMel(double freq_hz)
{
    return 2595 * std::log10(1 + freq_hz / 700.0);
}

double Specgram::melToHz(double freq_mel)
{
    return 700 * (std::pow(10, freq_mel / 2595.0) - 1);
}

void Specgram::linspace(double a, double b, vector<double>& interval)
{
    double delta = (b - a) / (interval.size() - 1);
    for (auto &x : interval) {
        x = a;
        a += delta;
    }
    return;
}

void Specgram::fill_filterbanks(int num_filters, int ffts, uint32_t samplingRate, cv::Mat &fbank) {
    double min_mel_freq = hzToMel(0.0);
    double max_mel_freq = hzToMel(samplingRate / 2.0);

    // Get mel-scale bin centers
    int num_mel = num_filters + 2;
    vector<double> mel_intervals (num_mel);
    vector<int> bins (num_mel);

    linspace(min_mel_freq, max_mel_freq, mel_intervals);

    for (int k=0; k < num_mel; ++k) {
        bins[k] = std::floor((1 + ffts) * melToHz(mel_intervals[k]) / samplingRate);
    }

    fbank::create(num_filters, 1 + ffts / 2, CV_32F);

    for (int j=0; j<num_filters; ++j) {
        for (int i=bins[j]; i<bins[j+1]; ++i) {
            fbank.at<float>(j, i) = (i - bins[j]) / (1.0*(bins[j + 1] - bins[j]));
        }
        for (int i=bins[j+1]; i<bins[j+2]; ++i) {
            fbank.at<float>(j, i) = (bins[j+2]-i) / (1.0*(bins[j + 2] - bins[j+1]));
        }
    }
    return;
}

void Specgram::extractFeatures(Mat& spectrogram, Mat& features) {
    Mat powspec = spectrogram.mul(spectrogram);
    powspec *= 1.0 / _frame_length_tn;
    Mat cepsgram = powspec * _fbank;
    log(cepsgram, cepsgram);
    if (_feature == MFSC) {
        features = cepsgram;
        return;
    }
    int pad_cols = cepsgram.cols;
    int pad_rows = cepsgram.rows;
    if (cepsgram.cols % 2 != 0) {
        pad_cols = 1 + cepsgram.cols;
    }
    if (cepsgram.rows % 2 != 0) {
        pad_rows = 1 + cepsgram.rows;
    }
    Mat padcepsgram = Mat::zeros(pad_rows, pad_cols, CV_32F);
    cepsgram.copyTo(padcepsgram(Range(0, cepsgram.rows), Range(0, cepsgram.cols)));
    dct(padcepsgram, padcepsgram, cv::DFT_ROWS);
    cepsgram = padcepsgram(Range(0, cepsgram.rows), Range(0, cepsgram.cols));
    features = cepsgram(Range::all(), Range(0, _num_cepstra));
}
