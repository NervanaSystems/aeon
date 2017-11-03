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

#include "specgram.hpp"
#include "util.hpp"

using cv::Mat;
using cv::Range;
using namespace std;
using namespace nervana;

// These can all be static
void specgram::wav_to_specgram(const Mat& wav_col_mat,
                               const int  frame_length_tn,
                               const int  frame_stride_tn,
                               const int  max_time_steps,
                               const Mat& window,
                               Mat&       specgram)
{
    // const Mat& wav_mat = wav.get_data();
    // Read as a row vector
    Mat wav_mat = wav_col_mat.reshape(1, 1);

    // TODO: support more sample formats
    if (wav_mat.elemSize1() != 2)
    {
        throw std::runtime_error("Unsupported number of bytes per sample: " +
                                 std::to_string(wav_mat.elemSize1()));
    }

    // Go from time domain to strided signal
    Mat wav_frames;
    {
        int num_frames = ((wav_mat.cols - frame_length_tn) / frame_stride_tn) + 1;
        num_frames     = std::min(num_frames, max_time_steps);
        // ensure that there is enough data for at least one frame
        affirm(num_frames >= 0, "number of frames is negative");

        wav_frames.create(num_frames, frame_length_tn, wav_mat.type());
        for (int frame = 0; frame < num_frames; frame++)
        {
            int start = frame * frame_stride_tn;
            int end   = start + frame_length_tn;
            wav_mat.colRange(start, end).copyTo(wav_frames.row(frame));
        }
    }

    // Prepare for DFT by converting to float
    Mat input;
    wav_frames.convertTo(input, CV_32FC1);

    // Apply window if it has been created
    if (window.cols == frame_length_tn)
    {
        input = input.mul(cv::repeat(window, input.rows, 1));
    }

    Mat planes[] = {input, Mat::zeros(input.size(), CV_32FC1)};
    Mat compx;
    cv::merge(planes, 2, compx);
    cv::dft(compx, compx, cv::DFT_ROWS);
    compx = compx(Range::all(), Range(0, frame_length_tn / 2 + 1));

    cv::split(compx, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);

    // NOTE: at this point we are returning the specgram representation in
    // (time_steps, freq_steps) shape order.

    specgram = planes[0];

    return;
}

/** \brief Create an array of frequency weights to convert from linear
* frequencies to mel-frequencies.
* For reference, see:
* http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#computing-the-mel-filterbank
*/
void specgram::create_filterbanks(const int num_filters,
                                  const int fftsz,
                                  const int sample_freq_hz,
                                  Mat&      fbank)
{
    double min_mel_freq   = hz_to_mel(0.0);
    double max_mel_freq   = hz_to_mel(sample_freq_hz / 2.0);
    double mel_freq_delta = (max_mel_freq - min_mel_freq) / (num_filters + 1);

    // Determine the nearest freq index for each mel-frequency band
    vector<int> bins;
    double      scale_by = (1 + fftsz) / (double)sample_freq_hz;
    for (int j = 0; j <= 1; ++j)
    {
        bins.push_back(floor(scale_by * mel_to_hz(min_mel_freq + j * mel_freq_delta)));
    }

    int num_freqs = fftsz / 2 + 1;
    fbank         = cv::Mat::zeros(num_freqs, num_filters, CV_32F);

    // Create triangular windows from three neighboring bins
    for (int j = 0; j < num_filters; ++j)
    {
        bins.push_back(floor(scale_by * mel_to_hz(min_mel_freq + (j + 2) * mel_freq_delta)));
        // Left side of triangle
        // weights are ratio of distance from left edge to distance from center to left
        for (int i = bins[j]; i < bins[j + 1]; ++i)
        {
            fbank.at<float>(i, j) = (i - bins[j]) / (1.0 * (bins[j + 1] - bins[j]));
        }
        // Right side of triangle
        for (int i = bins[j + 1]; i < bins[j + 2]; ++i)
        {
            fbank.at<float>(i, j) = (bins[j + 2] - i) / (1.0 * (bins[j + 2] - bins[j + 1]));
        }
    }
    return;
}

void specgram::specgram_to_cepsgram(const Mat& specgram, const Mat& filter_bank, Mat& cepsgram)
{
    cepsgram = (specgram.mul(specgram) / (2 * (specgram.cols - 1))) * filter_bank;
    cv::log(cepsgram, cepsgram);
    return;
}

void specgram::cepsgram_to_mfcc(const Mat& cepsgram, const int num_cepstra, Mat& mfcc)
{
    affirm(num_cepstra <= cepsgram.cols, "num_cepstra <= cepsgram.cols");
    Mat padcepsgram;
    if (cepsgram.cols % 2 == 0)
    {
        padcepsgram = cepsgram;
    }
    else
    {
        cv::copyMakeBorder(cepsgram, padcepsgram, 0, 0, 0, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    cv::dct(padcepsgram, padcepsgram, cv::DCT_ROWS);
    mfcc = padcepsgram(Range::all(), Range(0, num_cepstra));
    return;
}

void specgram::create_window(const std::string& window_type, const int n, Mat& win)
{
    if (window_type == "none")
    {
        return;
    }

    win.create(1, n, CV_32FC1);

    float twopi_by_n = 2.0 * CV_PI / (float)(n - 1);
    for (int i = 0; i < n; i++)
    {
        if (window_type == "hann")
        {
            win.at<float>(0, i) = 0.5 - 0.5 * cos(twopi_by_n * i);
        }
        else if (window_type == "blackman")
        {
            win.at<float>(0, i) = 0.42 - 0.5 * cos(twopi_by_n * i) + 0.08 * cos(2 * twopi_by_n * i);
        }
        else if (window_type == "hamming")
        {
            win.at<float>(0, i) = 0.54 - 0.46 * cos(twopi_by_n * i);
        }
        else if (window_type == "bartlett")
        {
            win.at<float>(0, i) = 1.0 - 2.0 * fabs(i - n / 2.0) / n;
        }
        else
        {
            throw std::runtime_error("Unsupported window function");
        }
    }
}
