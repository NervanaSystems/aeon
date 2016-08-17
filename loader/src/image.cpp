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

#include <iostream>

#include "image.hpp"

using namespace nervana;
using namespace std;

void image::rotate(const cv::Mat& input, cv::Mat& output, int angle, bool interpolate, const cv::Scalar& border)
{
    if (angle == 0) {
        output = input;
    } else {
        cv::Point2i pt(input.cols / 2, input.rows / 2);
        cv::Mat rot = cv::getRotationMatrix2D(pt, angle, 1.0);
        int flags;
        if(interpolate) {
            flags = cv::INTER_LINEAR;
        } else {
            flags = cv::INTER_NEAREST;
        }
        cv::warpAffine(input, output, rot, input.size(), flags, cv::BORDER_CONSTANT, border);
    }
}

void image::resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size, bool interpolate)
{
    if (size == input.size()) {
        output = input;
    } else {
        int inter;
        if(interpolate) {
            inter = input.size().area() < size.area() ? CV_INTER_CUBIC : CV_INTER_AREA;
        } else {
            inter = CV_INTER_NN;
        }
        cv::resize(input, output, size, 0, 0, inter);
    }
}

void image::convert_mix_channels(vector<cv::Mat>& source, vector<cv::Mat>& target, vector<int>& from_to)
{
    if(source.size() == 0) throw invalid_argument("convertMixChannels source size must be > 0");
    if(target.size() == 0) throw invalid_argument("convertMixChannels target size must be > 0");
    if(from_to.size() == 0) throw invalid_argument("convertMixChannels from_to size must be > 0");

    const vector<cv::Mat>* prepared_source = &source;
    vector<cv::Mat>  tmp_source;
    if(source[0].depth() != target[0].depth()) {
        // Conversion required
        for(cv::Mat mat : source) {
            cv::Mat tmp;
            mat.convertTo(tmp, target[0].type());
            tmp_source.push_back(tmp);
        }
        prepared_source = &tmp_source;
    }

    if(prepared_source->size() == 1 && target.size() == 1) {
        size_t size = target[0].total() * target[0].elemSize();
        memcpy(target[0].data, (*prepared_source)[0].data, size);
    }
    else {
        cv::mixChannels(*prepared_source, target, from_to);
    }
}

tuple<float,cv::Size> image::calculate_scale_shape(cv::Size size, int min_size, int max_size) {
    int im_size_min = std::min(size.width,size.height);
    int im_size_max = max(size.width,size.height);
    float im_scale = float(min_size) / float(im_size_min);
    // Prevent the biggest axis from being more than FRCN_MAX_SIZE
    if(round(im_scale * im_size_max) > max_size) {
        im_scale = float(max_size) / float(im_size_max);
    }
    cv::Size im_shape{int(round(size.width*im_scale)), int(round(size.height*im_scale))};
    return make_tuple(im_scale, im_shape);
}

cv::Size2f image::cropbox_max_proportional(const cv::Size2f& in_size, const cv::Size2f& out_size) {
    cv::Size2f result = out_size;
    float scale = in_size.width / result.width;
    result = result * scale;
    if(result.height > in_size.height) {
        scale = in_size.height / result.height;
        result = result * scale;
    }
    return result;
}

cv::Size2f image::cropbox_linear_scale(const cv::Size2f& in_size, float scale)
{
    return in_size * scale;
}

cv::Size2f image::cropbox_area_scale(const cv::Size2f& in_size, float scale)
{
    cv::Size2f result = in_size;
    return result;
}

cv::Point2f image::cropbox_shift(const cv::Size2f& in_size, const cv::Size2f& crop_box, float xoff, float yoff)
{
    cv::Point2f result;
    result.x = (in_size.width - crop_box.width) * xoff;
    result.y = (in_size.height - crop_box.height) * yoff;
    return result;
}

/* Transform:
    image::config will be a supplied bunch of params used by this provider.
    on each record, the transformer will use the config along with the supplied
    record to fill a transform_params structure which will have

    Spatial distortion params:
    randomly sampled crop box (based on params->center, params->horizontal_distortion, params->scale_pct, record size)
    randomly determined flip (based on params->flip)
    randomly sampled rotation angle (based on params->angle)

    Photometric distortion params:
    randomly sampled contrast, brightness, saturation, lighting values (based on params->cbs, lighting bounds)

*/

image::photometric::photometric() :
    _CPCA{{0.39731118,  0.70119634, -0.59200296},
                            {-0.81698062, -0.02354167, -0.5761844},
                            {0.41795513, -0.71257945, -0.56351045}},
    CPCA(3, 3, CV_32FC1, (float*)_CPCA),
    CSTD(3, 1, CV_32FC1, {19.72083305, 37.09388853, 121.78006099}),
    GSCL(3, 1, CV_32FC1, {0.114, 0.587, 0.299})
{
}

/*
Implements colorspace noise perturbation as described in:
Krizhevsky et. al., "ImageNet Classification with Deep Convolutional Neural Networks"
Constructs a random coloring pixel that is uniformly added to every pixel of the image.
lighting is filled with normally distributed values prior to calling this function.
*/
void image::photometric::lighting(cv::Mat& inout, vector<float> lighting, float color_noise_std)
{
    // Skip transformations if given deterministic settings
    if (lighting.size() > 0) {
        cv::Mat alphas(3, 1, CV_32FC1, lighting.data());
        alphas = (CPCA * CSTD.mul(alphas));  // this is the random coloring pixel
        auto pixel = alphas.reshape(3, 1).at<cv::Scalar_<float>>(0, 0);
        inout = (inout + pixel) / (1.0 + color_noise_std);
    }
}

/*
Implements contrast, brightness, and saturation jittering using the following definitions:
Contrast: Add some multiple of the grayscale mean of the image.
Brightness: Magnify the intensity of each pixel by photometric[1]
Saturation: Add some multiple of the pixel's grayscale value to itself.
photometric is filled with uniformly distributed values prior to calling this function
*/
// adjusts contrast, brightness, and saturation according
// to values in photometric[0], photometric[1], photometric[2], respectively
void image::photometric::cbsjitter(cv::Mat& inout, const vector<float>& photometric)
{
    // Skip transformations if given deterministic settings
    if (photometric.size() > 0) {
        /****************************
        *  BRIGHTNESS & SATURATION  *
        *****************************/
        cv::Mat satmtx = photometric[1] * (photometric[2] * cv::Mat::eye(3, 3, CV_32FC1) +
                                (1 - photometric[2]) * cv::Mat::ones(3, 1, CV_32FC1) * GSCL.t());
        cv::transform(inout, inout, satmtx);

        /*************
        *  CONTRAST  *
        **************/
        cv::Mat gray_mean;
        cv::cvtColor(cv::Mat(1, 1, CV_32FC3, cv::mean(inout)), gray_mean, CV_BGR2GRAY);
        inout = photometric[0] * inout + (1 - photometric[0]) * gray_mean.at<cv::Scalar_<float>>(0, 0);
    }
}
