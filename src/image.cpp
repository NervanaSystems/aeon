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

#include <iostream>

#include "image.hpp"
#include "util.hpp"
#include "log.hpp"

using namespace nervana;
using namespace std;

void image::rotate(
    const cv::Mat& input, cv::Mat& output, int angle, bool interpolate, const cv::Scalar& border)
{
    if (angle == 0)
    {
        output = input;
    }
    else
    {
        cv::Point2i pt(input.cols / 2, input.rows / 2);
        cv::Mat     rot = cv::getRotationMatrix2D(pt, angle, 1.0);
        int         flags;
        if (interpolate)
        {
            flags = cv::INTER_LINEAR;
        }
        else
        {
            flags = cv::INTER_NEAREST;
        }
        cv::warpAffine(input, output, rot, input.size(), flags, cv::BORDER_CONSTANT, border);
    }
}

void image::add_padding(cv::Mat& input, int padding, cv::Size2i crop_offset)
{
    // crop overlaps completely with input image
    if (padding == 0 || (crop_offset.width == padding && crop_offset.height == padding))
    {
        return;
    }

    cv::Mat    paddedImage;
    cv::Scalar blackPixel{0, 0, 0};
    cv::copyMakeBorder(
        input, paddedImage, padding, padding, padding, padding, cv::BORDER_CONSTANT, blackPixel);
    cv::Rect cropbox{crop_offset.width, crop_offset.height, input.cols, input.rows};
    input = paddedImage(cropbox);
}

void image::resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size, bool interpolate)
{
    if (size == input.size())
    {
        output = input;
    }
    else
    {
        int inter;
        if (interpolate)
        {
            inter = input.size().area() < size.area() ? CV_INTER_CUBIC : CV_INTER_AREA;
        }
        else
        {
            inter = CV_INTER_NN;
        }
        cv::resize(input, output, size, 0, 0, inter);
    }
}

void image::convert_mix_channels(vector<cv::Mat>& source,
                                 vector<cv::Mat>& target,
                                 vector<int>&     from_to)
{
    if (source.size() == 0)
        throw invalid_argument("convertMixChannels source size must be > 0");
    if (target.size() == 0)
        throw invalid_argument("convertMixChannels target size must be > 0");
    if (from_to.size() == 0)
        throw invalid_argument("convertMixChannels from_to size must be > 0");

    const vector<cv::Mat>* prepared_source = &source;
    vector<cv::Mat>        tmp_source;
    if (source[0].depth() != target[0].depth())
    {
        // Conversion required
        for (cv::Mat mat : source)
        {
            cv::Mat tmp;
            mat.convertTo(tmp, target[0].type());
            tmp_source.push_back(tmp);
        }
        prepared_source = &tmp_source;
    }

    if (prepared_source->size() == 1 && target.size() == 1)
    {
        size_t size = target[0].total() * target[0].elemSize();
        memcpy(target[0].data, (*prepared_source)[0].data, size);
    }
    else
    {
        cv::mixChannels(*prepared_source, target, from_to);
    }
}

float image::calculate_scale(const cv::Size& size, int output_width, int output_height)
{
    float      im_scale = (float)output_width / (float)size.width;
    cv::Size2f result   = size;
    result              = result * im_scale;
    if (result.height > output_height)
    {
        im_scale = (float)output_height / (float)size.height;
    }
    return im_scale;
}

cv::Size2f image::cropbox_max_proportional(const cv::Size2f& in_size, const cv::Size2f& out_size)
{
    cv::Size2f result = out_size;
    float      scale  = in_size.width / result.width;
    result            = result * scale;
    if (result.height > in_size.height)
    {
        scale  = in_size.height / result.height;
        result = result * scale;
    }
    return result;
}

cv::Size2f image::cropbox_linear_scale(const cv::Size2f& in_size, float scale)
{
    return in_size * scale;
}

cv::Size2f image::cropbox_area_scale(const cv::Size2f& in_size,
                                     const cv::Size2f& cropbox_size,
                                     float             scale)
{
    cv::Size2f result     = cropbox_size;
    float      in_area    = in_size.area();
    float      crop_area  = cropbox_size.area();
    float      size_ratio = crop_area / in_area;
    if (size_ratio > scale)
    {
        float crop_aspect_ratio = cropbox_size.width / cropbox_size.height;
        crop_area               = in_area * scale;
        float w2                = crop_area * crop_aspect_ratio;
        float width             = sqrt(w2);
        float height            = crop_area / width;
        result                  = cv::Size2f(width, height);
    }
    return result;
}

cv::Point2f image::cropbox_shift(const cv::Size2f& in_size,
                                 const cv::Size2f& crop_box,
                                 float             xoff,
                                 float             yoff)
{
    cv::Point2f result;
    result.x = (in_size.width - crop_box.width) * xoff;
    result.y = (in_size.height - crop_box.height) * yoff;
    return result;
}

// get expand_ratio and expand_prob from ExpandParameter in config
void image::expand(const cv::Mat& input,
                   cv::Mat&       output,
                   cv::Size_<int> offset,
                   cv::Size_<int> output_size)
{
    if (input.cols + offset.width > output_size.width ||
        input.rows + offset.height > output_size.height || offset.area() < 0 ||
        input.size().area() <= 0)
    {
        stringstream ss;
        ss << "Invalid parameters to expand image:" << endl
           << "input size: " << input.size() << endl
           << "offset: " << offset << endl
           << "output size: " << output_size;
        throw std::invalid_argument(ss.str());
    }
    if (output_size == input.size())
    {
        output = input;
        return;
    }

    output.create(output_size, input.type());
    output.setTo(cv::Scalar(0));

    cv::Rect bbox_roi(offset.width, offset.height, input.cols, input.rows);
    input.copyTo((output)(bbox_roi));
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

const float image::photometric::_CPCA[3][3]{{0.39731118, 0.70119634, -0.59200296},
                                            {-0.81698062, -0.02354167, -0.57618440},
                                            {0.41795513, -0.71257945, -0.56351045}};
const cv::Mat image::photometric::CPCA(3, 3, CV_32FC1, (float*)_CPCA);
const cv::Mat image::photometric::CSTD(3, 1, CV_32FC1, {19.72083305, 37.09388853, 121.78006099});

image::photometric::photometric()
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
    if (lighting.size() > 0)
    {
        cv::Mat alphas(3, 1, CV_32FC1, lighting.data());
        alphas     = (CPCA * CSTD.mul(alphas)); // this is the random coloring pixel
        auto pixel = alphas.reshape(3, 1).at<cv::Scalar_<float>>(0, 0);
        inout      = (inout + pixel) / (1.0 + color_noise_std);
    }
}

/*
Implements hue shift as well as contrast, brightness, and saturation jittering using the following definitions:
Hue: Add some multiple of 2 degrees to hue channel.
Contrast: Add some multiple of the grayscale mean of the image.
Brightness: Magnify the intensity of each pixel by photometric[1]
Saturation: Add some multiple of the pixel's grayscale value to itself.
photometric is filled with uniformly distributed values prior to calling this function
*/
// adjusts contrast, brightness, and saturation according
// to values in photometric[0], photometric[1], photometric[2], respectively
void image::photometric::cbsjitter(
    cv::Mat& inout, float contrast, float brightness, float saturation, int hue)
{
    // Skip transformations if given deterministic settings
    if (brightness != 1.0 || saturation != 1.0)
    {
        /****************************
        *  BRIGHTNESS & SATURATION  *
        *****************************/
        // float data[] = {0.114, 0.587, 0.299};   // NTSC
        float         data[] = {0.0820, 0.6094, 0.3086};
        const cv::Mat GSCL(3, 1, CV_32FC1, data);
        cv::Mat       satmtx = brightness * (saturation * cv::Mat::eye(3, 3, CV_32FC1) +
                                       (1 - saturation) * cv::Mat::ones(3, 1, CV_32FC1) * GSCL.t());
        cv::transform(inout, inout, satmtx);
    }

    if (hue != 0)
    {
        /*************
        *  HUE SHIFT *
        **************/
        cv::Mat hsv;
        // Convert to HSV colorspae.
        cv::cvtColor(inout, hsv, CV_BGR2HSV);

        // Adjust the hue.
        uint8_t* p = hsv.data;
        for (int i = 0; i < hsv.size().area(); i++)
        {
            *p = (*p + hue) % 180;
            p += 3;
        }

        // Back to BGR colorspace.
        cvtColor(hsv, inout, CV_HSV2BGR);
    }

    if (contrast != 1.0)
    {
        /*************
        *  CONTRAST  *
        **************/
        cv::Mat dst_img;
        inout.convertTo(dst_img, CV_32FC3, contrast);
        dst_img += (1.0 - contrast) * cv::mean(inout);
        dst_img.convertTo(inout, CV_8UC3);
    }
}

// void image::photometric::cbs(cv::Mat& inout, float contrast, float brightness, float saturation)
// {
//     /****************************
//     *  BRIGHTNESS & SATURATION  *
//     *****************************/

//     const cv::Mat GSCL(3, 1, CV_32FC1, {0.114, 0.587, 0.299});
//     cv::Mat satmtx = brightness * (saturation * cv::Mat::eye(3, 3, CV_32FC1) +
//                             (1 - saturation) * cv::Mat::ones(3, 1, CV_32FC1) * GSCL.t());
//     cv::transform(inout, inout, satmtx);

//     INFO << "\n" << satmtx;

//     /*************
//     *  CONTRAST  *
//     **************/
//     Mat gray_mean;
//     cv::cvtColor(Mat(1, 1, CV_32FC3, cv::mean(inout)), gray_mean, CV_BGR2GRAY);
//     inout = contrast * inout + (1 - contrast) * gray_mean.at<Scalar_<float>>(0, 0);
// }

// void image::photometric::transform_hsv(cv::Mat& inout, const float hue, const float saturation, const float brightness )
// {
//     cv::Mat hsv;
//     cv::cvtColor(inout, hsv, CV_BGR2HSV);
//     if (brightness != 1.0 || saturation != 1.0)
//     {
//         hsv = hsv.mul(cv::Scalar(1.0, saturation, brightness));
//     }
//     if (hue != 0)
//     {
//         hue /= 2; // hue is 0-360, but opencv used 0-180 to fit in a byte.
//         uint8_t* p = hsv.data;
//         for (int i = 0; i < hsv.size().area(); i++)
//         {
//             *p = (*p + hue) % 180;
//             p += 3;
//         }
//     }
//     cv::cvtColor(hsv, inout, CV_HSV2BGR);
// }

// void image::photometric::transform_hsv(cv::Mat& image, const float h_gain, const float s_gain, const float v_gain )
// {
//     const float VSU = v_gain*s_gain*cos(h_gain*M_PI/180);
//     const float VSW = v_gain*s_gain*sin(h_gain*M_PI/180);

//     INFO << "hue " << h_gain;
//     INFO << "sat " << s_gain;
//     INFO << "val " << v_gain;

//     uint8_t* p = image.data;
//     for (int i = 0; i < image.size().area(); i++)
//     {
//         uint8_t& b = p[0];
//         uint8_t& g = p[1];
//         uint8_t& r = p[2];

//         float tr   = (.299*v_gain+.701*VSU+.168*VSW)*(float)r
//             + (.587*v_gain-.587*VSU+.330*VSW)*(float)g
//             + (.114*v_gain-.114*VSU-.497*VSW)*(float)b;
//         float tg   = (.299*v_gain-.299*VSU-.328*VSW)*(float)r
//             + (.587*v_gain+.413*VSU+.035*VSW)*(float)g
//             + (.114*v_gain-.114*VSU+.292*VSW)*(float)b;
//         float tb   = (.299*v_gain-.300*VSU+1.25*VSW)*(float)r
//             + (.587*v_gain-.588*VSU-1.05*VSW)*(float)g
//             + (.114*v_gain+.886*VSU-.203*VSW)*(float)b;

//         r = (uint8_t)min<int>((tr+0.5), 255);
//         g = (uint8_t)min<int>((tg+0.5), 255);
//         b = (uint8_t)min<int>((tb+0.5), 255);

//         p += 3;
//     }
// }
