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

#pragma once
#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "media.hpp"

typedef struct {
    cv::Rect cropBox;
    int angle;
    bool flip;
    float colornoise[3];  //pixelwise random values
    float cbs[3];  // contrast, brightness, saturation
} AugParams;

// These are the eigenvectors of the pixelwise covariance matrix
static float _CPCA[3][3] = {{0.39731118,  0.70119634, -0.59200296},
                    {-0.81698062, -0.02354167, -0.5761844},
                    {0.41795513, -0.71257945, -0.56351045}};
const cv::Mat CPCA(3, 3, CV_32FC1, _CPCA);

// These are the square roots of the eigenvalues of the pixelwise covariance matrix
const cv::Mat CSTD(3, 1, CV_32FC1, {19.72083305, 37.09388853, 121.78006099});

// This is the set of coefficients for converting BGR to grayscale
const cv::Mat GSCL(3, 1, CV_32FC1, {0.114, 0.587, 0.299});

extern void resizeInput(std::vector<char> &jpgdata, int maxDim);

class ImageParams : public MediaParams {
public:
    ImageParams(int channelCount, int height, int width,
                bool center, bool flip,
                int scaleMin, int scaleMax,
                int contrastMin, int contrastMax,
                int rotateMin, int rotateMax,
                int aspectRatio, bool subtractMean,
                int redMean, int greenMean, int blueMean,
                int grayMean);

    void dump();

    void getDistortionValues(cv::RNG &rng, const cv::Size2i &inputSize, AugParams *agp);

    const cv::Size2i getSize();

public:
    int                         _channelCount;
    int                         _height;
    int                         _width;
    bool                        _center;
    bool                        _flip;
    // Pixel scale to jitter at (image from which to crop will have
    // short side in [scaleMin, Max])
    int                         _scaleMin;
    int                         _scaleMax;
    int                         _contrastMin;
    int                         _contrastMax;
    int                         _rotateMin;
    int                         _rotateMax;
    int                         _aspectRatio;
    bool                        _subtractMean;
    int                         _redMean;
    int                         _greenMean;
    int                         _blueMean;
    int                         _grayMean;
    float                       _colorNoiseStd;
};

class ImageIngestParams : public MediaParams {
public:
    ImageIngestParams(bool resizeAtIngest, bool lossyEncoding,
                      int sideMin, int sideMax);

public:
    bool                        _resizeAtIngest;
    bool                        _lossyEncoding;
    // Minimum value of the short side
    int                         _sideMin;
    // Maximum value of the short side
    int                         _sideMax;

};

class Image: public Media {
friend class Video;
public:
    Image(ImageParams *params, ImageIngestParams* ingestParams, int id);

    void transform(char* item, int itemSize, char* buf, int bufSize);

    void dump_agp();

    void ingest(char** dataBuf, int* dataBufLen, int* dataLen);

    void save_binary(char *filn, char* item, int itemSize, char* buf);

private:
    void decode(char* item, int itemSize, cv::Mat* dst);

    void transformDecodedImage(const cv::Mat& decodedImage, char* buf, int bufSize);

    void rotate(const cv::Mat& input, cv::Mat& output, int angle);

    void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);

    /*
    Implements colorspace noise perturbation as described in:
    Krizhevsky et. al., "ImageNet Classification with Deep Convolutional Neural Networks"
    Constructs a random coloring pixel that is uniformly added to every pixel of the image.
    pixelstd is filled with normally distributed values prior to calling this function.
    */
    void lighting(cv::Mat& inout, float pixelstd[]);

    /*
    Implements contrast, brightness, and saturation jittering using the following definitions:
    Contrast: Add some multiple of the grayscale mean of the image.
    Brightness: Magnify the intensity of each pixel by cbs[1]
    Saturation: Add some multiple of the pixel's grayscale value to itself.
    cbs is filled with uniformly distributed values prior to calling this function
    */
    // adjusts contrast, brightness, and saturation according
    // to values in cbs[0], cbs[1], cbs[2], respectively
    void cbsjitter(cv::Mat& inout, float cbs[]);

    void split(cv::Mat& img, char* buf, int bufSize);

    void createRandomAugParams(const cv::Size2i& size);

private:
    ImageParams*                _params;
    ImageIngestParams*          _ingestParams;
    cv::Size2i                  _innerSize;
    cv::RNG                     _rng;
    int                         _numPixels;
    AugParams                   _augParams;
};
