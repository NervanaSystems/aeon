#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace nervana {
    namespace image {
        // These functions may be common across different transformers
        void resize(const cv::Mat&, cv::Mat&, const cv::Size2i&, bool interpolate=true);
        void shift_cropbox(const cv::Size2f &, cv::Rect &, float, float);
        void rotate(const cv::Mat& input, cv::Mat& output, int angle, bool interpolate=true, const cv::Scalar& border=cv::Scalar());
    }
}