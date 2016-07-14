#include "image.hpp"

using namespace nervana;

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

void image::shift_cropbox(const cv::Size2f &in_size, cv::Rect &crop_box, float xoff, float yoff)
{
    crop_box.x = (in_size.width - crop_box.width) * xoff;
    crop_box.y = (in_size.height - crop_box.height) * yoff;
}
