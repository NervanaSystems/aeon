#pragma once

#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace nervana {
    namespace image {
        // These functions may be common across different transformers
        void resize(const cv::Mat&, cv::Mat&, const cv::Size2i&, bool interpolate=true);
        void shift_cropbox(const cv::Size2f &, cv::Rect &, float, float);
        void rotate(const cv::Mat& input, cv::Mat& output, int angle, bool interpolate=true, const cv::Scalar& border=cv::Scalar());
        void convertMixChannels(std::vector<cv::Mat>& source, std::vector<cv::Mat>& target, std::vector<int>& from_to);

        std::tuple<float,cv::Size> calculate_scale_shape(cv::Size size, int min_size, int max_size);

        class photometric {
        public:
            photometric();
            void lighting(cv::Mat& inout, std::vector<float>, float color_noise_std);
            void cbsjitter(cv::Mat& inout, const std::vector<float>&);

            // These are the eigenvectors of the pixelwise covariance matrix
            const float _CPCA[3][3];
            const cv::Mat CPCA;

            // These are the square roots of the eigenvalues of the pixelwise covariance matrix
            const cv::Mat CSTD;

            // This is the set of coefficients for converting BGR to grayscale
            const cv::Mat GSCL;
        };
    }
}