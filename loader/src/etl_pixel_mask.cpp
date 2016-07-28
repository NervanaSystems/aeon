#include "etl_pixel_mask.hpp"

using namespace std;
using namespace nervana;

pixel_mask::extractor::extractor(const image::config& config)
{
}

pixel_mask::extractor::~extractor()
{
}

shared_ptr<image::decoded> pixel_mask::extractor::extract(const char* inbuf, int insize)
{
    cv::Mat image;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, CV_8UC1, const_cast<char*>(inbuf));
    cv::imdecode(input_img, CV_LOAD_IMAGE_ANYDEPTH, &image);

    // convert input image to single channel if needed
    if(image.channels()>1)
    {
        // copy channel 0 from source image to channel 0 of target image where
        // target is a single channel image
        cv::Mat target(image.rows, image.cols, CV_8UC1);
        int from_to[] = {0,0};
        cv::mixChannels(&image, 1, &target, 1, from_to, 1);
        image = target;
    }

    return make_shared<image::decoded>(image);
}

pixel_mask::transformer::transformer(const image::config& config)
{
}

pixel_mask::transformer::~transformer()
{
}

std::shared_ptr<image::decoded> pixel_mask::transformer::transform(
                    std::shared_ptr<image::params> img_xform,
                    std::shared_ptr<image::decoded> image_list)
{
    if(image_list->get_image_count() != 1) throw invalid_argument("pixel_mask transform only supports a single image");

    cv::Mat rotatedImage;
    cv::Scalar border{0,0,0};
    image::rotate(image_list->get_image(0), rotatedImage, img_xform->angle, false, border);

    cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

    cv::Mat resizedImage;
    image::resize(croppedImage, resizedImage, img_xform->output_size, false);

    cv::Mat *finalImage = &resizedImage;
    cv::Mat flippedImage;
    if (img_xform->flip) {
        cv::flip(resizedImage, flippedImage, 1);
        finalImage = &flippedImage;
    }

    return make_shared<image::decoded>(*finalImage);
}
