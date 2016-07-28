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

    auto rc = make_shared<image::decoded>();
    rc->add(image);    // don't need to check return for single image
    return rc;
}

pixel_mask::transformer::transformer(const image::config& config)
{
}

pixel_mask::transformer::~transformer()
{
}

std::shared_ptr<image::decoded> pixel_mask::transformer::transform(
                    std::shared_ptr<image::params> img_xform,
                    std::shared_ptr<image::decoded> img)
{
    vector<cv::Mat> finalImageList;
    for(int i=0; i<img->get_image_count(); i++) {
        cv::Mat rotatedImage;
        cv::Scalar border{0,0,0};
        image::rotate(img->get_image(i), rotatedImage, img_xform->angle, false, border);

        cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

        cv::Mat resizedImage;
        image::resize(croppedImage, resizedImage, img_xform->output_size, false);

        cv::Mat *finalImage = &resizedImage;
        cv::Mat flippedImage;
        if (img_xform->flip) {
            cv::flip(resizedImage, flippedImage, 1);
            finalImage = &flippedImage;
        }
        finalImageList.push_back(*finalImage);
    }

    auto rc = make_shared<image::decoded>();
    if(rc->add(finalImageList) == false) {
        rc = nullptr;
    }
    return rc;
}

pixel_mask::loader::loader(const image::config& _config) :
    cfg(_config)
{

}

pixel_mask::loader::~loader()
{

}

void pixel_mask::loader::load(char* outbuf, std::shared_ptr<image::decoded> input)
{
    cv::Mat  image = input->get_image(0);
    uint8_t* data = image.data;
    int*     out = (int*)outbuf;
    int      image_size = image.total();

    for(int i=0; i<image_size; i++)
    {
        out[i] = data[i];
    }
}
