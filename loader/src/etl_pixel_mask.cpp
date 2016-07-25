#include "etl_pixel_mask.hpp"

using namespace std;
using namespace nervana;

pixel_mask::transformer::transformer(const image::config&)
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