#include "etl_pixel_mask.hpp"

using namespace std;
using namespace nervana;

pixel_mask::extractor::extractor(const image::config& cfg)
{
    if (!(cfg.channels == 1 || cfg.channels == 3))
    {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg.channels;
        throw std::runtime_error(ss.str());
    } else {
        _pixel_type = cfg.channels == 1 ? CV_8UC1 : CV_8UC3;
        _color_mode = cfg.channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    }
}

pixel_mask::extractor::~extractor() {}

std::shared_ptr<image::decoded> pixel_mask::extractor::extract(const char* buf, int bufSize)
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, bufSize, _pixel_type, const_cast<char*>(buf));
    cv::imdecode(input_img, _color_mode, &output_img);

    auto rc = make_shared<image::decoded>();
    rc->add(output_img);    // don't need to check return for single image
    return rc;
}



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




pixel_mask::loader::loader(const image::config& cfg) :
    _cfg{cfg}
{
}

pixel_mask::loader::~loader()
{
}

void pixel_mask::loader::load(char* buf, std::shared_ptr<image::decoded> mp)
{
}

