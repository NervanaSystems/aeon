#include <memory>
#include <vector>

#include "etl_multicrop.hpp"

using namespace std;
using namespace nervana;

shared_ptr<image::decoded> multicrop::transformer::transform(
                                                shared_ptr<image::params> unused,
                                                shared_ptr<image::decoded> input)
{
    cv::Size2i in_size = input->get_image_size();
    int short_side_in = std::min(in_size.width, in_size.height);
    vector<cv::Rect> cropboxes;

    // Get the positional crop boxes
    for (const float &s: _cfg.multicrop_scales) {
        cv::Size2i boxdim(short_side_in * s, short_side_in * s);
        cv::Size2i border = in_size - boxdim;
        for (const cv::Point2f &offset: _cfg.offsets) {
            cv::Point2i corner(border);
            corner.x *= offset.x;
            corner.y *= offset.y;
            cropboxes.push_back(cv::Rect(corner, boxdim));
        }
    }

    auto out_imgs = make_shared<image::decoded>();
    add_resized_crops(input->get_image(0), out_imgs, cropboxes);
    if (_cfg.include_flips) {
        cv::Mat input_img;
        cv::flip(input->get_image(0), input_img, 1);
        add_resized_crops(input_img, out_imgs, cropboxes);
    }
    return out_imgs;
}

void multicrop::transformer::add_resized_crops(
                const cv::Mat& input,
                shared_ptr<image::decoded> &out_img,
                vector<cv::Rect> &cropboxes)
{
    for (auto cropbox: cropboxes) {
        cv::Mat img_out;
        image::resize(input(cropbox), img_out, _cfg.output_size);
        out_img->add(img_out);
    }
}
