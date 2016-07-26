#include "etl_image.hpp"

using namespace std;
using namespace nervana;

void image::params::dump(ostream & ostr)
{
    ostr << "Angle: " << setw(3) << angle << " ";
    ostr << "Flip: " << flip << " ";
    ostr << "Lighting: ";
    for_each (lighting.begin(), lighting.end(), [&ostr](float &l) {ostr << l << " ";});
    ostr << "Photometric: ";
    for_each (photometric.begin(), photometric.end(), [&ostr](float &p) {ostr << p << " ";});
    ostr << "\n" << "Crop Box: " << cropbox << "\n";
}


/* Extract */
image::extractor::extractor(const image::config& cfg)
{
    if (!(cfg.channels == 1 || cfg.channels == 3))
    {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg.channels;
        throw std::runtime_error(ss.str());
    } else {
        _pixel_type = CV_MAKETYPE(CV_8U, cfg.channels);
        _color_mode = cfg.channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    }
}

shared_ptr<image::decoded> image::extractor::extract(const char* inbuf, int insize)
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, _pixel_type, const_cast<char*>(inbuf));
    cv::imdecode(input_img, _color_mode, &output_img);

    auto rc = make_shared<image::decoded>();
    rc->add(output_img);    // don't need to check return for single image
    return rc;
}


/* Transform:
    image::config will be a supplied bunch of params used by this provider.
    on each record, the transformer will use the config along with the supplied
    record to fill a transform_params structure which will have

    Spatial distortion params:
    randomly sampled crop box (based on params->center, params->aspect_ratio, params->scale_pct, record size)
    randomly determined flip (based on params->flip)
    randomly sampled rotation angle (based on params->angle)

    Photometric distortion params:
    randomly sampled contrast, brightness, saturation, lighting values (based on params->cbs, lighting bounds)

*/

image::transformer::transformer(const image::config&)
{
}

shared_ptr<image::decoded> image::transformer::transform(
                                                 shared_ptr<image::params> img_xform,
                                                 shared_ptr<image::decoded> img)
{
    vector<cv::Mat> finalImageList;
    for(int i=0; i<img->get_image_count(); i++) {
        cv::Mat rotatedImage;
        rotate(img->get_image(i), rotatedImage, img_xform->angle);

        cv::Mat croppedImage = rotatedImage(img_xform->cropbox);

        cv::Mat resizedImage;
        image::resize(croppedImage, resizedImage, img_xform->output_size);
        photo.cbsjitter(resizedImage, img_xform->photometric);
        photo.lighting(resizedImage, img_xform->lighting, img_xform->color_noise_std);

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

shared_ptr<image::params>
image::param_factory::make_params(shared_ptr<const decoded> input)
{
    auto imgstgs = shared_ptr<image::params>(new image::params());

    imgstgs->output_size = cv::Size2i(_cfg.width, _cfg.height);

    imgstgs->angle = _cfg.angle(_dre);
    imgstgs->flip  = _cfg.flip(_dre);

    cv::Size2f in_size = input->get_image_size();

    float scale = _cfg.scale(_dre);
    float aspect_ratio = _cfg.aspect_ratio(_dre);
    scale_cropbox(in_size, imgstgs->cropbox, aspect_ratio, scale);

    float c_off_x = _cfg.crop_offset(_dre);
    float c_off_y = _cfg.crop_offset(_dre);
    shift_cropbox(in_size, imgstgs->cropbox, c_off_x, c_off_y);

    if (_cfg.lighting.stddev() != 0) {
        for( int i=0; i<3; i++ ) {
            imgstgs->lighting.push_back(_cfg.lighting(_dre));
        }
        imgstgs->color_noise_std = _cfg.lighting.stddev();
    }
    if (_cfg.photometric.a()!=_cfg.photometric.b()) {
        for( int i=0; i<3; i++ ) {
            imgstgs->photometric.push_back(_cfg.photometric(_dre));
        }
    }
    return imgstgs;
}

void image::param_factory::scale_cropbox(
                            const cv::Size2f &in_size,
                            cv::Rect &crop_box,
                            float tgt_aspect_ratio,
                            float tgt_scale )
{

    float out_a_r = static_cast<float>(_cfg.width) / _cfg.height;
    float in_a_r  = in_size.width / in_size.height;

    float crop_a_r = out_a_r * tgt_aspect_ratio;

    if (_cfg.do_area_scale) {
        // Area scaling -- use pctge of original area subject to aspect ratio constraints
        float max_scale = in_a_r > crop_a_r ? crop_a_r /  in_a_r : in_a_r / crop_a_r;
        float tgt_area  = std::min(tgt_scale, max_scale) * in_size.area();

        crop_box.height = sqrt(tgt_area / crop_a_r);
        crop_box.width  = crop_box.height * crop_a_r;
    } else {
        // Linear scaling -- make the long crop box side  the scale pct of the short orig side
        float short_side = std::min(in_size.width, in_size.height);

        if (crop_a_r < 1) { // long side is height
            crop_box.height = tgt_scale * short_side;
            crop_box.width  = crop_box.height * crop_a_r;
        } else {
            crop_box.width  = tgt_scale * short_side;
            crop_box.height = crop_box.width / crop_a_r;
        }
    }
}

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

void image::loader::load(char* outbuf, shared_ptr<image::decoded> input)
{
    // TODO: Generalize this to also handle multi_crop case
    auto img = input->get_image(0);
    int image_size = img.channels() * img.total();

    for (int i=0; i < input->get_image_count(); i++) {
        auto outbuf_i = outbuf + (i * image_size);

        if (_cfg.channel_major) {
            this->split(img, outbuf_i);
        } else {
            memcpy(outbuf_i, img.data, image_size);
        }
    }
}

void image::loader::split(cv::Mat& img, char* buf)
{
    // split `img` into individual channels so that buf is in c
    int pix_per_channel = img.total();
    int num_channels = img.channels();

    if (num_channels == 1) {
        memcpy(buf, img.data, pix_per_channel);
    } else {
        // Split into separate channels
        cv::Size2i size = img.size();
        cv::Mat b(size, CV_8U, buf);
        cv::Mat g(size, CV_8U, buf + pix_per_channel);
        cv::Mat r(size, CV_8U, buf + 2 * pix_per_channel);

        cv::Mat channels[3] = {b, g, r};
        cv::split(img, channels);
    }
}
