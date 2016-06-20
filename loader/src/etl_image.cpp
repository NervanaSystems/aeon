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
image::extractor::extractor(shared_ptr<const image::config> cfg)
{
    if (!(cfg->channels == 1 || cfg->channels == 3))
    {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << cfg->channels;
        throw std::runtime_error(ss.str());
    } else {
        _pixel_type = cfg->channels == 1 ? CV_8UC1 : CV_8UC3;
        _color_mode = cfg->channels == 1 ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
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

image::transformer::transformer(shared_ptr<const image::config>) :
    _CPCA{{0.39731118,  0.70119634, -0.59200296},
                            {-0.81698062, -0.02354167, -0.5761844},
                            {0.41795513, -0.71257945, -0.56351045}},
    CPCA(3, 3, CV_32FC1, (float*)_CPCA),
    CSTD(3, 1, CV_32FC1, {19.72083305, 37.09388853, 121.78006099}),
    GSCL(3, 1, CV_32FC1, {0.114, 0.587, 0.299})
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
        cbsjitter(resizedImage, img_xform->photometric);
        lighting(resizedImage, img_xform->lighting, img_xform->color_noise_std);

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

void image::transformer::rotate(const cv::Mat& input, cv::Mat& output, int angle)
{
    if (angle == 0) {
        output = input;
    } else {
        cv::Point2i pt(input.cols / 2, input.rows / 2);
        cv::Mat rot = cv::getRotationMatrix2D(pt, angle, 1.0);
        cv::warpAffine(input, output, rot, input.size());
    }
}

void image::resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size)
{
    if (size == input.size()) {
        output = input;
    } else {
        int inter = input.size().area() < size.area() ? CV_INTER_CUBIC : CV_INTER_AREA;
        cv::resize(input, output, size, 0, 0, inter);
    }
}

/*
Implements colorspace noise perturbation as described in:
Krizhevsky et. al., "ImageNet Classification with Deep Convolutional Neural Networks"
Constructs a random coloring pixel that is uniformly added to every pixel of the image.
lighting is filled with normally distributed values prior to calling this function.
*/
void image::transformer::lighting(cv::Mat& inout, vector<float> lighting, float color_noise_std)
{
    // Skip transformations if given deterministic settings
    if (lighting.size() > 0) {
        cv::Mat alphas(3, 1, CV_32FC1, lighting.data());
        alphas = (CPCA * CSTD.mul(alphas));  // this is the random coloring pixel
        auto pixel = alphas.reshape(3, 1).at<cv::Scalar_<float>>(0, 0);
        inout = (inout + pixel) / (1.0 + color_noise_std);
    }
}

/*
Implements contrast, brightness, and saturation jittering using the following definitions:
Contrast: Add some multiple of the grayscale mean of the image.
Brightness: Magnify the intensity of each pixel by photometric[1]
Saturation: Add some multiple of the pixel's grayscale value to itself.
photometric is filled with uniformly distributed values prior to calling this function
*/
// adjusts contrast, brightness, and saturation according
// to values in photometric[0], photometric[1], photometric[2], respectively
void image::transformer::cbsjitter(cv::Mat& inout, const vector<float>& photometric)
{
    // Skip transformations if given deterministic settings
    if (photometric.size() > 0) {
        /****************************
        *  BRIGHTNESS & SATURATION  *
        *****************************/
        cv::Mat satmtx = photometric[1] * (photometric[2] * cv::Mat::eye(3, 3, CV_32FC1) +
                                (1 - photometric[2]) * cv::Mat::ones(3, 1, CV_32FC1) * GSCL.t());
        cv::transform(inout, inout, satmtx);

        /*************
        *  CONTRAST  *
        **************/
        cv::Mat gray_mean;
        cv::cvtColor(cv::Mat(1, 1, CV_32FC3, cv::mean(inout)), gray_mean, CV_BGR2GRAY);
        inout = photometric[0] * inout + (1 - photometric[0]) * gray_mean.at<cv::Scalar_<float>>(0, 0);
    }
}

shared_ptr<image::params>
image::param_factory::make_params(shared_ptr<const decoded> input)
{
    auto imgstgs = make_shared<image::params>();

    imgstgs->output_size = cv::Size2i(_cfg->width, _cfg->height);

    imgstgs->angle = _cfg->angle(_dre);
    imgstgs->flip  = _cfg->flip(_dre);

    cv::Size2f in_size = input->get_image_size();

    float scale = _cfg->scale(_dre);
    float aspect_ratio = _cfg->aspect_ratio(_dre);
    scale_cropbox(in_size, imgstgs->cropbox, aspect_ratio, scale);

    float c_off_x = _cfg->crop_offset(_dre);
    float c_off_y = _cfg->crop_offset(_dre);
    shift_cropbox(in_size, imgstgs->cropbox, c_off_x, c_off_y);

    if (_cfg->lighting.stddev() != 0) {
        for( int i=0; i<3; i++ ) {
            imgstgs->lighting.push_back(_cfg->lighting(_dre));
        }
        imgstgs->color_noise_std = _cfg->lighting.stddev();
    }
    if (_cfg->photometric.a()!=_cfg->photometric.b()) {
        for( int i=0; i<3; i++ ) {
            imgstgs->photometric.push_back(_cfg->photometric(_dre));
        }
    }
    return imgstgs;
}

void image::shift_cropbox(const cv::Size2f &in_size, cv::Rect &crop_box, float xoff, float yoff)
{
    crop_box.x = (in_size.width - crop_box.width) * xoff;
    crop_box.y = (in_size.height - crop_box.height) * yoff;
}

void image::param_factory::scale_cropbox(
                            const cv::Size2f &in_size,
                            cv::Rect &crop_box,
                            float tgt_aspect_ratio,
                            float tgt_scale )
{

    float out_a_r = static_cast<float>(_cfg->width) / _cfg->height;
    float in_a_r  = in_size.width / in_size.height;

    float crop_a_r = out_a_r * tgt_aspect_ratio;

    if (_cfg->do_area_scale) {
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
    for (const float &s: _cfg->multicrop_scales) {
        cv::Size2i boxdim(short_side_in * s, short_side_in * s);
        cv::Size2i border = in_size - boxdim;
        for (cv::Point2f &offset: _cfg->offsets) {
            cv::Point2i corner(border);
            corner.x *= offset.x;
            corner.y *= offset.y;
            cropboxes.push_back(cv::Rect(corner, boxdim));
        }
    }

    auto out_imgs = make_shared<image::decoded>();
    add_resized_crops(input->get_image(0), out_imgs, cropboxes);
    if (_cfg->include_flips) {
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
        image::resize(input(cropbox), img_out, _cfg->output_size);
        out_img->add(img_out);
    }
}

image::loader::loader(shared_ptr<const image::config> cfg)
{
    _channel_major = cfg->channel_major;
}

void image::loader::load(char* outbuf, int outsize, shared_ptr<image::decoded> input)
{
    // for (int i=0; i<input->get_image_size(); i++ )
    auto img = input->get_image(0);
    int all_pixels = img.channels() * img.total();

    if (all_pixels > outsize) {
        throw std::runtime_error("Load failed - buffer too small");
    }

    if (_channel_major) {
        this->split(img, outbuf);
    } else {
        memcpy(outbuf, img.data, all_pixels);
    }
}

void image::loader::split(cv::Mat& img, char* buf)
{
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

