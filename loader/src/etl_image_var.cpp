#include "etl_image_var.hpp"
#include "etl_localization.hpp"

using namespace std;
using namespace nervana;

void image_var::params::dump(ostream & ostr)
{
    ostr << "Flip: " << flip << "\n";
}


/* Extract */
image_var::extractor::extractor(const image_var::config& cfg)
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

shared_ptr<image_var::decoded> image_var::extractor::extract(const char* inbuf, int insize)
{
    cv::Mat output_img;

    // It is bad to cast away const, but opencv does not support a const Mat
    // The Mat is only used for imdecode on the next line so it is OK here
    cv::Mat input_img(1, insize, _pixel_type, const_cast<char*>(inbuf));
    cv::imdecode(input_img, _color_mode, &output_img);

    return make_shared<image_var::decoded>(output_img);
}


/* Transform:
    image::config will be a supplied bunch of params used by this provider.
    on each record, the transformer will use the config along with the supplied
    record to fill a transform_params structure which will have

    Spatial distortion params:
    randomly determined flip (based on params->flip)

*/

image_var::transformer::transformer(const image_var::config& cfg)
{
    min_size = cfg.min_size;
    max_size = cfg.max_size;
}

shared_ptr<image_var::decoded> image_var::transformer::transform(
                                                 shared_ptr<image_var::params> img_xform,
                                                 shared_ptr<image_var::decoded> img)
{
    cv::Mat image = img->get_image();
    cv::Mat output;
    cv::Size im_size;
    float im_scale;
    tie(im_scale, im_size) = localization::transformer::calculate_scale_shape(image.size(), min_size, max_size);

    nervana::image::resize(image, output, im_size);

    if (img_xform->flip) {
        cv::Mat flippedImage;
        cv::flip(output, flippedImage, 1);
        output = flippedImage;
    }

    auto rc = make_shared<image_var::decoded>(output);
    return rc;
}

shared_ptr<image_var::params>
image_var::param_factory::make_params(shared_ptr<const decoded> input)
{
    auto imgstgs = shared_ptr<image_var::params>();

    imgstgs->flip  = _cfg.flip_distribution(_dre);

    return imgstgs;
}

image_var::loader::loader(const image_var::config& cfg)
{
    _channel_major = cfg.channel_major;
    _load_size     = 1;
}

void image_var::loader::load(const vector<void*>& outlist, shared_ptr<image_var::decoded> input)
{
    char* outbuf = (char*)outlist[0];
    auto img = input->get_image();
    int image_size = img.channels() * img.total();

    if (_channel_major) {
        this->split(img, outbuf);
    } else {
        memcpy(outbuf, img.data, image_size);
    }
}

void image_var::loader::split(cv::Mat& img, char* buf)
{
    // split `img` into individual channels
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
