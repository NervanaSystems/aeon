#include "restruct.hpp"


virtual shared_ptr<DecodedMedia> ImageExtractor::decode(char* inbuf, int insize) override
{
    auto ip = dynamic_cast<shared_ptr<ImageParams>>(_extract_params);
    int channelCount = ip->getChannelCount();
    if (channelCount == 1 || channelCount == 3) {
        auto output = make_shared<DecodedImage>();
        bool gray = channelCount == 1;
        cv::Mat input_img(1, insize, gray ? CV_8UC1 : CV_8UC3, inbuf);
        cv::imdecode(input_img,
                     gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR,
                     output->getImgRef());
        return dynamic_cast<shared_ptr<DecodedMedia>>(output);
    } else {
        std::stringstream ss;
        ss << "Unsupported number of channels in image: " << channelCount;
        throw std::runtime_error(ss.str());
    }
}










virtual void ImageLoader::load(shared_ptr<DecodedMedia> input, char* outbuf, int outsize) override
{
    auto ilp = dynamic_cast<shared_ptr<ImageLoaderParams>>(_loader_params);
    auto img = dynamic_cast<shared_ptr<DecodedImage>>(input)->getImgRef();

    split(img, outbuf, outsize);
}

void ImageLoader::split(Mat& img, char* buf, int bufSize) {
    Size2i size = img.size();
    if (img.channels() * img.total() > (uint) bufSize) {
        throw std::runtime_error("Decode failed - buffer too small");
    }
    if (img.channels() == 1) {
        memcpy(buf, img.data, img.total());
        return;
    }

    // Split into separate channels
    Mat blue(size, CV_8U, buf);
    Mat green(size, CV_8U, buf + size.area());
    Mat red(size, CV_8U, buf + 2 * size.area());

    Mat channels[3] = {blue, green, red};
    cv::split(img, channels);
}
