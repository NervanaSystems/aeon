#include <vector>

#include "image.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

void resizeInput(vector<char> &jpgdata, int maxDim){
    // Takes the buffer containing encoded jpg, determines if its shortest dimension
    // is greater than maxDim.  If so, it scales it down so that the shortest dimension
    // is equal to maxDim.  equivalent to "512x512^>" for maxDim=512 geometry argument in
    // imagemagick

    Mat image(1, jpgdata.size(), CV_8UC3, &jpgdata[0]);
    Mat decodedImage = cv::imdecode(image, CV_LOAD_IMAGE_COLOR);

    int minDim = std::min(decodedImage.rows, decodedImage.cols);
    // If no resizing necessary, just return, original image still in jpgdata;
    if (minDim <= maxDim)
        return;

    vector<int> param = {CV_IMWRITE_JPEG_QUALITY, 90};
    double scaleFactor = (double) maxDim / (double) minDim;
    Mat resizedImage;
    cv::resize(decodedImage, resizedImage, Size2i(0, 0), scaleFactor, scaleFactor, CV_INTER_AREA);
    cv::imencode(".jpg", resizedImage, *(reinterpret_cast<vector<uchar>*>(&jpgdata)), param);
    return;
}

