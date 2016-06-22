#include "etl_localization.hpp"

using namespace std;

template<typename T> string join(const T& v, const string& sep) {
    ostringstream ss;
    for(const auto& x : v) {
        if(&x != &v[0]) ss << sep;
        ss << x;
    }
    return ss.str();
}

cv::Mat nervana::localization::transformer::generate_anchors(int base_size, const vector<float>& ratios, const vector<float>& scales) {
    vector<float> anchor = {0.,0.,(float)(base_size-1),(float)(base_size-1)};
    cv::Mat ratio_anchors = ratio_enum(anchor, ratios);

    cv::Mat result(ratio_anchors.rows*scales.size(), 4, CV_32FC1);
    float* result_ptr = result.ptr<float>();
    for(int row=0; row<ratio_anchors.rows; row++) {
        cv::Mat row_data = ratio_anchors.row(row);
        vector<float> row_values;
        for(int col=0; col<row_data.cols; col++) {
            row_values.push_back(row_data.at<float>(col));
        }
        cv::Mat se = scale_enum(row_values, scales);
        float* se_ptr = se.ptr<float>();
        for(int i=0; i<se.size().area(); i++) {
            *result_ptr++ = *se_ptr++;
        }
    }

    return result;
}

cv::Mat nervana::localization::transformer::mkanchors(const vector<float>& ws, const vector<float>& hs, float x_ctr, float y_ctr) {
    cv::Mat rc(ws.size(),4,CV_32FC1);
    for(int i=0; i<ws.size(); i++) rc.at<float>(i,0) = x_ctr - 0.5 *(ws[i]-1);
    for(int i=0; i<ws.size(); i++) rc.at<float>(i,1) = y_ctr - 0.5 *(hs[i]-1);
    for(int i=0; i<ws.size(); i++) rc.at<float>(i,2) = x_ctr + 0.5 *(ws[i]-1);
    for(int i=0; i<ws.size(); i++) rc.at<float>(i,3) = y_ctr + 0.5 *(hs[i]-1);
    return rc;
}

cv::Mat nervana::localization::transformer::ratio_enum(const vector<float>& anchor, const vector<float>& ratios) {
    float w;
    float h;
    float x_ctr;
    float y_ctr;
    tie(w,h,x_ctr,y_ctr) = whctrs(anchor);

    int size = w * h;
    vector<float> size_ratios;
    vector<float> ws;
    vector<float> hs;
    for(float ratio : ratios)      { size_ratios.push_back(size/ratio); }
    for(float sr : size_ratios)    { ws.push_back(round(sqrt(sr))); }
    for(int i=0; i<ws.size(); i++) { hs.push_back(round(ws[i]*ratios[i])); }

    return mkanchors(ws, hs, x_ctr, y_ctr);
}

cv::Mat nervana::localization::transformer::scale_enum(const vector<float>& anchor, const vector<float>& scales) {
    float w;
    float h;
    float x_ctr;
    float y_ctr;
    tie(w,h,x_ctr,y_ctr) = whctrs(anchor);

    vector<float> ws;
    vector<float> hs;
    for(float scale : scales) {
        ws.push_back(w*scale);
        hs.push_back(h*scale);
    }
    return mkanchors(ws, hs, x_ctr, y_ctr);
}

tuple<float,float,float,float> nervana::localization::transformer::whctrs(const vector<float>& anchor) {
    float w = anchor[2] - anchor[0] + 1;
    float h = anchor[3] - anchor[1] + 1;
    float x_ctr = (float)anchor[0] + 0.5 * (float)(w - 1);
    float y_ctr = (float)anchor[1] + 0.5 * (float)(h - 1);
    return make_tuple(w, h, x_ctr, y_ctr);
}
