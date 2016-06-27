#include "etl_localization.hpp"
#include "box.hpp"

using namespace std;
using namespace nervana;

template<typename T> string join(const T& v, const string& sep) {
    ostringstream ss;
    for(const auto& x : v) {
        if(&x != &v[0]) ss << sep;
        ss << x;
    }
    return ss.str();
}

bool nervana::localization::config::set_config(nlohmann::json js)
{
    bbox::config::set_config(js);

    parse_opt(images_per_batch, "images_per_batch", js);
    parse_opt(rois_per_image, "rois_per_image", js);
    parse_opt(min_size, "min_size", js);
    parse_opt(max_size, "max_size", js);
    parse_opt(base_size, "base_size", js);
    parse_opt(scale, "scale", js);
    parse_opt(ratios, "ratios", js);
    parse_opt(scales, "scales", js);
    parse_opt(negative_overlap, "negative_overlap", js);
    parse_opt(positive_overlap, "positive_overlap", js);
    parse_opt(foreground_fraction, "foreground_fraction", js);

    return validate();
}

bool nervana::localization::config::validate() {
    return max_size > min_size &&
           negative_overlap >= 0.0 &&
           negative_overlap <= 1.0 &&
           positive_overlap >= 0.0 &&
           positive_overlap <= 1.0 &&
           foreground_fraction <= 1.0;}

localization::extractor::extractor(std::shared_ptr<const localization::config> cfg) :
    bbox_extractor{cfg}
{

}


localization::transformer::transformer(std::shared_ptr<const localization::config> cfg) :
    _anchor{MAX_SIZE,MIN_SIZE},
    rois_per_image{cfg->rois_per_image}
{
//    cout << "anchors " << anchors.size() << endl;
//    for(const anchor::box& b : anchors) {
//        cout << "[" << b.xmin << "," << b.ymin << "," << b.xmax-b.xmin << "," << b.ymax-b.ymin << "]" << endl;
//    }
}

shared_ptr<localization::decoded> localization::transformer::transform(
                    shared_ptr<nervana::params> txs,
                    shared_ptr<localization::decoded> mp) {
    cv::Size im_size{mp->width(), mp->height()};
    float im_scale;
    tie(im_scale, im_size) = calculate_scale_shape(im_size);

    vector<box> anchors = _anchor.inside_im_bounds(im_size.width, im_size.height);

    vector<float> labels(anchors.size());
    fill_n(labels.begin(),anchors.size(),-1.);

    // compute bbox overlaps
//    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
//                         np.ascontiguousarray(db['gt_bb'] * im_scale, dtype=np.float))
    vector<box> scaled_bbox;
    for(const bbox::box& b : mp->boxes()) {
        box r{
        (float)b.xmin * im_scale,
        (float)b.ymin * im_scale,
        (float)b.xmax * im_scale,
        (float)b.ymax * im_scale};
        scaled_bbox.push_back(r);
    }
    cv::Mat overlaps = bbox_overlaps(anchors, scaled_bbox);
//    for(int row=0; row<overlaps.rows; row++) {
//        cout << row << "   ";
//        for(int col=0; col<overlaps.cols; col++) {
//            cout << setw(12) << overlaps.at<float>(row,col) << ",";
//        }
//        cout << endl;
//    }

    // assign bg labels first
    vector<float> row_max(overlaps.rows);
    vector<float> column_max(overlaps.cols);
    fill_n(row_max.begin(),overlaps.cols,0);
    fill_n(column_max.begin(),overlaps.cols,0);
    for(int row=0; row<overlaps.rows; row++) {
        for(int col=0; col<overlaps.cols; col++) {
            auto value = overlaps.at<float>(row,col);
            row_max[row]    = std::max(row_max[row],   value);
            column_max[col] = std::max(column_max[col],value);
        }
    }

    for(int row=0; row<overlaps.rows; row++) {
        if(row_max[row] < NEGATIVE_OVERLAP) {
            labels[row] = 0;
        }
    }

    // assigning fg labels
    // 1. for each gt box, anchor with higher overlaps [including ties]
//    for(float f : column_max) { cout << f << "   "; } cout << endl;
    for(int row=0; row<overlaps.rows; row++) {
        for(int col=0; col<overlaps.cols; col++) {
            // This should be fixed as it is comparing floats
            if(overlaps.at<float>(row,col) == column_max[col]) {
                labels[row] = 1;
            }
        }
    }

    // 2. any anchor above the overlap threshold with any gt box
    for(int row=0; row<overlaps.rows; row++) {
        if(row_max[row] >= POSITIVE_OVERLAP) {
            labels[row] = 1;
        }
    }

//    for(int i=0; i<labels.size(); i++) {
//        cout << i << "   " << labels[i] << endl;
//    }


    // For every anchor, compute the regression target compared
    // to the gt box that it has the highest overlap with
    // the indicies of labels should match these targets
    vector<box> argmax;
    for(int row=0; row<overlaps.rows; row++) {
        int index = 0;
        float max = 0;
        for(int col=0; col<overlaps.cols; col++) {
            auto value = overlaps.at<float>(row,col);
            if(value > max) {
                index = col;
                max = value;
            }
        }
        argmax.push_back(scaled_bbox[index]);
    }
//    cout << argmax << endl;

    auto bbox_targets = compute_targets(argmax, anchors);

    // results
    // labels
    // bbox_targets
    tie(mp->labels, mp->bbox_targets, mp->anchor_index) = sample_anchors(labels, bbox_targets);

//    cout << "result_idx\n"; for(int i=0; i<result_idx.size(); i++) { cout << "   " << i << "  " << result_idx[i] << endl; }

    return mp;
}


//def _sample_anchors(self, db, nrois, fg_fractions):
tuple<vector<float>,vector<localization::target>,vector<int>>
    localization::transformer::sample_anchors(const vector<float>& labels,
                                              const vector<target>& bbox_targets) {
    cout << "sample_anchors labels " << labels.size() << " bbox_targets " << bbox_targets.size()  << endl;
    // subsample labels if needed
    int num_fg = int(FG_FRACTION * rois_per_image);
    cout << "num_fg " << num_fg << endl;
    cout << "rois_per_image " << rois_per_image << endl;
    vector<int> fg_idx;
    vector<int> bg_idx;
    for(int i=0; i<labels.size(); i++) {
        if(labels[i] == 1.) {
            fg_idx.push_back(i);
        } else if(labels[i] == 0.) {
            bg_idx.push_back(i);
        }
    }
    cout << "fg_idx.size() " << fg_idx.size() << endl;
    cout << "bg_idx.size() " << bg_idx.size() << endl;
    if(fg_idx.size() > num_fg) {
        shuffle(fg_idx.begin(), fg_idx.end(),random);
        fg_idx.resize(num_fg);
    }
    int remainder = rois_per_image-fg_idx.size();
    if(bg_idx.size() > remainder) {
        shuffle(bg_idx.begin(), bg_idx.end(),random);
        bg_idx.resize(remainder);
    }
    cout << "post fg_idx.size() " << fg_idx.size() << endl;
    cout << "post bg_idx.size() " << bg_idx.size() << endl;

    vector<int> result_idx;
    result_idx.insert(result_idx.begin(), fg_idx.begin(), fg_idx.end());
    result_idx.insert(result_idx.begin(), bg_idx.begin(), bg_idx.end());
//    idx = np.hstack([fg_idx, bg_idx])
//    assert len(idx) == nrois
    cout << "post result_idx.size() " << result_idx.size() << endl;

    vector<float>  result_labels;
    vector<target> result_targets;
    for(int i : result_idx) {
        result_labels.push_back(labels[i]);
        result_targets.push_back(bbox_targets[i]);
    }

//    # return labels, bbox_targets, and anchor indicies
//    return (db['labels'][idx], db['bbox_targets'][idx, :], idx[:])
    return make_tuple(result_labels, result_targets, result_idx);
}

//def _compute_targets(gt_bb, rp_bb):
vector<localization::target> localization::transformer::compute_targets(const vector<box>& gt_bb, const vector<box>& rp_bb) {
    //  Given ground truth bounding boxes and proposed boxes, compute the regresssion
    //  targets according to:

    //  t_x = (x_gt - x) / w
    //  t_y = (y_gt - y) / h
    //  t_w = log(w_gt / w)
    //  t_h = log(h_gt / h)

    //  where (x,y) are bounding box centers and (w,h) are the box dimensions

    // the target will be how to adjust the bbox's center and width/height
    // note that the targets are generated based on the original RP, which has not
    // been scaled by the image resizing
    vector<target> targets;
    for(int i=0; i<gt_bb.size(); i++) {
        const box& gt = gt_bb[i];
        const box& rp = rp_bb[i];
        float dx = (gt.xcenter() - rp.xcenter()) / rp.width();
        float dy = (gt.ycenter() - rp.ycenter()) / rp.height();
        float dw = log(gt.width() / rp.width());
        float dh = log(gt.height() / rp.height());
        targets.emplace_back(dx, dy, dw, dh);

//        cout << i << "   " << dx << "," << dy << "," << dw << "," << dh << endl;
    }

    return targets;
}


cv::Mat localization::transformer::bbox_overlaps(const vector<box>& boxes, const vector<box>& query_boxes) {
    // Parameters
    // ----------
    // boxes: (N, 4) ndarray of float
    // query_boxes: (K, 4) ndarray of float
    // Returns
    // -------
    // overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    uint32_t N = boxes.size();
    uint32_t K = query_boxes.size();
    cv::Mat overlaps(N,K,CV_32FC1);
    overlaps = 0.;
    float iw, ih, box_area;
    float ua;
    uint32_t k, n;
    k = 0;
    for(const box& query_box : query_boxes) {
        box_area = (query_box.xmax - query_box.xmin + 1) *
                   (query_box.ymax - query_box.ymin + 1);
        n = 0;
        for(const box& b : boxes ) {
            iw = min(b.xmax, query_box.xmax) -
                 max(b.xmin, query_box.xmin) + 1;
            if(iw > 0) {
                ih = min(b.ymax, query_box.ymax) -
                     max(b.ymin, query_box.ymin) + 1;
                if(ih > 0) {
                    ua = (b.xmax - b.xmin + 1.) *
                         (b.ymax - b.ymin + 1.) +
                          box_area - iw * ih;
                    overlaps.at<float>(n, k) = iw * ih / ua;
                }
            }
            n++;
        }
        k++;
    }
//    return overlaps
    return overlaps;
}

tuple<float,cv::Size> localization::transformer::calculate_scale_shape(cv::Size size) {
    int im_size_min = std::min(size.width,size.height);
    int im_size_max = max(size.width,size.height);
    float im_scale = float(MIN_SIZE) / float(im_size_min);
    // Prevent the biggest axis from being more than FRCN_MAX_SIZE
    if(round(im_scale * im_size_max) > MAX_SIZE) {
        im_scale = float(MAX_SIZE) / float(im_size_max);
    }
    cv::Size im_shape{int(round(size.width*im_scale)), int(round(size.height*im_scale))};
    return make_tuple(im_scale, im_shape);
}



localization::anchor::anchor(int max_size, int min_size) :
    MAX_SIZE{max_size},
    MIN_SIZE{min_size},
    conv_size{int(std::floor(MAX_SIZE * SCALE))}
{
    all_anchors = add_anchors();
    cout << "all_anchors " << all_anchors.size() << endl;
}

vector<box> localization::anchor::inside_im_bounds(int width, int height) {
    vector<box> rc;
    for(const box& b : all_anchors) {
        if( b.xmin >= 0 && b.ymin >= 0 && b.xmax < width && b.ymax < height ) {
            rc.emplace_back(b);
        }
    }
    return rc;
}


vector<box> localization::anchor::add_anchors() {
    vector<box> anchors = generate_anchors(16,{0.5, 1., 2.},{8,16,32});

    // generate shifts to apply to anchors
    // note: 1/self.SCALE is the feature stride
    vector<float> shift_x;
    vector<float> shift_y;
    for(float i=0; i<conv_size; i++) {
        shift_x.push_back(i * 1. / SCALE);
        shift_y.push_back(i * 1. / SCALE);
    }

    vector<box> shifts;
    for(int y=0; y<shift_y.size(); y++) {
        for(int x=0; x<shift_x.size(); x++) {
            shifts.emplace_back(shift_x[x], shift_y[y], shift_x[x], shift_y[y]);
        }
    }

    vector<box> all_anchors;
    for(const box& anchor_data : anchors ) {
        for(const box& row_data : shifts ) {
            box b = row_data+anchor_data;
//            if(b.xmin>=0 && b.ymin>=0) {
                all_anchors.push_back(b);
//            }
        }
    }

    return all_anchors;
}

vector<box> localization::anchor::generate_anchors(int base_size, const vector<float>& ratios, const vector<float>& scales) {
    box anchor{0.,0.,(float)(base_size-1),(float)(base_size-1)};
    vector<box> ratio_anchors = ratio_enum(anchor, ratios);

    vector<box> result;
    for(const box& ratio_anchor : ratio_anchors) {
        for(const box& b : scale_enum(ratio_anchor, scales)) {
            result.push_back(b);
        }
    }

    return result;
}

vector<box> localization::anchor::mkanchors(const vector<float>& ws, const vector<float>& hs, float x_ctr, float y_ctr) {
    vector<box> rc;
    for(int i=0; i<ws.size(); i++) {
        rc.emplace_back(x_ctr - 0.5 *(ws[i]-1),
                        y_ctr - 0.5 *(hs[i]-1),
                        x_ctr + 0.5 *(ws[i]-1),
                        y_ctr + 0.5 *(hs[i]-1));
    }
    return rc;
}

vector<box> localization::anchor::ratio_enum(const box& anchor, const vector<float>& ratios) {
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

vector<box> localization::anchor::scale_enum(const box& anchor, const vector<float>& scales) {
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

tuple<float,float,float,float> localization::anchor::whctrs(const box& anchor) {
    float w = anchor.xmax - anchor.xmin + 1;
    float h = anchor.ymax - anchor.ymin + 1;
    float x_ctr = (float)anchor.xmin + 0.5 * (float)(w - 1);
    float y_ctr = (float)anchor.ymin + 0.5 * (float)(h - 1);
    return make_tuple(w, h, x_ctr, y_ctr);
}
