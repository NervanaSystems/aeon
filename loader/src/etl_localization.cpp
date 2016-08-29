/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#include "etl_localization.hpp"
#include "box.hpp"

using namespace std;
using namespace nervana;

localization::config::config(nlohmann::json js, const image_var::config& iconfig) :
    min_size{iconfig.min_size},
    max_size{iconfig.max_size}
{
    if(js.is_null()) {
        throw std::runtime_error("missing localization config in json config");
    }

    for(auto& info : config_list) {
        info->parse(js);
    }
    verify_config("localization", config_list, js);

    // # For training, the RPN needs:
    // # 0. bounding box target coordinates
    // # 1. bounding box target masks (keep positive anchors only)
    // self.dev_y_bbtargets = self.be.zeros((self._total_anchors * 4, 1))
    // self.dev_y_bbtargets_mask = self.be.zeros((self._total_anchors * 4, 1))
    add_shape_type({total_anchors() * 4}, "float");
    add_shape_type({total_anchors() * 4}, "float");

    // # 2. anchor labels of objectness
    // # 3. objectness mask (ignore neutral anchors)
    // self.dev_y_labels_flat = self.be.zeros((1, self._total_anchors), dtype=np.int32)
    // self.dev_y_labels_mask = self.be.zeros((2 * self._total_anchors, 1), dtype=np.int32)
    add_shape_type({1, total_anchors() * 2}, "int32_t");
    add_shape_type({total_anchors() * 2, 1}, "int32_t");

    // # we also consume some metadata for the proposalLayer
    // self.im_shape = self.be.zeros((2, 1), dtype=np.int32)  # image shape
    // self.gt_boxes = self.be.zeros((64, 4), dtype=np.float32)  # gt_boxes, padded to 64
    // self.num_gt_boxes = self.be.zeros((1, 1), dtype=np.int32)  # number of gt_boxes
    // self.gt_classes = self.be.zeros((64, 1), dtype=np.int32)   # gt_classes, padded to 64
    // self.im_scale = self.be.zeros((1, 1), dtype=np.float32)    # image scaling factor
    add_shape_type({2, 1}, "int32_t");
    add_shape_type({max_gt_boxes,4}, "float");
    add_shape_type({1, 1}, "int32_t");
    add_shape_type({64, 1}, "int32_t");
    add_shape_type({1, 1}, "float");

    // 'difficult' tag for gt_boxes
    add_shape_type({max_gt_boxes,1}, "int32_t");

    label_map.clear();
    for( int i=0; i<labels.size(); i++ ) {
        label_map.insert({labels[i],i});
    }

    validate();
}

void ::localization::config::validate()
{
    if(max_size < min_size) throw invalid_argument("max_size < min_size");
}

localization::extractor::extractor(const localization::config& cfg) :
    bbox_extractor{cfg.label_map}
{
}

localization::transformer::transformer(const localization::config& _cfg) :
    cfg{_cfg},
    all_anchors{anchor::generate(_cfg)}
{
}

shared_ptr<localization::decoded> localization::transformer::transform(
                    shared_ptr<image_var::params> settings,
                    shared_ptr<localization::decoded> mp)
{
    cv::Size im_size{mp->width(), mp->height()};
    auto crop = cv::Rect(0, 0, im_size.width, im_size.height);

    float im_scale;
    im_scale = image::calculate_scale(im_size, cfg.min_size, cfg.max_size);
    im_size = cv::Size{int(unbiased_round(im_size.width*im_scale)), int(unbiased_round(im_size.height*im_scale))};
    mp->image_scale = im_scale;
    mp->output_image_size = im_size;

    vector<int> idx_inside = anchor::inside_image_bounds(im_size.width, im_size.height, all_anchors);
    vector<box> anchors_inside;
    for(int i : idx_inside) anchors_inside.push_back(all_anchors[i]);

    // compute bbox overlaps
    mp->gt_boxes = boundingbox::transformer::transform_box(mp->boxes(), crop, settings->flip, im_scale, im_scale);
    cv::Mat overlaps = bbox_overlaps(anchors_inside, mp->gt_boxes);

    vector<int> labels(overlaps.rows, -1.0);

    // assign bg labels first
    vector<float> row_max(overlaps.rows, 0.0);
    vector<float> column_max(overlaps.cols, 0.0);
    for(int row=0; row<overlaps.rows; row++) {
        for(int col=0; col<overlaps.cols; col++) {
            auto value = overlaps.at<float>(row,col);
            row_max[row]    = std::max(row_max[row],   value);
            column_max[col] = std::max(column_max[col],value);
        }
    }

    for(int row=0; row<overlaps.rows; row++) {
        if(row_max[row] < cfg.negative_overlap) {
            labels[row] = 0;
        }
    }

    // assigning fg labels
    // 1. for each gt box, anchor with higher overlaps [including ties]
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
        if(row_max[row] >= cfg.positive_overlap) {
            labels[row] = 1;
        }
    }

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
        if(index < mp->gt_boxes.size()) {
            argmax.push_back(mp->gt_boxes[index]);
        }
    }

    auto bbox_targets = compute_targets(argmax, anchors_inside);

    // map lists to original canvas
    {
        vector<int> t_labels(all_anchors.size(), -1);
        vector<target> t_bbox_targets(all_anchors.size());
        for(int i=0; i<idx_inside.size(); i++) {
            int index = idx_inside[i];
            t_labels[index] = labels[i];
            t_bbox_targets[index] = bbox_targets[i];
        }
        labels = move(t_labels);
        bbox_targets = move(t_bbox_targets);
    }

    mp->anchor_index = sample_anchors(labels,settings->debug_deterministic);
    mp->labels = labels;
    mp->bbox_targets = bbox_targets;

    return mp;
}

vector<int> localization::transformer::sample_anchors(const vector<int>& labels, bool debug)
{
    // subsample labels if needed
    int num_fg = int(cfg.foreground_fraction * cfg.rois_per_image);
    vector<int> fg_idx;
    vector<int> bg_idx;
    for(int i=0; i<labels.size(); i++) {
        if(labels[i] >= 1) {
            fg_idx.push_back(i);
        } else if(labels[i] == 0) {
            bg_idx.push_back(i);
        }
    }
    if(debug == false) {
        shuffle(fg_idx.begin(), fg_idx.end(),random);
        shuffle(bg_idx.begin(), bg_idx.end(),random);
    }
    if(fg_idx.size() > num_fg) {
        fg_idx.resize(num_fg);
    }
    int remainder = cfg.rois_per_image - fg_idx.size();
    if(bg_idx.size() > remainder) {
        bg_idx.resize(remainder);
    }

    vector<int> result_idx;
    result_idx.insert(result_idx.end(), fg_idx.begin(), fg_idx.end());
    result_idx.insert(result_idx.end(), bg_idx.begin(), bg_idx.end());

    return result_idx;
}

vector<localization::target> localization::transformer::compute_targets(const vector<box>& gt_bb, const vector<box>& rp_bb)
{
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
    }

    return targets;
}

cv::Mat localization::transformer::bbox_overlaps(const vector<box>& boxes, const vector<boundingbox::box>& bounding_boxes)
{
    // Parameters
    // ----------
    // boxes: (N, 4) ndarray of float
    // query_boxes: (K, 4) ndarray of float
    // Returns
    // -------
    // overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    uint32_t N = boxes.size();
    uint32_t K = bounding_boxes.size();
    cv::Mat overlaps(N,K,CV_32FC1);
    overlaps = 0.;
    float iw, ih, box_area;
    float ua;
    for(uint32_t k=0; k<bounding_boxes.size(); k++) {
        const box& bounding_box = bounding_boxes[k];
        box_area = (bounding_box.xmax - bounding_box.xmin + 1) *
                   (bounding_box.ymax - bounding_box.ymin + 1);
        for(uint32_t n=0; n<boxes.size(); n++) {
            const box& b = boxes[n];
            iw = min(b.xmax, bounding_box.xmax) -
                 max(b.xmin, bounding_box.xmin) + 1;
            if(iw > 0) {
                ih = min(b.ymax, bounding_box.ymax) -
                     max(b.ymin, bounding_box.ymin) + 1;
                if(ih > 0) {
                    ua = (b.xmax - b.xmin + 1.) *
                         (b.ymax - b.ymin + 1.) +
                          box_area - iw * ih;
                    overlaps.at<float>(n, k) = iw * ih / ua;
                }
            }
        }
    }
    return overlaps;
}

localization::loader::loader(const localization::config& cfg)
{
    total_anchors = cfg.total_anchors();
    shape_type_list = cfg.get_shape_type_list();
    max_gt_boxes = cfg.max_gt_boxes;
}

void localization::loader::load(const vector<void*>& buf_list, std::shared_ptr<localization::decoded> mp)
{
    // # 0. bounding box target coordinates
    // # 1. bounding box target masks (keep positive anchors only)
    // self.dev_y_bbtargets = self.be.zeros((self._total_anchors * 4, 1))
    // self.dev_y_bbtargets_mask = self.be.zeros((self._total_anchors * 4, 1))
    float*   bbtargets          = (float*)buf_list[0];
    float*   bbtargets_mask     = (float*)buf_list[1];

    // # 2. anchor labels of objectness
    // # 3. objectness mask (ignore neutral anchors)
    // self.dev_y_labels_flat = self.be.zeros((1, self._total_anchors), dtype=np.int32)
    // self.dev_y_labels_mask = self.be.zeros((2 * self._total_anchors, 1), dtype=np.int32)
    int32_t* labels_flat        = (int32_t*)buf_list[2];
    int32_t* labels_mask        = (int32_t*)buf_list[3];

    // # we also consume some metadata for the proposalLayer
    // self.im_shape = self.be.zeros((2, 1), dtype=np.int32)  # image shape
    // self.gt_boxes = self.be.zeros((64, 4), dtype=np.float32)  # gt_boxes, padded to 64
    // self.num_gt_boxes = self.be.zeros((1, 1), dtype=np.int32)  # number of gt_boxes
    // self.gt_classes = self.be.zeros((64, 1), dtype=np.int32)   # gt_classes, padded to 64
    // self.im_scale = self.be.zeros((1, 1), dtype=np.float32)    # image scaling factor
    int32_t* im_shape           = (int32_t*)buf_list[4];
    float*   gt_boxes           = (float*  )buf_list[5];
    int32_t* num_gt_boxes       = (int32_t*)buf_list[6];
    int32_t* gt_classes         = (int32_t*)buf_list[7];
    float*   im_scale           = (float*  )buf_list[8];
    int32_t* gt_difficult       = (int32_t*)buf_list[9];

    // Initialize all of the buffers
    for(int i = 0; i<total_anchors * 4; i++) bbtargets[i] = 0.;
    for(int i = 0; i<total_anchors * 4; i++) bbtargets_mask[i] = 0.;
    for(int i = 0; i<total_anchors;     i++) labels_flat[i] = 1;
    for(int i = 0; i<total_anchors;     i++) labels_flat[i + total_anchors] = 0;
    for(int i = 0; i<total_anchors * 2; i++) labels_mask[i] = 0;

    for(int index : mp->anchor_index) {
        if(mp->labels[index] == 1) {
            labels_flat[index + total_anchors * 0] = 0;
            labels_flat[index + total_anchors * 1] = 1;

            bbtargets_mask[index + total_anchors * 0] = 1.;
            bbtargets_mask[index + total_anchors * 1] = 1.;
            bbtargets_mask[index + total_anchors * 2] = 1.;
            bbtargets_mask[index + total_anchors * 3] = 1.;
        }
        labels_mask[index] = 1;
        labels_mask[index+total_anchors] = 1;

        bbtargets[index + total_anchors * 0] = mp->bbox_targets[index].dx;
        bbtargets[index + total_anchors * 1] = mp->bbox_targets[index].dy;
        bbtargets[index + total_anchors * 2] = mp->bbox_targets[index].dw;
        bbtargets[index + total_anchors * 3] = mp->bbox_targets[index].dh;
    }

    im_shape[0] = mp->output_image_size.width;
    im_shape[1] = mp->output_image_size.height;

    *num_gt_boxes = min(max_gt_boxes, mp->gt_boxes.size());
    for(int i=0; i<*num_gt_boxes; i++) {
        const boundingbox::box& gt = mp->gt_boxes[i];
        *gt_boxes++ = gt.xmin;
        *gt_boxes++ = gt.ymin;
        *gt_boxes++ = gt.xmax;
        *gt_boxes++ = gt.ymax;
        *gt_classes++ = gt.label;
        *gt_difficult++ = gt.difficult;
    }
    for(int i=*num_gt_boxes; i<max_gt_boxes; i++) {
        *gt_boxes++ = 0;
        *gt_boxes++ = 0;
        *gt_boxes++ = 0;
        *gt_boxes++ = 0;
        *gt_classes++ = 0;
        *gt_difficult++ = 0;
    }

    *im_scale = mp->image_scale;
}

vector<box> localization::anchor::generate(const localization::config& cfg)
{
    int conv_size = int(std::floor(cfg.max_size * cfg.scaling_factor));

    vector<box> anchors = generate_anchors(cfg.base_size, cfg.ratios, cfg.scales);
    std::vector<box> all_anchors;

    // generate shifts to apply to anchors
    // note: 1/SCALE is the feature stride
    vector<float> shift_x;
    vector<float> shift_y;
    for(float i=0; i<conv_size; i++) {
        shift_x.push_back(i * 1. / cfg.scaling_factor);
        shift_y.push_back(i * 1. / cfg.scaling_factor);
    }

    vector<box> shifts;
    for(int y=0; y<shift_y.size(); y++) {
        for(int x=0; x<shift_x.size(); x++) {
            shifts.emplace_back(shift_x[x], shift_y[y], shift_x[x], shift_y[y]);
        }
    }

    for(const box& anchor_data : anchors ) {
        for(const box& row_data : shifts ) {
            box b = row_data+anchor_data;
            all_anchors.push_back(b);
        }
    }
    return all_anchors;
}

vector<int> localization::anchor::inside_image_bounds(int width, int height, const vector<box>& all_anchors)
{
    vector<int> rc;
    for(int i=0; i<all_anchors.size(); i++) {
        const box& b = all_anchors[i];
        if( b.xmin >= 0 && b.ymin >= 0 && b.xmax < width && b.ymax < height ) {
            rc.emplace_back(i);
        }
    }
    return rc;
}

vector<box> localization::anchor::generate_anchors(size_t base_size, const vector<float>& ratios, const vector<float>& scales)
{
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

vector<box> localization::anchor::mkanchors(const vector<float>& ws, const vector<float>& hs, float x_ctr, float y_ctr)
{
    vector<box> rc;
    for(int i=0; i<ws.size(); i++) {
        rc.emplace_back(x_ctr - 0.5 *(ws[i]-1),
                        y_ctr - 0.5 *(hs[i]-1),
                        x_ctr + 0.5 *(ws[i]-1),
                        y_ctr + 0.5 *(hs[i]-1));
    }
    return rc;
}

vector<box> localization::anchor::ratio_enum(const box& anchor, const vector<float>& ratios)
{
    int size = anchor.width() * anchor.height();
    vector<float> size_ratios;
    vector<float> ws;
    vector<float> hs;
    for(float ratio : ratios)      { size_ratios.push_back(size/ratio); }
    for(float sr : size_ratios)    { ws.push_back(round(sqrt(sr))); }
    for(int i=0; i<ws.size(); i++) { hs.push_back(round(ws[i]*ratios[i])); }

    return mkanchors(ws, hs, anchor.xcenter(), anchor.ycenter());
}

vector<box> localization::anchor::scale_enum(const box& anchor, const vector<float>& scales)
{
    vector<float> ws;
    vector<float> hs;
    for(float scale : scales) {
        ws.push_back(anchor.width()*scale);
        hs.push_back(anchor.height()*scale);
    }
    return mkanchors(ws, hs, anchor.xcenter(), anchor.ycenter());
}
