/*
 Copyright 2017 Nervana Systems Inc.
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

#pragma once

#include <opencv2/core/core.hpp>

#include "box.hpp"
#include "log.hpp"

namespace nervana
{
    namespace boundingbox
    {
        class box;
    }
}

std::ostream& operator<<(std::ostream&, const nervana::boundingbox::box&);
std::ostream& operator<<(std::ostream&, const std::vector<nervana::boundingbox::box>&);

class nervana::boundingbox::box final : public nervana::box
{
public:
    box(bool normalized = false)
        : nervana::box(normalized)
    {
    }

    box(float xmin, float ymin, float xmax, float ymax, bool normalized = false)
        : nervana::box(xmin, ymin, xmax, ymax, normalized)
    {
    }

    bool operator==(const boundingbox::box& b) const;
    bool operator!=(const boundingbox::box& b) const;
    box operator+(const cv::Size& s) const;
    void normalize(float width, float height);
    box unnormalize(float width, float height) const;
    box rescale(float x, float y) const;
    void expand_bbox(const cv::Size2i& expand_offset,
                     const cv::Size2i& expand_size,
                     const float       expand_ratio);
    float size() const;

    float jaccard_overlap(const boundingbox::box& second_bbox) const;
    float coverage(const boundingbox::box& second_bbox) const;
    boundingbox::box intersect(const boundingbox::box& second_bbox) const;

    static void normalize_bboxes(std::vector<box>& bboxes, int width, int height);

    bool difficult = false;
    bool truncated = false;
    int  label     = -1;
};
