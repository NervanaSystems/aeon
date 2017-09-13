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

std::ostream& operator<<(std::ostream&, const nervana::boundingbox::box&);
std::ostream& operator<<(std::ostream&, const std::vector<nervana::boundingbox::box>&);

class nervana::boundingbox::box final : public nervana::box
{
public:
    box() = default;
    box(float xmin,
        float ymin,
        float xmax,
        float ymax,
        int   label     = default_label,
        bool  difficult = default_difficult,
        bool  truncated = default_truncated)
        : nervana::box(xmin, ymin, xmax, ymax)
        , m_label(label)
        , m_difficult(difficult)
        , m_truncated(truncated)
    {
    }
    ~box() {}
    float jaccard_overlap(const nervana::boundingbox::box& second_bbox) const;
    float coverage(const nervana::boundingbox::box& second_bbox) const;
    nervana::boundingbox::box intersect(const nervana::boundingbox::box& second_bbox) const;

    bool operator==(const nervana::boundingbox::box& b) const;
    bool operator!=(const nervana::boundingbox::box& b) const;
    boundingbox::box& operator=(const nervana::boundingbox::box& b);
    boundingbox::box operator+(const cv::Point& s) const;
    boundingbox::box operator+(const boundingbox::box& b) const;
    boundingbox::box rescale(float x, float y) const;
    boundingbox::box expand(const cv::Size2i& expand_offset,
                            const cv::Size2i& expand_size,
                            const float       expand_ratio) const;
    int               label() const { return m_label; }
    bool              difficult() const { return m_difficult; }
    bool              truncated() const { return m_truncated; }
    const static int  default_label     = -1;
    const static bool default_difficult = false;
    const static bool default_truncated = false;

    nervana::normalized_box::box normalize(float width, float height) const;

protected:
    int  m_label     = default_label;
    bool m_difficult = default_difficult;
    bool m_truncated = default_truncated;
};
