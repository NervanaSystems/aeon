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

std::ostream& operator<<(std::ostream&, const nervana::normalized_boundingbox::box&);
std::ostream& operator<<(std::ostream&, const std::vector<nervana::normalized_boundingbox::box>&);

class nervana::normalized_boundingbox::box final : public nervana::box
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
        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Creating improperly normalized box in constructor of normalized_boundingbox.";
            throw;
        }
    }
    ~box() {}
    float    width() const override { return m_xmax - m_xmin; }
    float    height() const override { return m_ymax - m_ymin; }
    cv::Rect rect() const override
    {
        throw std::runtime_error(
            "Called rect() on a normalized box. cv::Rect supports integers only.");
    }

    float jaccard_overlap(const nervana::normalized_boundingbox::box& second_bbox) const;
    float coverage(const nervana::normalized_boundingbox::box& second_bbox) const;
    nervana::normalized_boundingbox::box
        intersect(const nervana::normalized_boundingbox::box& second_bbox) const;

    const static nervana::normalized_boundingbox::box& zerobox()
    {
        const static normalized_boundingbox::box b(0, 0, 0, 0);
        return b;
    }

    bool operator==(const nervana::normalized_boundingbox::box& b) const;
    bool operator!=(const nervana::normalized_boundingbox::box& b) const;
    normalized_boundingbox::box& operator=(const nervana::normalized_boundingbox::box& b);
    normalized_boundingbox::box operator+(const nervana::normalized_boundingbox::box& b) const;
    normalized_boundingbox::box rescale(float x, float y) const;

    int  label() const { return m_label; }
    bool difficult() const { return m_difficult; }
    bool truncated() const { return m_truncated; }
    bool is_properly_normalized() const;

    const static int  default_label     = -1;
    const static bool default_difficult = false;
    const static bool default_truncated = false;

protected:
    void throw_if_improperly_normalized() const;

    int  m_label     = default_label;
    bool m_difficult = default_difficult;
    bool m_truncated = default_truncated;
};
