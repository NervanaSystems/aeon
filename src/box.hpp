/*
 Copyright 2016 Intel(R) Nervana(TM)
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

#include <ostream>
#include <opencv2/core/core.hpp>

#include "util.hpp"

namespace nervana
{
    class box;
    namespace boundingbox
    {
        class box;
    }
    namespace normalized_box
    {
        class box;
    }
    std::vector<normalized_box::box>
        normalize_bboxes(const std::vector<boundingbox::box>& bboxes, int width, int height);
}

class nervana::box
{
public:
    box() = default;
    box(float xmin, float ymin, float xmax, float ymax)
        : m_xmin(xmin)
        , m_ymin(ymin)
        , m_xmax(xmax)
        , m_ymax(ymax)
    {
    }
    virtual ~box() {}
    virtual float    width() const { return m_xmax - m_xmin + 1; }
    virtual float    height() const { return m_ymax - m_ymin + 1; }
    virtual cv::Rect rect() const
    {
        return cv::Rect(
            std::round(m_xmin), std::round(m_ymin), std::round(width()), std::round(height()));
    }
    float size() const
    {
        if (xmax() < xmin() || ymax() < ymin())
        {
            // If box is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
            return 0.0f;
        }
        else
        {
            return width() * height();
        }
    }

    nervana::box operator+(const nervana::box& b) const
    {
        return nervana::box(
            m_xmin + b.xmin(), m_ymin + b.ymin(), m_xmax + b.xmax(), m_ymax + b.ymax());
    }
    nervana::box operator*(float v) const
    {
        return nervana::box(m_xmin * v, m_ymin * v, m_xmax * v, m_ymax * v);
    }
    bool operator==(const nervana::box& b) const
    {
        return nervana::almost_equal(m_xmin, b.m_xmin) && nervana::almost_equal(m_xmax, b.m_xmax) &&
               nervana::almost_equal(m_ymin, b.m_ymin) && nervana::almost_equal(m_ymax, b.m_ymax);
    }
    bool operator!=(const nervana::box& b) const { return !((*this) == b); }
    nervana::box& operator=(const nervana::box& b);

    float ymin() const { return m_ymin; }
    float xmin() const { return m_xmin; }
    float ymax() const { return m_ymax; }
    float xmax() const { return m_xmax; }
    float xcenter() const { return m_xmin + (m_xmax - m_xmin) / 2.; }
    float ycenter() const { return m_ymin + (m_ymax - m_ymin) / 2.; }
    void set_xmin(float x) { m_xmin = x; }
    void set_ymin(float y) { m_ymin = y; }
    void set_xmax(float xmax) { m_xmax = xmax; }
    void set_ymax(float ymax) { m_ymax = ymax; }
protected:
    float m_xmin = 0;
    float m_ymin = 0;
    float m_xmax = -1;
    float m_ymax = -1;
};

std::ostream& operator<<(std::ostream& out, const nervana::box& b);
std::ostream& operator<<(std::ostream& out, const std::vector<nervana::box>& list);
