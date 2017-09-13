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

#pragma once

#include <ostream>
#include <opencv2/core/core.hpp>

namespace nervana
{
    class box;
}

class nervana::box
{
public:
    box(bool normalized = false)
        : m_xmin(0.f)
        , m_ymin(0.f)
        , m_xmax(0.f)
        , m_ymax(0.f)
        , m_normalized(normalized)
    {
    }

    box(float _xmin, float _ymin, float _xmax, float _ymax, bool normalized = false)
        : m_xmin{_xmin}
        , m_ymin{_ymin}
        , m_xmax{_xmax}
        , m_ymax{_ymax}
        , m_normalized{normalized}
    {
        if (m_normalized)
        {
            try
            {
                throw_if_improperly_normalized();
            }
            catch (std::exception& e)
            {
                std::stringstream ss;
                ss << "Constructor of the box object: " << e.what();
                throw std::invalid_argument(ss.str());
            }
        }
    }

    box(const box& b);
    box(const box&& b);

    box& operator=(const box& b);

    box operator+(const box& b) const
    {
        return box(m_xmin + b.m_xmin,
                   m_ymin + b.m_ymin,
                   m_xmax + b.m_xmax,
                   m_ymax + b.m_ymax,
                   m_normalized);
    }
    box operator*(float v) const
    {
        return box(m_xmin * v, m_ymin * v, (m_xmax + 1) * v - 1, (m_ymax + 1) * v - 1);
    }
    bool operator==(const box& b) const;

    cv::Rect rect() const
    {
        return cv::Rect(
            std::round(m_xmin), std::round(m_ymin), std::round(width()), std::round(height()));
    }
    float xcenter() const;
    float ycenter() const;
    void set_xmin(float x) { m_xmin = x; }
    float               xmin() const { return m_xmin; }
    void set_ymin(float y) { m_ymin = y; }
    float               ymin() const { return m_ymin; }
    void set_xmax(float xmax) { m_xmax = xmax; }
    float               xmax() const { return m_xmax; }
    void set_ymax(float ymax) { m_ymax = ymax; }
    float               ymax() const { return m_ymax; }
    float               width() const;
    float               height() const;
    void set_normalized(bool normalized) { m_normalized = normalized; }
    bool                     normalized() const { return m_normalized; }
    bool                     is_properly_normalized() const;
    void                     throw_if_improperly_normalized() const;

    const static nervana::box& zerobox();

protected:
    float m_xmin;
    float m_ymin;
    float m_xmax;
    float m_ymax;

    bool m_normalized;
};

std::ostream& operator<<(std::ostream& out, const nervana::box& b);
std::ostream& operator<<(std::ostream& out, const std::vector<nervana::box>& list);
