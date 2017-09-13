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

#include <sstream>
#include "etl_boundingbox.hpp"
#include "log.hpp"

using std::ostream;
using bbox = nervana::boundingbox::box;

ostream& operator<<(ostream& out, const bbox& b)
{
    out << static_cast<nervana::box>(b) << " label=" << b.label() << " difficult=" << b.difficult()
        << " truncated=" << b.truncated();
    return out;
}

ostream& operator<<(ostream& out, const std::vector<bbox>& boxes)
{
    out << "[";
    for (const bbox& box : boxes)
    {
        out << " {" << box << "}";
    }
    out << " ]";
    return out;
}

bbox& bbox::operator=(const bbox& b)
{
    if (&b != this)
    {
        m_xmin      = b.m_xmin;
        m_ymin      = b.m_ymin;
        m_xmax      = b.m_xmax;
        m_ymax      = b.m_ymax;
        m_label     = b.m_label;
        m_difficult = b.m_difficult;
        m_truncated = b.m_truncated;
    }
    return *this;
}

bool bbox::operator==(const bbox& b) const
{
    return nervana::box::operator==(b) && m_label == b.m_label && m_difficult == b.m_difficult &&
           m_truncated == b.m_truncated;
}

bool bbox::operator!=(const bbox& b) const
{
    return !((*this) == b);
}

bbox bbox::operator+(const cv::Point& s) const
{
    bbox rc = *this;
    rc.m_xmin += s.x;
    rc.m_ymin += s.y;
    rc.m_xmax += s.x;
    rc.m_ymax += s.y;
    return rc;
}

bbox bbox::rescale(float x, float y) const
{
    bbox rc   = *this;
    rc.m_xmin = rc.m_xmin * x;
    rc.m_ymin = rc.m_ymin * y;
    rc.m_xmax = (rc.m_xmax + 1) * x - 1;
    rc.m_ymax = (rc.m_ymax + 1) * y - 1;
    return rc;
}

bbox bbox::expand(const cv::Size2i& expand_offset,
                  const cv::Size2i& expand_size,
                  const float       expand_ratio) const
{
    if (m_xmax + expand_offset.width > expand_size.width ||
        m_ymax + expand_offset.height > expand_size.height)
    {
        std::stringstream ss;
        ss << "Invalid parameters to expand boundingbox: " << *this << std::endl
           << "Expand_offset: " << expand_offset << " expand_size: " << expand_size << std::endl
           << m_xmax << " + " << expand_offset.width << " > " << expand_size.width << " || "
           << m_ymax << " + " << expand_offset.height << " > " << expand_size.height;

        throw std::invalid_argument(ss.str());
    }

    return *this + expand_offset;
}

float bbox::jaccard_overlap(const bbox& second_bbox) const
{
    bbox  intersect_bbox = intersect(second_bbox);
    float intersect_size = intersect_bbox.size();
    float bbox1_size     = size();
    float bbox2_size     = second_bbox.size();

    if (intersect_size == 0.f)
        return 0.f;
    return intersect_size / (bbox1_size + bbox2_size - intersect_size);
}

float bbox::coverage(const bbox& second_bbox) const
{
    bbox  intersect_bbox = intersect(second_bbox);
    float intersect_size = intersect_bbox.size();
    if (intersect_size > 0)
    {
        float own_size = size();
        assert(own_size != 0.0);
        return intersect_size / own_size;
    }
    else
    {
        return 0.;
    }
}

bbox bbox::intersect(const bbox& second_bbox) const
{
    // Return [0, 0, -1, -1] if there is no intersection.
    if (second_bbox.xmin() > xmax() || second_bbox.xmax() < xmin() || second_bbox.ymin() > ymax() ||
        second_bbox.ymax() < ymin())
    {
        return box();
    }

    return bbox(std::max(xmin(), second_bbox.xmin()),
                std::max(ymin(), second_bbox.ymin()),
                std::min(xmax(), second_bbox.xmax()),
                std::min(ymax(), second_bbox.ymax()));
}

nervana::normalized_box::box bbox::normalize(float width, float height) const
{
    try
    {
        return nervana::normalized_box::box(
            xmin() / width, ymin() / height, (xmax() + 1) / width, (ymax() + 1) / height);
    }
    catch (std::exception&)
    {
        ERR << "Error when normalizing boundingbox: " << (*this) << ". Range had width: " << width
            << " height: " << height;
        throw;
    }
}
