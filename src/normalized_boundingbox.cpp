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
#include "util.hpp"
#include "log.hpp"

using std::ostream;
using bbox = nervana::normalized_boundingbox::box;
using nervana::almost_equal;

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

        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Error when assigning normalized bbox: " << b;
            throw;
        }
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

bbox bbox::operator+(const bbox& b) const
{
    bbox rc = *this;
    rc.m_xmin += b.xmin();
    rc.m_ymin += b.ymin();
    rc.m_xmax += b.xmax();
    rc.m_ymax += b.ymax();
    return rc;
}

bbox bbox::rescale(float x, float y) const
{
    bbox rc   = *this;
    rc.m_xmin = rc.m_xmin * x;
    rc.m_ymin = rc.m_ymin * y;
    rc.m_xmax = rc.m_xmax * x;
    rc.m_ymax = rc.m_ymax * y;
    return rc;
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
    // Return [0, 0, 0, 0, true] if there is no intersection.
    if (second_bbox.xmin() > xmax() || second_bbox.xmax() < xmin() || second_bbox.ymin() > ymax() ||
        second_bbox.ymax() < ymin())
    {
        return zerobox();
    }

    return bbox(std::max(xmin(), second_bbox.xmin()),
                std::max(ymin(), second_bbox.ymin()),
                std::min(xmax(), second_bbox.xmax()),
                std::min(ymax(), second_bbox.ymax()));
}

bool bbox::is_properly_normalized() const
{
    auto is_value_normalized = [](float x) {
        return almost_equal_or_greater(x, 0.0f) && almost_equal_or_less(x, 1.0);
    };
    return is_value_normalized(m_xmin) && is_value_normalized(m_xmax) &&
           is_value_normalized(m_ymin) && is_value_normalized(m_ymax);
}

void bbox::throw_if_improperly_normalized() const
{
    if (!is_properly_normalized())
    {
        std::stringstream ss;
        ss << "bounding box '" << (*this) << "' is not properly normalized";
        throw std::invalid_argument(ss.str());
    }
}
