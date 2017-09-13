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

using namespace std;
using namespace nervana;
using namespace nlohmann; // json stuff

using bbox = boundingbox::box;

ostream& operator<<(ostream& out, const bbox& b)
{
    out << (box)b << " label=" << b.label << " difficult=" << b.difficult
        << " truncated=" << b.truncated;
    return out;
}

ostream& operator<<(ostream& out, const vector<bbox>& boxes)
{
    out << "[";
    for (const bbox& box : boxes)
    {
        out << " {" << box << "}";
    }
    out << " ]";
    return out;
}

bool bbox::operator==(const bbox& b) const
{
    bool base_box_equal = ((nervana::box)(*this)) == ((nervana::box)b);
    if (!base_box_equal)
        return false;
    return difficult == b.difficult && truncated == b.truncated && label == b.label;
}

bool bbox::operator!=(const bbox& b) const
{
    return !((*this) == b);
}

bbox bbox::operator+(const cv::Size& s) const
{
    bbox rc = *this;
    rc.m_xmin += s.width;
    rc.m_ymin += s.height;
    rc.m_xmax += s.width;
    rc.m_ymax += s.height;
    rc.m_normalized = false;

    return rc;
}

void bbox::normalize(float width, float height)
{
    if (m_normalized)
        throw runtime_error("Cannot normalize box which is already normalized.");

    m_xmin /= width;
    m_xmax = (m_xmax + 1) / width;
    m_ymin /= height;
    m_ymax       = (m_ymax + 1) / height;
    m_normalized = true;
    try
    {
        throw_if_improperly_normalized();
    }
    catch (exception&)
    {
        ERR << "Error when normalizing boundingbox: " << *this << ". Range had width: " << width
            << " height: " << height;
        throw;
    }
}

bbox bbox::unnormalize(float width, float height) const
{
    if (!m_normalized)
        throw runtime_error("Cannot unnormalize box which is not normalized.");

    bbox rc         = *this;
    rc.m_xmin       = rc.m_xmin * width;
    rc.m_ymin       = rc.m_ymin * height;
    rc.m_xmax       = rc.m_xmax * width - 1;
    rc.m_ymax       = rc.m_ymax * height - 1;
    rc.m_normalized = false;
    return rc;
}

bbox bbox::rescale(float x, float y) const
{
    bbox rc = *this;
    if (m_normalized)
    {
        rc.m_xmin = rc.m_xmin * x;
        rc.m_ymin = rc.m_ymin * y;
        rc.m_xmax = rc.m_xmax * x;
        rc.m_ymax = rc.m_ymax * y;
    }
    else
    {
        rc.m_xmin = rc.m_xmin * x;
        rc.m_ymin = rc.m_ymin * y;
        rc.m_xmax = (rc.m_xmax + 1) * x - 1;
        rc.m_ymax = (rc.m_ymax + 1) * y - 1;
    }
    return rc;
}

void bbox::expand_bbox(const cv::Size2i& expand_offset,
                       const cv::Size2i& expand_size,
                       const float       expand_ratio)
{
    if (m_xmax + expand_offset.width > expand_size.width ||
        m_ymax + expand_offset.height > expand_size.height)
    {
        stringstream ss;
        ss << "Invalid parameters to expand boundingbox: " << *this << endl
           << "Expand_offset: " << expand_offset << " expand_size: " << expand_size << endl
           << m_xmax << " + " << expand_offset.width << " > " << expand_size.width << " || "
           << m_ymax << " + " << expand_offset.height << " > " << expand_size.height;

        throw std::invalid_argument(ss.str());
    }

    *this = *this + expand_offset;
}

float bbox::size() const
{
    if (xmax() < xmin() || ymax() < ymin())
    {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0.0f;
    }
    else
    {
        return width() * height();
    }
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
    if (second_bbox.normalized() != m_normalized)
        throw std::invalid_argument("Intersection of normalized and unnormalized boundingboxes.");

    // Return [0, 0, 0, 0, true] if there is no intersection.
    if (second_bbox.xmin() > xmax() || second_bbox.xmax() < xmin() || second_bbox.ymin() > ymax() ||
        second_bbox.ymax() < ymin())
    {
        return box::zerobox;
    }

    bbox intersect_bbox(m_normalized);
    intersect_bbox.set_xmin(std::max(xmin(), second_bbox.xmin()));
    intersect_bbox.set_ymin(std::max(ymin(), second_bbox.ymin()));
    intersect_bbox.set_xmax(std::min(xmax(), second_bbox.xmax()));
    intersect_bbox.set_ymax(std::min(ymax(), second_bbox.ymax()));
    return intersect_bbox;
}

void bbox::normalize_bboxes(vector<bbox>& bboxes, int width, int height)
{
    for (bbox& box : bboxes)
    {
        box.normalize(width, height);
    }
}
