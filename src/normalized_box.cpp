/*
 Copyright 2017 Intel(R) Nervana(TM)
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
using nbox = nervana::normalized_box::box;
using nervana::almost_equal;

nbox& nbox::operator=(const nbox& b)
{
    if (&b != this)
    {
        m_xmin = b.m_xmin;
        m_ymin = b.m_ymin;
        m_xmax = b.m_xmax;
        m_ymax = b.m_ymax;
        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Error when assigning normalized nbox: " << b;
            throw;
        }
    }
    return *this;
}

float nbox::jaccard_overlap(const nbox& second_nbox) const
{
    nbox  intersect_nbox = intersect(second_nbox);
    float intersect_size = intersect_nbox.size();
    float nbox1_size     = size();
    float nbox2_size     = second_nbox.size();

    if (intersect_size == 0.f)
        return 0.f;
    return intersect_size / (nbox1_size + nbox2_size - intersect_size);
}

float nbox::coverage(const nbox& second_nbox) const
{
    nbox  intersect_nbox = intersect(second_nbox);
    float intersect_size = intersect_nbox.size();
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

nbox nbox::intersect(const nbox& second_nbox) const
{
    // Return [0, 0, 0, 0, true] if there is no intersection.
    if (second_nbox.xmin() > xmax() || second_nbox.xmax() < xmin() || second_nbox.ymin() > ymax() ||
        second_nbox.ymax() < ymin())
    {
        return box();
    }

    return nbox(std::max(xmin(), second_nbox.xmin()),
                std::max(ymin(), second_nbox.ymin()),
                std::min(xmax(), second_nbox.xmax()),
                std::min(ymax(), second_nbox.ymax()));
}

bool nbox::is_properly_normalized() const
{
    auto is_value_normalized = [](float x) {
        return almost_equal_or_greater(x, 0.0f) && almost_equal_or_less(x, 1.0);
    };
    return is_value_normalized(m_xmin) && is_value_normalized(m_xmax) &&
           is_value_normalized(m_ymin) && is_value_normalized(m_ymax);
}

void nbox::throw_if_improperly_normalized() const
{
    if (!is_properly_normalized())
    {
        std::stringstream ss;
        ss << "bounding box '" << (*this) << "' is not properly normalized";
        throw std::invalid_argument(ss.str());
    }
}
