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

#include "box.hpp"
#include "util.hpp"
#include "log.hpp"

using namespace std;

using nervana::almost_equal;

ostream& operator<<(ostream& out, const nervana::box& b)
{
    out << "[" << b.width() << " x " << b.height() << " from (" << b.xmin() << ", " << b.ymin()
        << ")] normalized=" << b.normalized();
    return out;
}

ostream& operator<<(ostream& out, const vector<nervana::box>& list)
{
    for (const nervana::box& b : list)
    {
        out << b << "\n";
    }
    return out;
}

const nervana::box& nervana::box::zerobox()
{
    const static box b(0, 0, 0, 0, true);
    return b;
}

nervana::box& nervana::box::operator=(const box& b)
{
    m_xmin       = b.m_xmin;
    m_ymin       = b.m_ymin;
    m_xmax       = b.m_xmax;
    m_ymax       = b.m_ymax;
    m_normalized = b.m_normalized;

    try
    {
        if (m_normalized)
            throw_if_improperly_normalized();
    }
    catch (std::exception&)
    {
        ERR << "Error when assigning box: " << b;
        throw;
    }
    return *this;
}

nervana::box::box(const box& b)
    : m_xmin(b.m_xmin)
    , m_ymin(b.m_ymin)
    , m_xmax(b.m_xmax)
    , m_ymax(b.m_ymax)
    , m_normalized(b.m_normalized)
{
    if (m_normalized)
        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Invalid arguments to box copy constructor: " << b;
            throw;
        }
}

nervana::box::box(const box&& b)
    : m_xmin(b.m_xmin)
    , m_ymin(b.m_ymin)
    , m_xmax(b.m_xmax)
    , m_ymax(b.m_ymax)
    , m_normalized(b.m_normalized)
{
    if (m_normalized)
        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Invalid arguments to box move constructor: " << b;
            throw;
        }
}

bool nervana::box::operator==(const box& b) const
{
    return m_normalized == b.m_normalized && almost_equal(m_xmin, b.m_xmin) &&
           almost_equal(m_ymin, b.m_ymin) && almost_equal(m_xmax, b.m_xmax) &&
           almost_equal(m_ymax, b.m_ymax);
}

float nervana::box::xcenter() const
{
    return m_xmin + (m_xmax - m_xmin) / 2.;
}

float nervana::box::ycenter() const
{
    return m_ymin + (m_ymax - m_ymin) / 2.;
}

float nervana::box::width() const
{
    if (m_normalized)
    {
        return m_xmax - m_xmin;
    }
    else
    {
        return m_xmax - m_xmin + 1.;
    }
}

float nervana::box::height() const
{
    if (m_normalized)
    {
        return m_ymax - m_ymin;
    }
    else
    {
        return m_ymax - m_ymin + 1.;
    }
}

bool nervana::box::is_properly_normalized() const
{
    auto is_value_normalized = [](float x) {
        return almost_equal_or_greater(x, 0.0f) && almost_equal_or_less(x, 1.0);
    };
    if (is_value_normalized(m_xmin) && is_value_normalized(m_xmax) && is_value_normalized(m_ymin) &&
        is_value_normalized(m_ymax))
    {
        return true;
    }

    return false;
}

void nervana::box::throw_if_improperly_normalized() const
{
    if (!is_properly_normalized())
    {
        stringstream ss;
        ss << "bounding box '" << (*this) << "' is not properly normalized";
        throw invalid_argument(ss.str());
    }
}
