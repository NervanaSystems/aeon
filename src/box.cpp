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

#include "box.hpp"
#include "log.hpp"
#include "boundingbox.hpp"
#include "normalized_box.hpp"

using namespace std;

ostream& operator<<(ostream& out, const nervana::box& b)
{
    out << "[" << b.width() << " x " << b.height() << " from (" << b.xmin() << ", " << b.ymin()
        << ")]";
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

nervana::box& nervana::box::operator=(const nervana::box& b)
{
    if (this != &b)
    {
        m_xmin = b.m_xmin;
        m_ymin = b.m_ymin;
        m_xmax = b.m_xmax;
        m_ymax = b.m_ymax;
    }
    return *this;
}

namespace nervana
{
    std::vector<normalized_box::box>
        normalize_bboxes(const std::vector<boundingbox::box>& bboxes, int width, int height)
    {
        std::vector<normalized_box::box> rc(bboxes.size());
        for (int i = 0; i < rc.size(); i++)
        {
            rc[i] = bboxes[i].normalize(width, height);
        }
        return rc;
    }
}
