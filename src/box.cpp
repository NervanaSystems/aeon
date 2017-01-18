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

using namespace std;

ostream& operator<<(ostream& out, const nervana::box& b)
{
    out << "[" << b.xmax - b.xmin << " x " << b.ymax - b.ymin << " from (" << b.xmin << ", "
        << b.ymin << ")]";
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
