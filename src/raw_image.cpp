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

#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "raw_image.hpp"
#include "log.hpp"
#include "util.hpp"

using namespace std;
using namespace nervana;

raw_image::raw_image()
    : m_width{0}
    , m_height{0}
    , m_channels{0}
    , m_bitwidth{0}
    , m_is_float{false}
    , m_is_big_endian{false}
{
}

raw_image::raw_image(const string& filename)
{
    ifstream f(filename);
    if (f)
    {
        read(f);
    }
}

raw_image::raw_image(istream& in)
{
    read(in);
}

void raw_image::read(istream& f)
{
    string line;
    while (getline(f, line))
    {
        if (line.size() == 0)
        {
            break;
        }
        auto tokens = split(line, ':', true);
        if (tokens.size() == 2)
        {
            string tag   = tokens[0];
            string value = tokens[1];
            if (tag == "width")
            {
                m_width = stoi(value);
            }
            else if (tag == "height")
            {
                m_height = stoi(value);
            }
            else if (tag == "channels")
            {
                m_channels = stoi(value);
            }
            else if (tag == "bitwidth")
            {
                m_bitwidth = stoi(value);
            }
            else if (tag == "is_float")
            {
                m_is_float = to_bool(value);
            }
            else if (tag == "is_big_endian")
            {
                m_is_big_endian = to_bool(value);
            }
        }
    }

    m_data = shared_ptr<char>(new char[size()], std::default_delete<char[]>());
    f.read(&*m_data, size());
}

void raw_image::write(const string& filename)
{
    ofstream f(filename);
    if (f)
    {
        write(f);
    }
}

void raw_image::write(ostream& out)
{
    out << "width: " << m_width << "\n";
    out << "height: " << m_height << "\n";
    out << "channels: " << m_channels << "\n";
    out << "bitwidth: " << m_bitwidth << "\n";
    out << "is_float: " << m_is_float << "\n";
    out << "is_big_endian: " << m_is_big_endian << "\n";
    out << "\n";

    out.write(&*m_data, size());
}

raw_image raw_image::from_cvmat(cv::Mat& mat)
{
    raw_image rc;
    rc.m_channels = mat.channels();
    rc.m_width    = mat.cols;
    rc.m_height   = mat.rows;
    rc.m_bitwidth = mat.elemSize1() * 8;

    size_t size = rc.m_width * rc.m_height * rc.m_channels * (rc.m_bitwidth / 8);
    rc.m_data   = shared_ptr<char>(new char[size], std::default_delete<char[]>());
    memcpy(&*rc.m_data, mat.data, size);

    return rc;
}

cv::Mat raw_image::to_cvmat()
{
    int type = 0;

    if (m_is_float)
    {
        switch (m_bitwidth)
        {
        case 32: type = CV_MAKETYPE(CV_32F, m_channels); break;
        case 64: type = CV_MAKETYPE(CV_64F, m_channels); break;
        default: break;
        }
    }
    else
    {
        switch (m_bitwidth)
        {
        case 8: type  = CV_MAKETYPE(CV_8U, m_channels); break;
        case 16: type = CV_MAKETYPE(CV_16U, m_channels); break;
        default: break;
        }
    }

    cv::Mat rc{(int)m_height, (int)m_width, type};
    memcpy(rc.data, &*m_data, size());

    return rc;
}

size_t raw_image::size() const
{
    return m_width * m_height * m_channels * (m_bitwidth / 8);
}

bool raw_image::to_bool(const std::string& s) const
{
    bool   rc  = false;
    string str = to_lower(s);
    if (str == "true")
    {
        rc = true;
    }
    else if (str == "false")
    {
        rc = false;
    }
    else
    {
        rc = stoi(str);
    }
    return rc;
}
