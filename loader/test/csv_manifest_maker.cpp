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

#include <string.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "csv_manifest_maker.hpp"
#include "manifest_csv.hpp"
#include "file_util.hpp"
#include "gen_image.hpp"

using namespace std;
using namespace nervana;

manifest_maker::manifest_maker(size_t record_count, std::vector<size_t> sizes)
{
    manifest_name = tmp_manifest_file(record_count, sizes);
}

manifest_maker::manifest_maker(size_t record_count, int height, int width)
{
    manifest_name = image_manifest(record_count, height, width);
}

manifest_maker::manifest_maker()
{
}

manifest_maker::~manifest_maker()
{
    remove_files();
}

std::string manifest_maker::get_manifest_name()
{
    return manifest_name;
}

void manifest_maker::remove_files()
{
    for (auto it : tmp_filenames)
    {
        remove(it.c_str());
    }
}

string manifest_maker::tmp_filename(const string& extension)
{
    string tmpname = file_util::tmp_filename(extension);
    tmp_filenames.push_back(tmpname);
    return tmpname;
}

string manifest_maker::image_manifest(size_t record_count, int height, int width)
{
    string   tmpname = tmp_filename();
    ofstream f_manifest(tmpname);

    for (uint32_t i = 0; i < record_count; ++i)
    {
        cv::Mat mat = embedded_id_image::generate_image(height, width, i);
//        cv::Mat mat{height, width, CV_8UC3};
//        mat = cv::Scalar(0,0,0);
        string image_path = tmp_filename("_" + std::to_string(i) + ".png");
        string target_path = tmp_filename();
        f_manifest << image_path << manifest_csv::get_delimiter() << target_path << '\n';
        cv::imwrite(image_path, mat);
        {
            ofstream f(target_path);
            int value = 0;
            f.write((const char*)&value, sizeof(value));
        }
    }

    f_manifest.close();

    return tmpname;
}

string manifest_maker::tmp_manifest_file(size_t record_count, vector<size_t> sizes)
{
    string   tmpname = tmp_filename();
    ofstream f(tmpname);

    for (uint32_t i = 0; i < record_count; ++i)
    {
        // stick a unique uint32_t into each file
        for (uint32_t j = 0; j < sizes.size(); ++j)
        {
            if (j != 0)
            {
                f << manifest_csv::get_delimiter();
            }

            f << tmp_file_repeating(sizes[j], (i * sizes.size()) + j);
        }
        f << '\n';
    }

    f.close();

    return tmpname;
}

string manifest_maker::tmp_file_repeating(size_t size, uint32_t x)
{
    // create a temp file of `size` bytes filled with uint32_t x
    string   tmpname = tmp_filename();
    ofstream f(tmpname, ios::binary);

    uint32_t repeats = size / sizeof(x);
    for (uint32_t i = 0; i < repeats; ++i)
    {
        f.write(reinterpret_cast<const char*>(&x), sizeof(x));
    }

    f.close();

    return tmpname;
}

std::string manifest_maker::tmp_manifest_file_with_invalid_filename()
{
    string   tmpname = tmp_filename();
    ofstream f(tmpname);

    for (uint32_t i = 0; i < 10; ++i)
    {
        f << tmp_filename() + ".this_file_shouldnt_exist" << manifest_csv::get_delimiter();
        f << tmp_filename() + ".this_file_shouldnt_exist" << endl;
    }

    f.close();
    return tmpname;
}

std::string manifest_maker::tmp_manifest_file_with_ragged_fields()
{
    string   tmpname = tmp_filename();
    ofstream f(tmpname);

    for (uint32_t i = 0; i < 10; ++i)
    {
        for (uint32_t j = 0; j < i % 3 + 1; ++j)
        {
            if (j != 0)
            {
                f << manifest_csv::get_delimiter();
            }
            f << tmp_filename();
        }
        f << endl;
    }

    f.close();
    return tmpname;
}
