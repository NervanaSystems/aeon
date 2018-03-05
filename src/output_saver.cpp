/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <fstream>
#include <vector>

#include "boost/filesystem/path.hpp"

#include "output_saver.hpp"

using std::string;
using std::vector;

void nervana::output_saver::save(const nervana::fixed_buffer_map* batch)
{
    if (batch == nullptr)
    {
        return;
    }
    const buffer_fixed_size_elements* image_batch = (*batch)["image"];
    if (image_batch == nullptr)
    {
        return;
    }

    for (int i = 0; i < image_batch->get_item_count(); i++)
    {
        cv::Mat image(image_batch->get_item_as_mat(i));
        save(image);
    }
}

std::string nervana::output_saver::get_debug_file_id()
{
    unsigned int number = m_index++;
    return std::to_string(number);
}

void nervana::output_saver::save(const cv::Mat& image)
{
    string filename = get_filename();
    filename += +".png";
    cv::imwrite(filename, image);
}

void nervana::output_saver::save(const cv::Mat&                          image,
                                 std::shared_ptr<augment::image::params> img_xform)
{
    string filename = get_filename(img_xform->debug_output_directory);
    cv::imwrite(filename + ".png", image);
    std::ofstream ofs(filename + ".txt", std::ofstream::out);
    ofs << *img_xform;
    ofs.close();
}

string nervana::output_saver::get_filename()
{
    return get_filename(m_output_dir);
}

string nervana::output_saver::get_filename(const string& directory)
{
    auto filename = boost::filesystem::path(directory) / get_debug_file_id();
    return filename.string();
}
