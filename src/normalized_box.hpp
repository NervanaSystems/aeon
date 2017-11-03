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

#pragma once

#include <opencv2/core/core.hpp>

#include "box.hpp"
#include "log.hpp"

class nervana::normalized_box::box final : public nervana::box
{
public:
    box()
        : nervana::box(0, 0, 0, 0)
    {
    }
    box(float xmin, float ymin, float xmax, float ymax)
        : nervana::box(xmin, ymin, xmax, ymax)
    {
        try
        {
            throw_if_improperly_normalized();
        }
        catch (std::exception&)
        {
            ERR << "Creating improperly normalized box in constructor of normalized_box.";
            throw;
        }
    }
    ~box() {}
    nervana::normalized_box::box& operator=(const box& b);

    float    width() const override { return m_xmax - m_xmin; }
    float    height() const override { return m_ymax - m_ymin; }
    cv::Rect rect() const override
    {
        throw std::runtime_error(
            "Called rect() on a normalized box. cv::Rect supports integers only.");
    }

    float jaccard_overlap(const nervana::normalized_box::box& second_box) const;
    float coverage(const nervana::normalized_box::box& second_box) const;
    nervana::normalized_box::box intersect(const nervana::normalized_box::box& second_box) const;

    nervana::boundingbox::box unnormalize(float width, float height)
    {
        return nervana::boundingbox::box(
            xmin() * width, ymin() * height, xmax() * width - 1, ymax() * height - 1);
    }
    bool is_properly_normalized() const;

protected:
    void throw_if_improperly_normalized() const;
};
