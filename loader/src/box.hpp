#pragma once

#include <ostream>
#include <opencv2/core/core.hpp>

namespace nervana {
    class box;
}

class nervana::box
{
public:
    float xmin;
    float ymin;
    float xmax;
    float ymax;

    box(){}

    box(float _xmin, float _ymin, float _xmax, float _ymax) :
        xmin{_xmin}, ymin{_ymin}, xmax{_xmax}, ymax{_ymax}
    {}

    box operator+(const box& b) const {
        return box(xmin+b.xmin, ymin+b.ymin, xmax+b.xmax, ymax+b.ymax);
    }

    bool operator==(const box& b) const {
        return xmin==b.xmin && ymin==b.ymin && xmax==b.xmax && ymax==b.ymax;
    }

    cv::Rect rect() const {
        return cv::Rect(xmin,ymin,xmax-xmin,ymax-ymin);
    }

    float xcenter() const { return xmin + width() / 2; }
    float ycenter() const { return ymin + height() / 2; }
    float width() const { return xmax - xmin + 1; }
    float height() const { return ymax - ymin + 1; }
};

std::ostream& operator<<(std::ostream& out, const nervana::box& b);
std::ostream& operator<<(std::ostream& out, const std::vector<nervana::box>& list);
