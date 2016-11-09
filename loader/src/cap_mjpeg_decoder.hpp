#pragma once

#include <cinttypes>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <fstream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui/highgui_c.h>

#include "util.hpp"
#include "avi.hpp"

namespace nervana
{
    class MotionJpegCapture;
}

class nervana::MotionJpegCapture
{
public:
    MotionJpegCapture(const std::string&);
    MotionJpegCapture(char* buffer, size_t size);
    virtual ~MotionJpegCapture();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual bool retrieveFrame(int, cv::Mat&);
    virtual bool isOpened() const;

    // Return the type of the capture object: CAP_VFW, etc...
    virtual int getCaptureDomain() { return CV_CAP_ANY; }

    bool open();
    void close();
protected:

    bool parseRiff(std::istream& in_str);

    inline uint64_t getFramePos() const;
    std::vector<char> readFrame(frame_iterator it);

    std::shared_ptr<std::istream>       m_file_stream;
    bool                                m_is_first_frame;
    frame_list                          m_mjpeg_frames;

    frame_iterator                      m_frame_iterator;
    cv::Mat                             m_current_frame;

    //frame width/height and fps could be different for
    //each frame/stream. At the moment we suppose that they
    //stays the same within single avi file.
    uint32_t         m_frame_width;
    uint32_t         m_frame_height;
    double           m_fps;
};
