#pragma once

#include <cinttypes>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui/highgui_c.h>

namespace nervana {
    class MjpegInputStream;
    class MotionJpegCapture;
}

typedef std::deque< std::pair<uint64_t, uint32_t> > frame_list;
typedef frame_list::iterator frame_iterator;

class nervana::MjpegInputStream
{
public:
    MjpegInputStream();
    MjpegInputStream(const std::string& filename);
    ~MjpegInputStream();
    MjpegInputStream& read(char*, uint64_t);
    MjpegInputStream& seekg(uint64_t);
    uint64_t tellg();
    bool isOpened() const;
    bool open(const std::string& filename);
    void close();
    operator bool();

private:
    bool    m_is_valid;
    FILE*   m_f;
};

class nervana::MotionJpegCapture//: public IVideoCapture
{
public:
    virtual ~MotionJpegCapture();
    virtual double getProperty(int) const;
    virtual bool setProperty(int, double);
    virtual bool grabFrame();
    virtual bool retrieveFrame(int, cv::OutputArray);
    virtual bool isOpened() const;
    virtual int getCaptureDomain() { return CV_CAP_ANY; } // Return the type of the capture object: CAP_VFW, etc...
    MotionJpegCapture(const std::string&);

    bool open(const std::string&);
    void close();
protected:

    bool parseRiff(MjpegInputStream& in_str);

    inline uint64_t getFramePos() const;
    std::vector<char> readFrame(frame_iterator it);

    MjpegInputStream m_file_stream;
    bool             m_is_first_frame;
    frame_list       m_mjpeg_frames;

    frame_iterator   m_frame_iterator;
    cv::Mat          m_current_frame;

    //frame width/height and fps could be different for
    //each frame/stream. At the moment we suppose that they
    //stays the same within single avi file.
    uint32_t         m_frame_width;
    uint32_t         m_frame_height;
    double           m_fps;
};
