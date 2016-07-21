#pragma once

#include <cinttypes>
#include <string>
#include <sstream>
#include <vector>
#include <deque>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <opencv2/highgui/highgui_c.h>

namespace nervana {
    class MjpegInputStream;
    class MjpegFileInputStream;
    class MjpegMemoryInputStream;
    class MotionJpegCapture;
}

typedef std::deque< std::pair<uint64_t, uint32_t> > frame_list;
typedef frame_list::iterator frame_iterator;

class nervana::MjpegInputStream
{
public:
    MjpegInputStream(){};
    virtual ~MjpegInputStream(){};
    virtual MjpegInputStream& read(char*, uint64_t) = 0;
    virtual MjpegInputStream& seekg(uint64_t) = 0;
    virtual uint64_t tellg() = 0;
    virtual bool isOpened() const = 0;
    virtual bool open(const std::string& filename) = 0;
    virtual void close() = 0;
    virtual operator bool() = 0;
};

class nervana::MjpegFileInputStream : public nervana::MjpegInputStream
{
public:
    MjpegFileInputStream();
    MjpegFileInputStream(const std::string& filename);
    ~MjpegFileInputStream();
    MjpegInputStream& read(char*, uint64_t) override;
    MjpegInputStream& seekg(uint64_t) override;
    uint64_t tellg() override;
    bool isOpened() const override;
    bool open(const std::string& filename) override;
    void close() override;
    operator bool() override;

private:
    bool            m_is_valid;
    std::ifstream   m_f;
};

class nervana::MjpegMemoryInputStream : public nervana::MjpegInputStream
{
public:
    MjpegMemoryInputStream();
    MjpegMemoryInputStream(char* data, size_t size);
    ~MjpegMemoryInputStream();
    MjpegInputStream& read(char*, uint64_t) override;
    MjpegInputStream& seekg(uint64_t) override;
    uint64_t tellg() override;
    bool isOpened() const override;
    bool open(const std::string& filename) override;
    void close() override;
    operator bool() override;

private:
    bool            m_is_valid;
    std::istream    m_f;
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

    MjpegFileInputStream m_file_stream;
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
