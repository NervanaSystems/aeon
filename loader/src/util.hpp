#pragma once

#include <iostream>
#include <sstream>
#include <cassert>
#include <vector>

namespace nervana {

    // Unpack char array in little endian order
    template<typename T> T unpack_le(const char* data, int offset=0, int count=sizeof(T)) {
        T rc = 0;
        for(int i=0; i<count; i++) {
            rc += (unsigned char)data[offset+i] << (8*i);
        }
        return rc;
    }

    // Unpack char array in big endian order
    template<typename T> T unpack_be(const char* data, int offset=0, int count=sizeof(T)) {
        T rc = 0;
        for(int i=0; i<count; i++) {
            rc += (unsigned char)data[offset+i] << (8*(count-i-1));
        }
        return rc;
    }

    // Pack char array in little endian order
    template<typename T> void pack_le(char* data, T value, int offset=0, int count=sizeof(T)) {
        for(int i=0; i<count; i++) {
            data[offset+i] = (char)(value >> (8*i));
        }
    }

    // Pack char array in big endian order
    template<typename T> void pack_be(char* data, T value, int offset=0, int count=sizeof(T)) {
        for(int i=0; i<count; i++) {
            data[offset+i] = (char)(value >> (8*(count-i-1)));
        }
    }

    template<typename T> std::string join(const T& v, const std::string& sep) {
        std::ostringstream ss;
        for(const auto& x : v) {
            if(&x != &v[0]) ss << sep;
            ss << x;
        }
        return ss.str();
    }

    void dump( const void*, size_t );

    std::string tolower(const std::string& s);
    std::vector<std::string> split(const std::string& s, const std::string& delimiter);

    int LevenshteinDistance(const std::string& s1, const std::string& s2);

    template<typename CharT, typename TraitsT = std::char_traits<CharT> >
    class memstream : public std::basic_streambuf<CharT, TraitsT> {
    public:
        memstream(CharT* data, size_t size) {
            this->setg(data, data, data+size);
        }

        std::ios::pos_type seekoff(std::ios::off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
            switch(dir) {
            case std::ios_base::beg:
                this->setg(this->eback(), this->eback()+off, this->egptr());
                break;
            case std::ios_base::cur:
                this->setg(this->eback(), this->gptr()+off, this->egptr());
                break;
            case std::ios_base::end:
                this->setg(this->eback(), this->egptr()-off, this->egptr());
                break;
            default:
                break;
            }
            return this->gptr() - this->eback();
        }
        std::ios::pos_type seekpos(std::ios::pos_type pos, std::ios_base::openmode which) override {
            this->setg(this->eback(), this->eback()+pos, this->egptr());
            return this->gptr() - this->eback();
        }
    };

    class memory_stream : public std::istream {
    public:
        memory_stream(char* data, size_t size) :
            std::istream{&wrapper},
            wrapper{data, size}
        {
        }
    private:
        memstream<char> wrapper;
    };
}
