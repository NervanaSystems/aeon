#pragma once

#include <sstream>
#include <stdexcept>

namespace nervana {

class conststring
{
public:
    template<size_t SIZE>
    constexpr conststring(const char(&p)[SIZE]) :
        _string(p),
        _size(SIZE)
    {
    }

    constexpr char operator[](size_t i) const {
        return i < _size ? _string[i] : throw std::out_of_range("");
    }

    constexpr const char* get_ptr(size_t offset) const {
        return &_string[ offset ];
    }

    constexpr size_t size() const { return _size; }

private:
    const char* _string;
    size_t      _size;
};

constexpr const char* find_last(conststring s, size_t offset, char ch) {
    return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset+1) : find_last(s, offset-1, ch));
}

constexpr const char* find_last(conststring s, char ch) {
    return find_last(s, s.size()-1, ch);
}

constexpr const char* get_file_name(conststring s) {
    return find_last(s, '/');
}


enum class LOG_TYPE {
    _LOG_TYPE_ERROR,
    _LOG_TYPE_WARNING,
    _LOG_TYPE_INFO,
};

class log_helper {
public:
    log_helper(LOG_TYPE, const char* file, int line, const char* func);
    ~log_helper();

    std::ostream& stream() { return _stream; }

private:
    std::stringstream _stream;
};

class logger {
    friend class log_helper;
public:
    static void set_log_path(const std::string& path);
private:
    static void log_item(const std::string& s);
    static std::string log_path;
};

#define ERR  nervana::log_helper(nervana::LOG_TYPE::_LOG_TYPE_ERROR,   get_file_name(__FILE__), __LINE__, __PRETTY_FUNCTION__).stream()
#define WARN nervana::log_helper(nervana::LOG_TYPE::_LOG_TYPE_WARNING, get_file_name(__FILE__), __LINE__, __PRETTY_FUNCTION__).stream()
#define INFO nervana::log_helper(nervana::LOG_TYPE::_LOG_TYPE_INFO,    get_file_name(__FILE__), __LINE__, __PRETTY_FUNCTION__).stream()

#define ERR_OBJ(obj)  nervana::log_helper obj(nervana::LOG_TYPE::_LOG_TYPE_ERROR,   get_file_name(__FILE__), __LINE__, __PRETTY_FUNCTION__)
#define WARN_OBJ(obj) nervana::log_helper obj(nervana::LOG_TYPE::_LOG_TYPE_WARNING, get_file_name(__FILE__), __LINE__, __PRETTY_FUNCTION__)
#define INFO_OBJ(obj) nervana::log_helper obj(nervana::LOG_TYPE::_LOG_TYPE_INFO,    get_file_name(__FILE__), __LINE__, __PRETTY_FUNCTION__)
}
