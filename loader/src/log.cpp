#include <chrono>
#include <iomanip>
#include <iostream>
#include <ctime>

#include "log.hpp"

using namespace std;

std::string nervana::logger::log_path;

void nervana::logger::set_log_path(const std::string& path) {
    log_path = path;
}

void nervana::logger::log_item(const std::string& s) {
    cout << s << "\n";
}

nervana::log_helper::log_helper(LOG_TYPE type, const char* file, int line, const char* func) {
    switch(type){
    case LOG_TYPE::_LOG_TYPE_ERROR:
        _stream << "[ERR ] ";
        break;
    case LOG_TYPE::_LOG_TYPE_WARNING:
        _stream << "[WARN] ";
        break;
    case LOG_TYPE::_LOG_TYPE_INFO:
        _stream << "[INFO] ";
        break;
    }

    std::time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tm = std::gmtime(&tt);
    char buffer[256];
    strftime(buffer,sizeof(buffer), "%Y-%m-%d %H:%M:%S UTC", tm);
    _stream << buffer << " ";

    _stream << file;
    _stream << " " << line;
//    _stream << " " << func;
    _stream << "\t";
}

nervana::log_helper::~log_helper() {
    logger::log_item(_stream.str());
}
