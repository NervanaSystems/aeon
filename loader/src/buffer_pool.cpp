#include "buffer_pool.hpp"

using namespace nervana;

buffer_pool::buffer_pool() {
    _exceptions.push_back(nullptr);
    _exceptions.push_back(nullptr);
}

void buffer_pool::write_exception(std::exception_ptr exception_ptr) {
    _exceptions[_writePos] = exception_ptr;
}

void buffer_pool::clear_exception() {
    _exceptions[_writePos] = nullptr;
}

void buffer_pool::reraise_exception() {
    if(auto e = _exceptions[_readPos]) {
        clear_exception();
        std::rethrow_exception(e);
    }
}
