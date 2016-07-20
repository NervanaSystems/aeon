#include "buffer_pool.hpp"

buffer_pool::buffer_pool() {
    _exceptions.push_back(nullptr);
    _exceptions.push_back(nullptr);
}

void buffer_pool::writeException(std::exception_ptr exception_ptr) {
    _exceptions[_writePos] = exception_ptr;
}

void buffer_pool::clearException() {
    _exceptions[_writePos] = nullptr;
}

void buffer_pool::reraiseException() {
    if(auto e = _exceptions[_readPos]) {
        clearException();
        std::rethrow_exception(e);
    }
}
