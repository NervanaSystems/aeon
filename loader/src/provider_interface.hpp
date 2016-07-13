#pragma once
#include <memory>
#include "util.hpp"
#include "etl_interface.hpp"
#include "buffer_in.hpp"
#include "buffer_out.hpp"

namespace nervana {
    class provider_interface;
}

class nervana::provider_interface {
public:
    virtual void provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf) = 0;
};
