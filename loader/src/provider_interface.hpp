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
    virtual void post_process(buffer_out_array& out_buf) {}

    virtual const std::vector<nervana::shape_type>& get_oshapes() { return oshapes; }
    uint32_t num_inputs;
protected:
    std::vector<nervana::shape_type> oshapes;
};
