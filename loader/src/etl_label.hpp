#pragma once
#include <random>
#include "etl_interface.hpp"
#include "params.hpp"

using namespace std;

namespace nervana {

    namespace label {
        class params;
        class decoded;

        class extractor;
        class transformer;
        class loader;

        class settings;
    }
}

class nervana::label::settings : public nervana::settings {
public:
    int scale;
    int shift;
    settings() {}
};


class nervana::label::params : public nervana::json_parameter_collection {
public:
    int ex_offset = 0;

    std::uniform_int_distribution<int>    tx_scale{1, 1};
    std::uniform_int_distribution<int>    tx_shift{0, 0};

    float ld_offset = 0.0;
    bool ld_dofloat = false;

    params(std::string argString) {
        auto js = nlohmann::json::parse(argString);

        // Optionals with some standard defaults
        parse_opt(ex_offset,  "extract offset",  js);
        parse_dist<decltype(tx_scale)>(tx_scale, "transform scale dist params", js);
        parse_dist<decltype(tx_shift)>(tx_shift, "transform shift dist params", js);
        parse_opt(ld_offset,  "load offset",     js);
        parse_opt(ld_dofloat, "load do float",   js);
    }
};


class nervana::label::decoded : public nervana::decoded_media {
public:
    decoded(int index) :
        _index{index} {}
    virtual ~decoded() override {}

    inline MediaType get_type() override { return MediaType::TARGET; }
    inline int get_index() { return _index; }

private:
    decoded() = delete;
    int _index;
};


class nervana::label::extractor : public nervana::interface::extractor {
public:
    extractor(param_ptr pptr) {
        _ex_offset = static_pointer_cast<nervana::label::params>(pptr)->ex_offset;
    }

    ~extractor() {}

    media_ptr extract(char* buf, int bufSize) override {
        if (bufSize != 4) {
            throw runtime_error("Only 4 byte buffers can be loaded as int32");
        }
        return make_shared<nervana::label::decoded>(*reinterpret_cast<int *>(buf) + _ex_offset);
    }

private:
    int _ex_offset;
};


class nervana::label::transformer : public nervana::interface::transformer {
public:
    transformer(param_ptr pptr) {}

    ~transformer() {}

    media_ptr transform(settings_ptr tx, const media_ptr& mp) override {
        int old_index = static_pointer_cast<nervana::label::decoded>(mp)->get_index();
        shared_ptr<nervana::label::settings> txs = static_pointer_cast<nervana::label::settings>(tx);

        return make_shared<nervana::label::decoded>( old_index * txs->scale + txs->shift );
    }

    // Filling settings is done by the relevant params
    virtual void fill_settings(settings_ptr, const media_ptr&, std::default_random_engine &) override
    {}

};


class nervana::label::loader : public nervana::interface::loader {
public:
    loader(param_ptr pptr) {
        shared_ptr<nervana::label::params> lbl_ptr = static_pointer_cast<nervana::label::params>(pptr);
        _ld_offset = lbl_ptr->ld_offset;
        _ld_dofloat = lbl_ptr->ld_dofloat;
    }
    ~loader() {}

    void load(char* buf, int bufSize, const media_ptr& mp) override {
        int index = static_pointer_cast<nervana::label::decoded>(mp)->get_index();
        if (_ld_dofloat) {
            float ld_index = index + _ld_offset;
            memcpy(buf, &ld_index, bufSize);
        } else {
            memcpy(buf, &index, bufSize);
        }

    }

private:
    float _ld_offset;
    bool _ld_dofloat;
};
