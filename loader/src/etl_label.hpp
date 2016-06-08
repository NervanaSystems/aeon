#pragma once
#include <random>
#include "etl_interface.hpp"
#include "params.hpp"

using namespace std;

namespace nervana {

    namespace label {
        class config;
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


class nervana::label::config : public nervana::json_config_parser {
public:
    int ex_offset = 0;

    std::uniform_int_distribution<int>    tx_scale{1, 1};
    std::uniform_int_distribution<int>    tx_shift{0, 0};

    float ld_offset = 0.0;
    bool ld_dofloat = false;

    config(std::string argString) {
        auto js = nlohmann::json::parse(argString);

        // Optionals with some standard defaults
        parse_opt(ex_offset,  "extract offset",  js);
        parse_dist(tx_scale,  "dist_params/transform scale", js);
        parse_dist(tx_shift,  "dist_params/transform shift", js);
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


class nervana::label::extractor : public nervana::interface::extractor<nervana::label::decoded> {
public:
    extractor(shared_ptr<const nervana::label::config> cfg) {
        _ex_offset = cfg->ex_offset;
    }

    ~extractor() {}

    std::shared_ptr<nervana::label::decoded> extract(char* buf, int bufSize) override {
        if (bufSize != 4) {
            throw runtime_error("Only 4 byte buffers can be loaded as int32");
        }
        return make_shared<nervana::label::decoded>(*reinterpret_cast<int *>(buf) + _ex_offset);
    }

private:
    int _ex_offset;
};


class nervana::label::transformer : public nervana::interface::transformer<nervana::label::decoded> {
public:
    transformer(shared_ptr<const nervana::label::config>) {}

    ~transformer() {}

    std::shared_ptr<nervana::label::decoded> transform(settings_ptr tx, std::shared_ptr<nervana::label::decoded> mp) override {
        int old_index = static_pointer_cast<nervana::label::decoded>(mp)->get_index();
        auto txs = static_pointer_cast<nervana::label::settings>(tx);

        return make_shared<nervana::label::decoded>( old_index * txs->scale + txs->shift );
    }

    // Filling settings is done by the relevant params
    virtual void fill_settings(settings_ptr, std::shared_ptr<nervana::label::decoded>, std::default_random_engine &) override
    {}

};


class nervana::label::loader : public nervana::interface::loader<nervana::label::decoded> {
public:
    loader(shared_ptr<const nervana::label::config> cfg) {
        _ld_offset = cfg->ld_offset;
        _ld_dofloat = cfg->ld_dofloat;
    }
    ~loader() {}

<<<<<<< a6f241d1202fefa464640ff20763bc94544f5655
    void load(char* buf, int bufSize, std::shared_ptr<nervana::label::decoded> mp) override {
=======
    void load(char* buf, int bufSize, const media_ptr& mp) override {
>>>>>>> Checkpoint - class-typed shared_ptrs
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
