#pragma once
#include <random>
#include "etl_interface.hpp"
#include "params.hpp"
#include "util.hpp"

using namespace std;

namespace nervana {

    namespace label {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;
    }

    class label::config : public json_config_parser {
    public:
        int ex_offset = 0;

        std::uniform_int_distribution<int>    tx_scale{1, 1};
        std::uniform_int_distribution<int>    tx_shift{0, 0};

        float ld_offset = 0.0;
        bool ld_dofloat = false;

        config(std::string argString)
        {
            auto js = nlohmann::json::parse(argString);

            // Optionals with some standard defaults
            parse_opt(ex_offset,  "extract offset",  js);
            parse_dist(tx_scale,  "dist_params/transform scale", js);
            parse_dist(tx_shift,  "dist_params/transform shift", js);
            parse_opt(ld_offset,  "load offset",     js);
            parse_opt(ld_dofloat, "load do float",   js);
        }
    };


    class label::params : public nervana::params {
    public:
        int scale;
        int shift;
        params() {}
    };


    class label::param_factory {
    public:
        param_factory(std::shared_ptr<label::config> cfg) {
            _cfg = cfg;
        }
        ~param_factory() {}

        std::shared_ptr<label::params>
        make_params(std::shared_ptr<const decoded> lbl, std::default_random_engine& dre)
        {
            auto sptr = make_shared<label::params>();
            sptr->scale = _cfg->tx_scale(dre);
            sptr->shift = _cfg->tx_shift(dre);
            return sptr;
        }

    private:

        std::shared_ptr<label::config> _cfg;
    };


    class label::decoded : public decoded_media {
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


    class label::extractor : public interface::extractor<label::decoded> {
    public:
        extractor(shared_ptr<const label::config> cfg) {
            _ex_offset = cfg->ex_offset;
        }

        ~extractor() {}

        std::shared_ptr<label::decoded> extract(const char* buf, int bufSize) override
        {
            if (bufSize != 4) {
                throw runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return make_shared<label::decoded>(unpack_le<int>(buf)+_ex_offset);
        }

    private:
        int _ex_offset;
    };


    class label::transformer : public interface::transformer<label::decoded, label::params> {
    public:
        transformer(shared_ptr<const label::config>) {}

        ~transformer() {}

        std::shared_ptr<label::decoded> transform(
                            std::shared_ptr<label::params> txs,
                            std::shared_ptr<label::decoded> mp) override
        {
            int old_index = static_pointer_cast<label::decoded>(mp)->get_index();
            return make_shared<label::decoded>( old_index * txs->scale + txs->shift );
        }

    };


    class label::loader : public interface::loader<label::decoded> {
    public:
        loader(shared_ptr<const label::config> cfg)
        {
            _ld_offset = cfg->ld_offset;
            _ld_dofloat = cfg->ld_dofloat;
        }

        ~loader() {}

        void load(char* buf, int bufSize, std::shared_ptr<label::decoded> mp) override
        {
            int index = static_pointer_cast<label::decoded>(mp)->get_index();
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
}
