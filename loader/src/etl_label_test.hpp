#pragma once
#include <random>
#include "etl_interface.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {

    namespace label_test {
        class config;
        class params;
        class decoded;

        class param_factory; // goes from config -> params

        class extractor;
        class transformer;
        class loader;
    }

    class label_test::config : public json_config_parser {
    public:
        int ex_offset = 0;

        std::uniform_int_distribution<int>    tx_scale{1, 1};
        std::uniform_int_distribution<int>    tx_shift{0, 0};

        float ld_offset = 0.0;
        bool ld_dofloat = false;

        bool set_config(nlohmann::json js) override
        {
            // Optionals with some standard defaults
            parse_opt(ex_offset,  "extract offset",  js);
            parse_dist(tx_scale,  "dist_params/transform scale", js);
            parse_dist(tx_shift,  "dist_params/transform shift", js);
            parse_opt(ld_offset,  "load offset",     js);
            parse_opt(ld_dofloat, "load do float",   js);
            return validate();
        }

    private:
        bool validate() {
            return true;
        }
    };


    class label_test::params : public nervana::params {
    public:
        int scale;
        int shift;
        params() {}
    };


    class label_test::param_factory : public interface::param_factory<label_test::decoded, label_test::params> {
    public:
        param_factory(std::shared_ptr<label_test::config> cfg,
                      std::default_random_engine &dre) : _cfg{cfg}, _dre{dre} {}

        ~param_factory() {}

        std::shared_ptr<label_test::params>
        make_params(std::shared_ptr<const decoded> lbl)
        {
            auto sptr = std::make_shared<label_test::params>();
            sptr->scale = _cfg->tx_scale(_dre);
            sptr->shift = _cfg->tx_shift(_dre);
            return sptr;
        }

    private:

        std::shared_ptr<label_test::config> _cfg;
        std::default_random_engine &_dre;
    };


    class label_test::decoded : public decoded_media {
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


    class label_test::extractor : public interface::extractor<label_test::decoded> {
    public:
        extractor(std::shared_ptr<const label_test::config> cfg) {
            _ex_offset = cfg->ex_offset;
        }

        ~extractor() {}

        std::shared_ptr<label_test::decoded> extract(const char* buf, int bufSize) override
        {
            if (bufSize != 4) {
                throw std::runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return std::make_shared<label_test::decoded>(unpack_le<int>(buf)+_ex_offset);
        }

    private:
        int _ex_offset;
    };


    class label_test::transformer : public interface::transformer<label_test::decoded, label_test::params> {
    public:
        transformer(std::shared_ptr<const label_test::config>) {}

        ~transformer() {}

        std::shared_ptr<label_test::decoded> transform(
                            std::shared_ptr<label_test::params> txs,
                            std::shared_ptr<label_test::decoded> mp) override
        {
            int old_index = std::static_pointer_cast<label_test::decoded>(mp)->get_index();
            return std::make_shared<label_test::decoded>( old_index * txs->scale + txs->shift );
        }

    };


    class label_test::loader : public interface::loader<label_test::decoded> {
    public:
        loader(std::shared_ptr<const label_test::config> cfg)
        {
            _ld_offset = cfg->ld_offset;
            _ld_dofloat = cfg->ld_dofloat;
        }

        ~loader() {}

        void load(char* buf, int bufSize, std::shared_ptr<label_test::decoded> mp) override
        {
            int index = mp->get_index();
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
