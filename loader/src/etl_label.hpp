#pragma once
#include "etl_interface.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {

    namespace label {
        class config;
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }

    class label::config : public json_config_parser {
    public:
        bool binary = true;

        bool set_config(nlohmann::json js) override
        {
            parse_opt(binary, "binary", js);
            return true;
        }
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
        extractor(std::shared_ptr<const label::config> cfg = nullptr)
        {
            if (cfg != nullptr) {
                _binary = cfg->binary;
            }
        }

        ~extractor() {}

        std::shared_ptr<label::decoded> extract(const char* buf, int bufSize) override
        {
            int lbl;
            if (_binary) {
                if (bufSize != 4) {
                    throw std::runtime_error("Only 4 byte buffers can be loaded as int32");
                }
                lbl = unpack_le<int>(buf);
            } else {
                lbl = std::stoi(std::string(buf, (size_t) bufSize));
            }
            return std::make_shared<label::decoded>(lbl);
        }

    private:
        bool _binary = true;
    };


    class label::transformer : public interface::transformer<label::decoded, nervana::params> {
    public:
        transformer(std::shared_ptr<const label::config> = nullptr) {}

        ~transformer() {}

        std::shared_ptr<label::decoded> transform(
                            std::shared_ptr<nervana::params> txs,
                            std::shared_ptr<label::decoded> mp) override { return mp; }
    };

    class label::loader : public interface::loader<label::decoded> {
    public:
        loader(std::shared_ptr<const label::config> = nullptr) {}
        ~loader() {}

        void fill_info(count_size_type* cst) override
        {
            cst->count   = 1;
            cst->size    = 4;
            cst->type[0] = 'i';
        }

        void load(char* buf, std::shared_ptr<label::decoded> mp) override
        {
            int index = mp->get_index();
            memcpy(buf, &index, sizeof(int32_t));
        }

    };
}
