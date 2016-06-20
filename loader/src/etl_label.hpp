#pragma once
#include "etl_interface.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {

    namespace label {
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }

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
        extractor(std::shared_ptr<const json_config_parser> = nullptr) {}

        ~extractor() {}

        std::shared_ptr<label::decoded> extract(const char* buf, int bufSize) override
        {
            if (bufSize != 4) {
                throw std::runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return std::make_shared<label::decoded>(unpack_le<int>(buf));
        }
    };

    class label::transformer : public interface::transformer<label::decoded, nervana::params> {
    public:
        transformer(std::shared_ptr<const json_config_parser> = nullptr) {}

        ~transformer() {}

        std::shared_ptr<label::decoded> transform(
                            std::shared_ptr<nervana::params> txs,
                            std::shared_ptr<label::decoded> mp) override { return mp; }
    };

    class label::loader : public interface::loader<label::decoded> {
    public:
        loader(std::shared_ptr<const json_config_parser> = nullptr) {}

        ~loader() {}

        int get_load_size() override { return _loadsz; }

        void load(char* buf, std::shared_ptr<label::decoded> mp) override
        {
            int index = mp->get_index();
            memcpy(buf, &index, _loadsz);
        }
    private:
        int _loadsz = 4;
    };
}
