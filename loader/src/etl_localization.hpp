#pragma once
#include "etl_interface.hpp"
#include "etl_bbox.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {

    namespace localization {
        class decoded;

        class extractor;
        class transformer;
        class loader;
    }

    class localization::decoded : public decoded_media {
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


    class localization::extractor : public interface::extractor<localization::decoded> {
    public:
        extractor(std::shared_ptr<const json_config_parser> = nullptr) {}

        virtual ~extractor() {}

        std::shared_ptr<localization::decoded> extract(const char* buf, int bufSize) override
        {
            if (bufSize != 4) {
                throw std::runtime_error("Only 4 byte buffers can be loaded as int32");
            }
            return std::make_shared<localization::decoded>(unpack_le<int>(buf));
        }
    };

    class localization::transformer : public interface::transformer<localization::decoded, nervana::params> {
    public:
        transformer(std::shared_ptr<const json_config_parser> = nullptr) {}

        virtual ~transformer() {}

        std::shared_ptr<localization::decoded> transform(
                            std::shared_ptr<nervana::params> txs,
                            std::shared_ptr<localization::decoded> mp) override { return mp; }
    };

    class localization::loader : public interface::loader<localization::decoded> {
    public:
        loader(std::shared_ptr<const json_config_parser> = nullptr) {}

        virtual ~loader() {}

        void load(char* buf, int bufSize, std::shared_ptr<localization::decoded> mp) override
        {
            int index = mp->get_index();
            memcpy(buf, &index, bufSize);
        }
    };
}
