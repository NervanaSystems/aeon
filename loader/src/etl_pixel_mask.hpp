#pragma once
#include "etl_interface.hpp"
#include "etl_image.hpp"
#include "params.hpp"
#include "util.hpp"

namespace nervana {

    namespace pixel_mask {
        class extractor;
        class transformer;
        class loader;
    }

    class pixel_mask::extractor : public interface::extractor<image::decoded> {
    public:
        extractor(const image::config& cfg);
        ~extractor();
        std::shared_ptr<image::decoded> extract(const char* buf, int bufSize);
    private:
        int _pixel_type;
        int _color_mode;
    };


    class pixel_mask::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(const image::config&);
        ~transformer();
        std::shared_ptr<image::decoded> transform(
                            std::shared_ptr<image::params> txs,
                            std::shared_ptr<image::decoded> mp) override;
    };

    class pixel_mask::loader : public interface::loader<image::decoded> {
    public:
        loader(const image::config& cfg);
        ~loader();
        void load(char* buf, std::shared_ptr<image::decoded> mp) override;
    private:
        const image::config& _cfg;
    };
}
