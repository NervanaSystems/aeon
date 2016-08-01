#pragma once
#include "interface.hpp"
#include "etl_image.hpp"
#include "util.hpp"

namespace nervana {

    namespace pixel_mask {
        class extractor;
        class transformer;
        class loader;
    }

    //-------------------------------------------------------------------------
    // Extract
    //-------------------------------------------------------------------------

    class pixel_mask::extractor : public interface::extractor<image::decoded> {
    public:
        extractor(const image::config&);
        virtual ~extractor();
        virtual std::shared_ptr<image::decoded> extract(const char*, int) override;
    private:
    };

    //-------------------------------------------------------------------------
    // Transform
    //-------------------------------------------------------------------------

    class pixel_mask::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(const image::config&);
        ~transformer();
        std::shared_ptr<image::decoded> transform(
                            std::shared_ptr<image::params> txs,
                            std::shared_ptr<image::decoded> mp) override;
    };

    //-------------------------------------------------------------------------
    // Load
    //-------------------------------------------------------------------------

    // use image::loader
}
