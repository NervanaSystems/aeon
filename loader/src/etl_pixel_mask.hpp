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

    // image extractor used


    class pixel_mask::transformer : public interface::transformer<image::decoded, image::params> {
    public:
        transformer(const image::config&);
        ~transformer();
        std::shared_ptr<image::decoded> transform(
                            std::shared_ptr<image::params> txs,
                            std::shared_ptr<image::decoded> mp) override;
    };

    // image loader used
}
