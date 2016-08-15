#include "provider_video_only.hpp"

using namespace nervana;
using namespace std;

video_only::video_only(nlohmann::json js) :
    video_config(js["video"]),
    video_extractor(video_config),
    video_transformer(video_config),
    video_loader(video_config),
    video_factory(video_config)
{
    num_inputs = 1;
    oshapes.push_back(video_config.get_shape_type());
}

void video_only::provide(int idx, buffer_in_array& in_buf, buffer_out_array& out_buf)
{
    std::vector<char>& datum_in  = in_buf[0]->get_item(idx);
    char* datum_out  = out_buf[0]->get_item(idx);

    if (datum_in.size() == 0) {
        std::stringstream ss;
        ss << "received encoded video with size 0, at idx " << idx;
        throw std::runtime_error(ss.str());
    }

    // Process video data
    auto video_dec = video_extractor.extract(datum_in.data(), datum_in.size());
    auto video_params = video_factory.make_params(video_dec);
    video_loader.load({datum_out}, video_transformer.transform(video_params, video_dec));
}
