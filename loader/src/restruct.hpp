
class DecodedMedia {
public:
    DecodedMedia() {}
    virtual ~DecodedMedia();
}

class DecodedImage : public DecodedMedia {
public:
    DecodedImage() {}
    virtual ~DecodedImage();

    inline cv::Mat& getImgRef() {
        return _img;
    }

private:
    cv::Mat _img;
}



class Extracter {
public:
    Extracter(shared_ptr<ExtractParams> extract_params)
    : _extract_params(extract_params) {
    }

    virtual ~Extracter();
    virtual shared_ptr<DecodedMedia> decode(char* inbuf, int insize) = 0;

protected:
    shared_ptr<ExtractParams> _extract_params;
}


class ImageExtracter : public Extracter {
public:
    ~ImageExtracter() {}
    virtual shared_ptr<DecodedMedia> decode(char* inbuf, int insize) override;
}



class Transformer {
public:
    Transformer(shared_ptr<TransformParams> transform_params)
    : _transform_params(transform_params) {
    }

    virtual shared_ptr<DecodedMedia> transform(shared_ptr<DecodedMedia> input,
                                               shared_ptr<TransformSettings> settings) = 0;

protected:
    shared_ptr<TransformParams> _transform_params;
}


class ImageTransformer : public Transformer {
public:
    virtual shared_ptr<DecodedMedia> transform(shared_ptr<DecodedMedia> input,
                                               shared_ptr<TransformSettings> settings) override;

private:
    void rotate(const cv::Mat& input, cv::Mat& output, int angle);

    void resize(const cv::Mat& input, cv::Mat& output, const cv::Size2i& size);

    void lighting(cv::Mat& inout, float pixelstd[]);

    void cbsjitter(cv::Mat& inout, float cbs[]);

}



class Loader {
public:
    Loader(shared_ptr<LoaderParams> loader_params)
    : _loader_params(loader_params) {
    }
    virtual ~Loader();
    virtual void load(shared_ptr<DecodedMedia> input, char* outbuf, int outsize) = 0;

protected:
    shared_ptr<LoaderParams> _loader_params;
}


class ImageLoader : public Loader {
public:
    virtual void load(shared_ptr<DecodedMedia> input, char* outbuf, int outsize) override;

private:
    void split(cv::Mat& img, char* buf, int bufSize);
}


class Provider {
public:
    Provider(shared_ptr<Extracter> ex, shared_ptr<Transformer> tr, shared_ptr<Loader> lo)
    : _extracter(ex), _transformer(tr), _loader(lo) {
    }

    inline void provide(char *inbuf, int insize, char *outbuf, int outsize,
                        shared_ptr<TransformSettings> txs)
    {
        _loader->load(
                      _transformer->transform(
                                              _extracter->decode(inbuf, insize),
                                              txs),
                      outbuf, outsz);
    }

}

/*
// Sample code would do the following in setup:

shared_ptr<ImageExtracter> imgex = make_shared<ImageExtracter>(new ImageExParams());
shared_ptr<ImageTransformer> imgtr = make_shared<ImageTransformer>(new ImageTrParams());
shared_ptr<ImageLoader> imgex = make_shared<ImageLoader>(new ImageLoParams());

Provider imgprov(imgex, imgtr, imglo);

// Something that reads in metadata and makes bounding box targets
shared_ptr<JSONExtracter> jsonex = make_shared<JSONExtracter>(new JSONExParams());  // This first part is probably generic depending on how we are encoding our metadata
shared_ptr<BboxTransformer> bboxtr = make_shared<BboxTransformer>(new BboxTrParams());
shared_ptr<BboxLoader> bboxex = make_shared<BboxLoader>(new BboxLoParams());

Provider bboxprov(jsonex, bboxtr, bboxlo);


TransformSettings txs = blah; // Not sure how to deal with this just yet
imgprov.provide(recordbuf, recordsize, databuf, datasize, txs);
bboxprov.provide(metadatabuf, metadatasize, targetbuf, targetsize, txs);
*/

