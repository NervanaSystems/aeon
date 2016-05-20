typedef struct {
    int   _channelCount;
    int   _height;
    int   _width;
    bool  _center;
    bool  _flip;
    int   _scaleMin;
    int   _scaleMax;
    int   _contrastMin;
    int   _contrastMax;
    int   _rotateMin;
    int   _rotateMax;
    int   _aspectRatio;
    bool  _subtractMean;
    int   _redMean;
    int   _greenMean;
    int   _blueMean;
    int   _grayMean;
    float _colorNoiseStd;
} ImageParams;

ImageParams default_image(int height, int width);
