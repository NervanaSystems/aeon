extern "C" {

#include "im_struct.h"

extern ImageParams default_image(int height, int width) {
    ImageParams ss;
    ss._height = height;
    ss._width = width;
    ss._scaleMin = 100;
    return ss;
}

}
