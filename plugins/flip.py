# Copyright 2017 Intel(R) Nervana(TM)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import cv2
import json
from plugin import Plugin


class plugin(Plugin):
    probability = 0.5
    do_flip = False
    width = 0

    def __init__(self, param_string):
        if len(param_string) > 0:
            params = json.loads(param_string)
            if "probability" in params:
                self.probability = params["probability"]
            if "width" in params:
                self.width = params["width"]
            else:
                raise KeyError('width required for flip.py')

    def prepare(self):
        self.do_flip = np.random.uniform() < self.probability

    def augment_image(self, mat):
        if self.do_flip:
            dst = cv2.flip(mat, 1)
        else:
            dst = mat
        return dst

    def augment_boundingbox(self, boxes):
        if self.do_flip:
            for i in range(len(boxes)):
                xmax = boxes[i]["xmax"]
                boxes[i]["xmax"] = self.width - boxes[i]["xmin"] - 1
                boxes[i]["xmin"] = self.width - xmax - 1
        return boxes

    def augment_pixel_mask(self, mat):
        return self.augment_image(mat)

    def augment_depthmap(self, mat):
        return self.augment_image(mat)
