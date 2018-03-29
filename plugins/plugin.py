# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import sys


class Plugin:
    def __init__(self):
        pass

    def prepare(self):
        print('prepare not implemented')
        raise RuntimeError('Not implemented')

    def augment_image(self, image):
        print('augment image not implemented')
        raise RuntimeError('Not implemented')

    def augment_boundingbox(self, bboxes):
        print('augment boundingbox not implemented')
        raise RuntimeError('Not implemented')

    def augment_pixel_mask(self, pixel_mask):
        print('augment pixel mask not implemented')
        raise RuntimeError('Not implemented')

    def augment_depthmap(self, depthmap):
        print('augment depthmap not implemented')
        raise RuntimeError('Not implemented')

    def augment_audio(self, audio):
        print('augment audio not implemented')
        raise RuntimeError('Not implemented')
