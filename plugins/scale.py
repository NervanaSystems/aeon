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

import numpy as np
import yaml
from plugin import Plugin


class plugin(Plugin):
    do_scale = False
    probability = 0
    amplitude = 0
    amplitude_max = 2
    amplitude_min = 0.1

    def __init__(self, param_string):
        if len(param_string) > 0:
            params = yaml.safe_load(param_string)
            if "probability" in params:
                self.probability = params["probability"]
            if "sample_freq_hz" in params:
                self.sample_freq_hz = params["sample_freq_hz"]
            if "amplitude" in params:
                self.amplitude_min = params["amplitude"][0]
                self.amplitude_max = params["amplitude"][1]

    def prepare(self):
        self.do_scale = np.random.uniform() < self.probability
        self.amplitude = np.random.uniform(self.amplitude_min,
                                           self.amplitude_max)

    def augment_audio(self, mat):
        if self.do_scale:
            mat2 = (mat.astype(np.float32) * self.amplitude).astype(np.int16)
            return mat2
        else:
            dst = mat
        return dst
