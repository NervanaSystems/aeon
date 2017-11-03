#
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
#

set(SRC
    iterator.cpp
)

file(COPY ${TEST_DATA_DIR}/flowers.jpg DESTINATION .)
file(COPY ${TEST_DATA_DIR}/img_2112_70.jpg DESTINATION .)

add_executable(iterator ${SRC})
target_link_libraries(iterator aeon pthread sox ${CURL_LIBRARIES} opencv_imgproc opencv_highgui opencv_core ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
add_dependencies(iterator aeon)
