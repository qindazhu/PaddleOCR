# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# compare this file with structure_system/params.py for changes

from deploy.hubserving.structure_table_en.params import read_params as table_read_params


def read_params():
    cfg = table_read_params()

    # params for layout parser model

    # just for table, works for en and ch
    cfg.layout_model_dir = './inference/picodet_lcnet_x1_0_fgd_layout_table_infer'
    cfg.layout_dict_path = './ppocr/utils/dict/layout_dict/layout_table_dict.txt'

    # layout (table, figure, footer, etc.) en
    # cfg.layout_model_dir = './inference/picodet_lcnet_x1_0_fgd_layout_infer'
    # cfg.layout_dict_path = './ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt'

    # layout (table, figure, footer, etc.) ch
    # cfg.layout_model_dir = './inference/picodet_lcnet_x1_0_fgd_layout_cdla_infer'
    # cfg.layout_dict_path = './ppocr/utils/dict/layout_dict/layout_cdla_dict.txt'

    cfg.layout_score_threshold = 0.5
    cfg.layout_nms_threshold = 0.5

    # disable ocr for non-table areas, as we just want the table result
    cfg.ocr = False

    cfg.mode = 'structure'
    cfg.output = './output'
    return cfg
