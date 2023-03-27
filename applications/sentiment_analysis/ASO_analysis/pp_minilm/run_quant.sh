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

export CUDA_VISIBLE_DEVICES=0

python  quant_post.py \
        --base_model_name "ppminilm-6l-768h" \
        --static_model_dir "../checkpoints/pp_checkpoints/static" \
        --quant_model_dir "../checkpoints/pp_checkpoints/quant" \
        --algorithm "avg" \
        --dev_path "../data/cls_data/dev.txt" \
        --label_path "../data/cls_data/label.dict" \
        --batch_size 4 \
        --max_seq_len 256 \
        --save_model_filename "infer.pdmodel" \
        --save_params_filename "infer.pdiparams" \
        --input_model_filename "infer.pdmodel" \
        --input_param_filename "infer.pdiparams"

