# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

import unittest

import numpy as np
from test_pipelines_fastdeploy_common import FastDeployPipelineTesterMixin

from ppdiffusers import FastDeployStableDiffusionInpaintPipelineLegacy
from ppdiffusers.utils.testing_utils import (
    is_fastdeploy_available,
    load_image,
    nightly,
    require_fastdeploy,
)

if is_fastdeploy_available():
    import fastdeploy as fd


def create_runtime_option(device_id=-1, backend="paddle"):
    option = fd.RuntimeOption()
    if backend == "paddle":
        option.use_paddle_backend()
    else:
        option.use_ort_backend()
    if device_id == -1:
        option.use_cpu()
    else:
        option.use_gpu(device_id)
    return option


@require_fastdeploy
class FastDeployStableDiffusionInpaintPipelineLegacyFastTests(FastDeployPipelineTesterMixin, unittest.TestCase):
    # FIXME: add fast tests
    pass


@nightly
@require_fastdeploy
class FastDeployStableDiffusionInpaintPipelineLegacyIntegrationTests(FastDeployPipelineTesterMixin):
    @property
    def runtime_options(self):
        return {
            "text_encoder": create_runtime_option(0, "onnx"),  # use gpu
            "vae_encoder": create_runtime_option(0, "paddle"),  # use gpu
            "vae_decoder": create_runtime_option(0, "paddle"),  # use gpu
            "unet": create_runtime_option(0, "paddle"),  # use gpu
        }

    def test_inference(self):
        init_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/overture-creations-5sI6fQgYIuo.png"
        )
        mask_image = load_image(
            "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/overture-creations-5sI6fQgYIuo_mask.png"
        )

        # using the PNDM scheduler by default
        pipe = FastDeployStableDiffusionInpaintPipelineLegacy.from_pretrained(
            "CompVis/stable-diffusion-v1-4@fastdeploy",
            runtime_options=self.runtime_options,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "A red cat sitting on a park bench"

        generator = np.random.RandomState(0)
        output = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=15,
            generator=generator,
            output_type="np",
        )

        images = output.images
        image_slice = images[0, 255:258, 255:258, -1]

        assert images.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2514, 0.3007, 0.3517, 0.1790, 0.2382, 0.3167, 0.1944, 0.2273, 0.2464])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3
