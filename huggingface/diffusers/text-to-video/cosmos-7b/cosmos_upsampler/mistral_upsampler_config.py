# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from optimum.rbln import RBLNDecoderOnlyModelForCausalLMConfig


class RBLNMistralNeMoForTextUpsamplerConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    def __init__(self, precision="float32", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This variable "precision" is set to receive the weight dtype of the text-upsampler torch model.
        self.precision = precision
