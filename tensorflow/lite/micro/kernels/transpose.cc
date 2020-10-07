/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace transpose {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct TransposeContext {
    TransposeContext(TfLiteContext* context, TfLiteNode* node) {
        input = GetInput(context, node, 0);
        perm = GetInput(context, node, 1);
        output = GetOutput(context, node, 0);
    }
    const TfLiteTensor* input;
    const TfLiteTensor* perm;
    TfLiteTensor* output;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    TransposeContext op_context(context, node);

    // Ensure validity of input tensor.
    TF_LITE_ENSURE_MSG(context, NumDimensions(op_context.input) <= 5,
                        "Transpose op only supports 1D-5D input arrays.");
    TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                            op_context.output->type);

    return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TransposeContext op_context(context, node);

  // Retrieve the perm permutation array
  const int32_t* perm_data = GetTensorData<int32_t>(op_context.perm);
  
  // Determine the number of dimensions in the perm array
  const int size = op_context.perm->dims->data[0];

  // Prepare an params object to store the perm data whilst implementing
  // the conversion 
  TransposeParams params;
  params.perm_count = size;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }

  // TODO implement TRANSPOSE op

  return kTfLiteOk;
}

} // namespace transpose

TfLiteRegistration Register_TRANSPOSE() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/transpose::Prepare,
          /*invoke=*/transpose::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

} // namespace micro
} // namespace ops
} // namespace tflite
