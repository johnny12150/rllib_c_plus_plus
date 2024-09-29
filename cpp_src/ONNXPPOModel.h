#include "onnxruntime-osx-arm64-1.19.2/include/onnxruntime_cxx_api.h"
#include <iostream>
#include <vector>
#include <cmath>

class ONNXPPOModel {
public:
    ONNXPPOModel(const std::string& model_path) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PPOModel");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    }

    int select_action(const std::vector<float>& state) {
        // Prepare the input tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(state.size())};  // 1 batch, state size
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(state.data()), state.size(), input_shape.data(), input_shape.size());

        // Get input/output names
        Ort::AllocatedStringPtr input_name = session->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(0, allocator);

        // Extract C-style strings
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        // Run inference
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Get action probabilities from the output
        float* action_probs = output_tensors[0].GetTensorMutableData<float>();

        // Select action with the highest probability
        int action = std::max_element(action_probs, action_probs + 2) - action_probs;

        return action;
    }

private:
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;
};
