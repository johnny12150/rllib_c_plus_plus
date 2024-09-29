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
        // Prepare the input tensor for 'obs' (state)
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<int64_t> obs_shape = {1, static_cast<int64_t>(state.size())};  // Assuming a batch size of 1
        Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(state.data()), state.size(), obs_shape.data(), obs_shape.size());

        // Prepare the input tensor for 'state_ins' (empty or zero-sized input)
        std::vector<int64_t> state_ins_shape = {0};  // Assuming it's an empty tensor
        std::vector<float> state_ins;  // Empty state_ins input
        Ort::Value state_ins_tensor = Ort::Value::CreateTensor<float>(memory_info, state_ins.data(), state_ins.size(), state_ins_shape.data(), state_ins_shape.size());

        // Get input/output names (Assuming the model has two inputs: 'obs' and 'state_ins')
        Ort::AllocatedStringPtr input_name_0 = session->GetInputNameAllocated(0, allocator);  // 'obs'
        Ort::AllocatedStringPtr input_name_1 = session->GetInputNameAllocated(1, allocator);  // 'state_ins'
        Ort::AllocatedStringPtr output_name = session->GetOutputNameAllocated(0, allocator);  // Output name

        // Extract C-style strings
        const char* input_names[] = {input_name_0.get(), input_name_1.get()};
        const char* output_names[] = {output_name.get()};

        // Prepare input tensor array for inference
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(obs_tensor));  // Add 'obs' tensor
        input_tensors.push_back(std::move(state_ins_tensor));  // Add 'state_ins' tensor

        // Run inference
        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), input_tensors.size(), output_names, 1);

        // Get action probabilities from the output
        float* action_probs = output_tensors[0].GetTensorMutableData<float>();

        // Select action with the highest probability (assuming two action outputs)
        int action = std::max_element(action_probs, action_probs + 2) - action_probs;

        return action;
    }


private:
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;
};
