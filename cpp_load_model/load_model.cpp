#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* model_path = "models/ranking/tree_transformer.onnx";
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_smoke");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        Ort::Session session(env, model_path, session_options);

        std::vector<std::string> input_names_storage = session.GetInputNames();
        std::vector<std::string> output_names_storage = session.GetOutputNames();
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        input_names.reserve(input_names_storage.size());
        output_names.reserve(output_names_storage.size());

    // For debugging purposes, check input and output names and shapes
        std::cout << "Inputs:\n";
        for (size_t i = 0; i < input_names_storage.size(); ++i) {
            const std::string& name = input_names_storage[i];
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            std::cout << "  " << name << " shape=[";
            for (size_t d = 0; d < shape.size(); ++d) {
                std::cout << shape[d] << (d + 1 < shape.size() ? "," : "");
            }
            std::cout << "]\n";
        }

        std::cout << "Outputs:\n";
        for (size_t i = 0; i < output_names_storage.size(); ++i) {
            const std::string& name = output_names_storage[i];
            auto type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            std::cout << "  " << name << " shape=[";
            for (size_t d = 0; d < shape.size(); ++d) {
                std::cout << shape[d] << (d + 1 < shape.size() ? "," : "");
            }
            std::cout << "]\n";
        }

        for (const auto& name : input_names_storage) {
            input_names.push_back(name.c_str());
        }
        for (const auto& name : output_names_storage) {
            output_names.push_back(name.c_str());
        }

        // Define an example
        // Build variable-length inputs for token_ids, pos_encodings, token_mask.
        const int64_t batch = 1;
        const int64_t d_model = 40;

        // Example variable-length token IDs and positional encodings.
        std::vector<int64_t> token_ids = {
            3, 6, 7, 13, 9, 10, 7, 19, 8, 9, 10, 6, 7, 13, 9, 10, 7, 18, 8, 9, 10
        };
        std::vector<float> pos_enc = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        };
        int64_t seq = static_cast<int64_t>(token_ids.size());

        std::vector<int64_t> token_ids_shape = {batch, seq};
        std::vector<int64_t> pos_enc_shape = {batch, seq, d_model};
        std::vector<int64_t> token_mask_shape = {batch, seq};

        std::vector<uint8_t> token_mask(batch * seq, 0);  // all false (no padding)
        if (pos_enc.size() != static_cast<size_t>(seq * d_model)) {
            std::cerr << "pos_enc size mismatch: expected " << (seq * d_model)
                      << ", got " << pos_enc.size() << "\n";
            return 1;
        }

        // Wrap raw buffers into Ort::Value tensors for ONNX Runtime (CPU memory).
        // Each tensor uses the same shape as the model inputs:
        // - token_ids: int64 [B, T]
        // - pos_encodings: float32 [B, T, d_model]
        // - token_mask: bool [B, T] (true for padding; all false here)
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // token_ids: integer token indices.
        Ort::Value token_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem_info, token_ids.data(), token_ids.size(), token_ids_shape.data(), token_ids_shape.size());

        // pos_encodings: per-token positional vectors.
        Ort::Value pos_enc_tensor = Ort::Value::CreateTensor<float>(
            mem_info, pos_enc.data(), pos_enc.size(), pos_enc_shape.data(), pos_enc_shape.size());

        // token_mask: ONNX expects a boolean tensor, so we pass the raw byte buffer
        // and declare the element type as BOOL.
        Ort::Value token_mask_tensor = Ort::Value::CreateTensor(
            mem_info,
            token_mask.data(),
            token_mask.size() * sizeof(uint8_t),
            token_mask_shape.data(),
            token_mask_shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(3);
        input_tensors.emplace_back(std::move(token_ids_tensor));
        input_tensors.emplace_back(std::move(pos_enc_tensor));
        input_tensors.emplace_back(std::move(token_mask_tensor));

        // Run inference: feed input tensors by name and fetch requested outputs.
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names.data(),
            output_names.size());

        if (!output_tensors.empty()) {
            auto out_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto out_shape = out_info.GetShape();
            auto elem_type = out_info.GetElementType();
            std::cout << "Ran inference. Output[0] shape=[";
            for (size_t d = 0; d < out_shape.size(); ++d) {
                std::cout << out_shape[d] << (d + 1 < out_shape.size() ? "," : "");
            }
            std::cout << "]\n";

            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && out_shape.size() >= 2) {
                int64_t num_labels = out_shape.back();
                if (num_labels > 0) {
                    float* logits = output_tensors[0].GetTensorMutableData<float>();
                    // Sigmoid for the first batch element only.
                    std::cout << "Sigmoid (batch 0): [";
                    for (int64_t i = 0; i < num_labels; ++i) {
                        double x = static_cast<double>(logits[i]);
                        double p = 1.0 / (1.0 + std::exp(-x));
                        std::cout << p << (i + 1 < num_labels ? ", " : "");
                    }
                    std::cout << "]\n";
                }
            }
        }

    } catch (const Ort::Exception& ex) {
        std::cerr << "ONNX Runtime error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
