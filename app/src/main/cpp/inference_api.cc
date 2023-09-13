#include "xgen.h"
#include "src/main/cpp/inference_api.h"

#include <algorithm>
#include <android/log.h>
#include <jni.h>
#include <map>
#include <functional>
#include <memory>
#include <numeric>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define LOG_TAG "inference_api"
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__))

class XGenEngine {
public:
    XGenEngine(const char *pb_file, const char *data_file) {
//        LOGI("pb_file:%s data_file:%s", pb_file, data_file);
        h = XGenInitWithFiles(pb_file, data_file, XGenPowerDefault);
    }

    ~XGenEngine() {
        XGenShutdown(h);
        h = nullptr;
    }

    jobjectArray RunInference(JNIEnv *env, const jobjectArray input_data) {
        size_t input_tensor_num = XGenGetNumInputTensors(h);
//        LOGI("input_tensor_num:%zu", input_tensor_num);
        for (int i = 0; i < input_tensor_num; ++i) {
            auto array = (jfloatArray) env->GetObjectArrayElement(input_data, i);
            jfloat *array_data = env->GetFloatArrayElements(array, JNI_FALSE);
            XGenTensor *input_tensor = XGenGetInputTensor(h, i);
            size_t input_size_in_bytes = XGenGetTensorSizeInBytes(input_tensor);
            size_t input_size = input_size_in_bytes / sizeof(float);
//            LOGI("tensor_index:%d input_size:%zu", i, input_size);
            auto buffer = std::shared_ptr<float>(new float[input_size], std::default_delete<float[]>());
            for (int j = 0; j < input_size; ++j) {
                buffer.get()[j] = array_data[j];
            }
            XGenCopyBufferToTensor(input_tensor, buffer.get(), input_size_in_bytes);
        }

        if (XGenRun(h) != XGenOk) {
            LOGI("FATAL ERROR: XGen inference failed.");
            return nullptr;
        }

        size_t output_tensor_num = XGenGetNumOutputTensors(h);
//        LOGI("output_tensor_num:%zu", output_tensor_num);
        jclass floatArrayCls = env->FindClass("[F");
        jobjectArray jReturnData = env->NewObjectArray(output_tensor_num, floatArrayCls, nullptr);
        for (int i = 0; i < output_tensor_num; ++i) {
            XGenTensor *output_tensor = XGenGetOutputTensor(h, i);
            size_t output_size_in_bytes = XGenGetTensorSizeInBytes(output_tensor);
            size_t output_size = output_size_in_bytes / sizeof(float);
//            LOGI("tensor_index:%d output_size:%zu", i, output_size);
            auto output_data = std::shared_ptr<float>(new float[output_size], std::default_delete<float[]>());
            XGenCopyTensorToBuffer(output_tensor, output_data.get(), output_size_in_bytes);
            jfloatArray jOutputData = env->NewFloatArray(output_size);
            env->SetFloatArrayRegion(jOutputData, 0, (jsize) output_size, output_data.get());
            env->SetObjectArrayElement(jReturnData, i, jOutputData);
        }
        return jReturnData;
    }

private:
    XGenHandle *h;
};

static jlong jptr(XGenEngine *engine) {
    return (jlong) engine;
}

static struct XGenEngine *native(jlong ptr) {
    return (XGenEngine *) ptr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_cocopie_xgen_yolo_example_CoCoPIEJNIExporter_CreateOpt(JNIEnv *env, jclass instance, jstring pbPath, jstring dataPath) {
    auto *engine = new XGenEngine(env->GetStringUTFChars(pbPath, nullptr), env->GetStringUTFChars(dataPath, nullptr));
    return jptr(engine);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_com_cocopie_xgen_yolo_example_CoCoPIEJNIExporter_Inference(JNIEnv *env, jclass instance, jlong engine, jobjectArray input) {
    XGenEngine *xgen = native(engine);
    return xgen->RunInference(env, input);
}