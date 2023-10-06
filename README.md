## 1 Introduction

This is an Android demo app to show how to integrate the output YoloX models by XGen into an android app.

## 2 Integration of XGen output into an app

Readers can see `app/src/main/cpp/inference_api.cc` to see how the output of XGen is used in an app for AI. This part gives the explanation.

#### 2.1 Import XGen SDK

Please refer to [XGen Document](https://xgen.cocopie.ai/v1.3.0/5_Results/) for XGen generated files.

Find `*.data`, `*.pb` files from `*compiled_file/android/model` under XGen workplace, and put them into `app/src/main/assets` in this project. In this example, they are renamed
to `yolox_80.data` and `yolox_80.pb`.

![](/images/location_of_data_and_pb.png)

Modify `app/src/main/java/com/cocopie/xgen/yolo/example/Yolox.kt`, change the variable `XGEN_YOLOX_MODEL_NAME` value to the file name of the above `*.data` `*.pb`.

Put your label file into`app/src/main/assets` and change the `LABEL_YOLOX_FILE_NAME` value accordingly.

![](/images/yolox.png)

Find `libxgen.so` from `*compiled_file/android/lib` under XGen workplace, and put it into `app/src/main/jniLibs/arm64-v8a` in this project.

![](/images/location_of_so.png)

If you want to compare the speed of the original onnx file with the XGen outputs, you can put the onnx file into the `app/src/main/assets` and toggle it by clicking on the 'USING
XGEN' or 'USING ONNX' tab in the top right corner on the app.

![](/images/universal_threshold.png)

You can modify the `universalThreshold` variable in the `app/src/main/java/com/cocopie/xgen/yolo/example/Yolox.kt` file to adjust the detection threshold

#### 2.2 Initialize XGen

Call the _XGenInitWithFiles_ method to initialize XGen.

#### 2.3 Run XGen

First call the _XGenCopyBufferToTensor_ method to pass the preprocessed data into XGen runtime.

Then call the _XGenRun_ method to let XGen runtime call the AI model to conduct an inference.

And finally call the _XGenCopyTensorToBuffer_ method to copy the result into a buffer.

```c
    jobjectArray RunInference(JNIEnv *env, const jobjectArray input_data) {
        size_t input_tensor_num = XGenGetNumInputTensors(h);
        for (int i = 0; i < input_tensor_num; ++i) {
            auto array = (jfloatArray) env->GetObjectArrayElement(input_data, i);
            jfloat *array_data = env->GetFloatArrayElements(array, JNI_FALSE);
            XGenTensor *input_tensor = XGenGetInputTensor(h, i);
            size_t input_size_in_bytes = XGenGetTensorSizeInBytes(input_tensor);
            size_t input_size = input_size_in_bytes / sizeof(float);
            LOGI("tensor_index:%d input_size:%zu", i, input_size);
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
        jclass floatArrayCls = env->FindClass("[F");
        jobjectArray jReturnData = env->NewObjectArray(output_tensor_num, floatArrayCls, nullptr);
        for (int i = 0; i < output_tensor_num; ++i) {
            XGenTensor *output_tensor = XGenGetOutputTensor(h, i);
            size_t output_size_in_bytes = XGenGetTensorSizeInBytes(output_tensor);
            size_t output_size = output_size_in_bytes / sizeof(float);
            auto output_data = std::shared_ptr<float>(new float[output_size], std::default_delete<float[]>());
            XGenCopyTensorToBuffer(output_tensor, output_data.get(), output_size_in_bytes);
            jfloatArray jOutputData = env->NewFloatArray(output_size);
            env->SetFloatArrayRegion(jOutputData, 0, (jsize) output_size, output_data.get());
            env->SetObjectArrayElement(jReturnData, i, jOutputData);
        }
        return jReturnData;
    }
```

#### 2.4 Screenshot

![](/images/screenshot_on_888.jpg)

#### 2.5 Performance comparison

Faster and smaller model files with almost identical accuracy.

| Model                | Terminal Latency (ms) | Demo FPS (Snapdragon 888) | Size (MB) |
|----------------------|-----------------------|---------------------------|-----------|
| Original_YOLOX(ONNX) | 714.3                 | 1.4                       | 28.0      |
| XGen_YOLOX(Large)    | 94.3                  | 10.6                      | 14.1      |
| XGen_YOLOX(Small)    | 49.7                  | 20.1                      | 2.99      |

## 3 Copyright

© 2022 CoCoPIE Inc. All Rights Reserved.

CoCoPIE Inc., its logo and certain names, product and service names reference herein may be registered trademarks, trademarks, trade names or service marks of CoCoPIE Inc. in
certain jurisdictions.

The material contained herein is proprietary, privileged and confidential and owned by CoCoPIE Inc. or its third-party licensors. The information herein is provided only to be
person or entity to which it is addressed, for its own use and evaluation; therefore, no disclosure of the content of this manual will be made to any third parities without
specific written permission from CoCoPIE Inc.. The content herein is subject to change without further notice. Limitation of Liability – CoCoPIE Inc. shall not be liable.

All other trademarks are the property of their respective owners. Other company and brand products and service names are trademarks or registered trademarks of their respective
holders.

Limitation of Liability

CoCoPIE Inc. shall not be liable.
