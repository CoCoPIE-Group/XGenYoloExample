#ifndef INFERENCE_API_H_
#define INFERENCE_API_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_cocopie_xgen_yolo_example_CoCoPIEJNIExporter_CreateOpt(JNIEnv *env, jclass instance,
                                                                jstring pbPath,
                                                                jstring dataPath);

JNIEXPORT jlong JNICALL
Java_com_cocopie_xgen_yolo_example_CoCoPIEJNIExporter_CreateFallback(JNIEnv *env, jclass instance,
                                                                     jstring fallbackPath);

JNIEXPORT jobjectArray JNICALL
Java_com_cocopie_xgen_yolo_example_CoCoPIEJNIExporter_Inference(JNIEnv *env, jclass instance,
                                                                jlong engine,
                                                                jobjectArray input);

#ifdef __cplusplus
}
#endif

#endif
