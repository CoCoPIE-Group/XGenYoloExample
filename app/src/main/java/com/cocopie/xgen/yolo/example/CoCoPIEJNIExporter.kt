package com.cocopie.xgen.yolo.example

object CoCoPIEJNIExporter {

    init {
        try {
            System.loadLibrary("inference_api_jni")
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    @JvmStatic
    external fun CreateOpt(pbPath: String?, dataPath: String?): Long

    @JvmStatic
    external fun Inference(engine: Long, input: Array<FloatArray>): Array<FloatArray>?
}