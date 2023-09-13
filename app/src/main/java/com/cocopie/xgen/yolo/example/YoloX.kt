package com.cocopie.xgen.yolo.example

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.Collections
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min

class YoloX(val context: Context) {

    companion object {
        const val BATCH_SIZE = 1
        const val INPUT_SIZE = 640
        const val PIXEL_SIZE = 3

        const val XGEN_MODEL_NAME = "yolox_80_fp16"
        const val ONNX_FILE_NAME = "yolox_80.onnx"
        const val LABEL_NAME = "yolox_80.txt"
    }

    private lateinit var classes: Array<String>
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var xGenEngine: Long = -1

    private val objThreshold = 0.7f
    private val nmsThreshold = 0.45f

    private val testBitmap by lazy { BitmapFactory.decodeResource(context.resources, R.drawable.test_2) }

    fun loadModel() {
        val assetManager = context.assets
        val outputFile = File(context.filesDir.toString(), ONNX_FILE_NAME)
        assetManager.open(ONNX_FILE_NAME).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
        ortEnvironment = OrtEnvironment.getEnvironment()
        ortSession = ortEnvironment.createSession(
            outputFile.absolutePath,
            OrtSession.SessionOptions()
        )

        val model = XGEN_MODEL_NAME
        val pbFile = File(context.filesDir, "${model}.pb")
        val dataFile = File(context.filesDir, "${model}.data")
        CoCoPIEUtils.copyAssetsFile(context, pbFile.absolutePath, "${model}.pb")
        CoCoPIEUtils.copyAssetsFile(context, dataFile.absolutePath, "${model}.data")
        xGenEngine = CoCoPIEJNIExporter.CreateOpt(pbFile.absolutePath, dataFile.absolutePath)
    }

    fun loadLabel(): Array<String> {
        BufferedReader(InputStreamReader(context.assets.open(LABEL_NAME))).use { reader ->
            var line: String?
            val classList = ArrayList<String>()
            while (reader.readLine().also { line = it } != null) {
                classList.add(line!!)
            }
            classes = classList.toTypedArray()
        }
        return classes
    }

    private fun imageToBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        return Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        // NCHW 1x3x640x640
        val buffer = FloatBuffer.allocate(BATCH_SIZE * PIXEL_SIZE * INPUT_SIZE * INPUT_SIZE)
        buffer.rewind()

        val area = INPUT_SIZE * INPUT_SIZE
        val bitmapData = IntArray(area)
        bitmap.getPixels(bitmapData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        for (i in 0 until INPUT_SIZE - 1) {
            for (j in 0 until INPUT_SIZE - 1) {
                val idx = INPUT_SIZE * i + j
                val pixelValue = bitmapData[idx]
                buffer.put(idx, (pixelValue shr 16 and 0xff).toFloat())
                buffer.put(idx + area, (pixelValue shr 8 and 0xff).toFloat())
                buffer.put(idx + area * 2, (pixelValue and 0xff).toFloat())
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun prevProcess(imageProxy: ImageProxy): FloatBuffer {
        val bitmap = imageToBitmap(imageProxy)
        return bitmapToFloatBuffer(testBitmap)
    }

    fun inference(imageProxy: ImageProxy, useXGen: Boolean): Pair<ArrayList<Result>, Long> {
        val inferenceTime: Long
        var time = System.currentTimeMillis()
        val floatBuffer = prevProcess(imageProxy)
        Log.e("YoloX", "PrevProcess:${System.currentTimeMillis() - time}")
        if (useXGen) {
            val inputArray = floatBuffer.array()
            time = System.currentTimeMillis()
            val xgenResult = CoCoPIEJNIExporter.Inference(xGenEngine, arrayOf(inputArray))
            inferenceTime = System.currentTimeMillis() - time
            Log.e("YoloX", "XGen Inference:${inferenceTime}")
            if (xgenResult.isNullOrEmpty()) {
                return Pair(ArrayList(), inferenceTime)
            }

            val outputArray = xgenResult[0] // 71400
            Log.e("YoloX", "XGen Output:" + outputArray.contentToString())
            time = System.currentTimeMillis()
            val result = xGenPostProcess(outputArray)
            Log.e("YoloX", "XGen PostProcess:${System.currentTimeMillis() - time}")
            return Pair(result, inferenceTime)
        } else {
            val inputArray = floatBuffer.array()
            time = System.currentTimeMillis()
            val inputName = ortSession.inputNames.iterator().next()
            val inputShape = longArrayOf(
                BATCH_SIZE.toLong(),
                PIXEL_SIZE.toLong(),
                INPUT_SIZE.toLong(),
                INPUT_SIZE.toLong()
            ) // 1x3x640x640
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, inputShape)
            val resultTensor = ortSession.run(Collections.singletonMap(inputName, inputTensor))
            val onnxResult = resultTensor.get(0).value as Array<*>
            inferenceTime = System.currentTimeMillis() - time
            Log.e("YoloX", "ONNX Inference:${inferenceTime}")

            val outputArray = onnxResult[0] as Array<FloatArray> // 8400x85
            Log.e("YoloX", "ONNX Output:" + outputArray[0].contentToString())
            time = System.currentTimeMillis()
            val result = onnxPostProcess(outputArray)
            Log.e("YoloX", "ONNX PostProcess:${System.currentTimeMillis() - time}")
            return Pair(result, inferenceTime)
        }
    }

    private fun xGenPostProcess(outputs: FloatArray): ArrayList<Result> {
        val rows = 8400
        val cols: Int = classes.size + 5
        val output = Array(rows) { FloatArray(cols) }
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                output[i][j] = outputs[i * cols + j]
            }
        }
        return postProcess(output, rows)
    }

    private fun onnxPostProcess(output: Array<FloatArray>): ArrayList<Result> {
        val rows = 8400
        return postProcess(output, rows)
    }

    private fun postProcess(output: Array<FloatArray>, rows: Int): ArrayList<Result> {
        val results = ArrayList<Result>()
        for (i in 0 until rows) {
            val objConfidence = output[i][4]
            if (objConfidence > objThreshold) {
                val xPos = output[i][0]
                val yPos = output[i][1]
                val width = output[i][2]
                val height = output[i][3]
                val rectF = RectF(
                    max(0f, xPos - width / 2f),
                    max(0f, yPos - height / 2f),
                    min(INPUT_SIZE - 1f, xPos + width / 2f),
                    min(INPUT_SIZE - 1f, yPos + height / 2f)
                )
                val classConfidences = FloatArray(classes.size)
                System.arraycopy(output[i], 5, classConfidences, 0, classes.size)

                var detectionClass = -1
                var maxClassConfidence = 0f
                for (c in classes.indices) {
                    if (classConfidences[c] > maxClassConfidence) {
                        detectionClass = c
                        maxClassConfidence = classConfidences[c]
                    }
                }

                val confidenceInClass = maxClassConfidence * objConfidence
                val recognition = Result(detectionClass, confidenceInClass, rectF)
                results.add(recognition)
            }
        }
        return nms(results)
    }

    private fun nms(results: ArrayList<Result>): ArrayList<Result> {
        val list = ArrayList<Result>()
        for (i in classes.indices) {
            val pq = PriorityQueue<Result>(50) { o1, o2 ->
                o1.score.compareTo(o2.score)
            }
            val classResults = results.filter { it.classIndex == i }
            pq.addAll(classResults)

            while (pq.isNotEmpty()) {
                val detections = pq.toTypedArray()
                val max = detections[0]
                list.add(max)
                pq.clear()

                for (k in 1 until detections.size) {
                    val detection = detections[k]
                    val rectF = detection.rectF
                    if (CoCoPIEUtils.boxIOU(max.rectF, rectF) < nmsThreshold) {
                        pq.add(detection)
                    }
                }
            }
        }
        return list
    }
}