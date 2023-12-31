package com.cocopie.xgen.yolo.example

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.core.graphics.applyCanvas
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.nio.FloatBuffer
import java.util.Collections
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class YoloX(val context: Context) {

    companion object {
        const val BATCH_SIZE = 1
        const val INPUT_WIDTH = 640
        const val INPUT_HEIGHT = 640
        const val PIXEL_SIZE = 3

        const val PREVIEW_WIDTH = 640 // Just like ncnn's demo
        const val PREVIEW_HEIGHT = 480

        const val ENGINE_XGEN_YOLOX = 0
        const val ENGINE_ONNX_YOLOX = 1

        const val XGEN_YOLOX_MODEL_NAME = "yolox_80_${INPUT_HEIGHT}x${INPUT_WIDTH}"
        const val ONNX_YOLOX_FILE_NAME = "yolox_80_${INPUT_HEIGHT}x${INPUT_WIDTH}.onnx"
        const val LABEL_YOLOX_FILE_NAME = "yolox_80.txt"

        private const val POST_PROCESS_ROW = INPUT_WIDTH / 8 * INPUT_HEIGHT / 8 + INPUT_WIDTH / 16 * INPUT_HEIGHT / 16 + INPUT_WIDTH / 32 * INPUT_HEIGHT / 32
    }

    private lateinit var classes: Array<String>
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var xGenYoloxEngine: Long = -1

    private val universalThreshold = 0.45f // 0.5 -> 0.45
    private val objThreshold = 0.0f // 0.45 -> 0.0
    private val nmsThreshold = 0.6f // 0.45 -> 0.6

    private val modelMean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f);
    private val modelStd: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f);

    var inferenceTime: Long = 0

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    private val inputBuffer = FloatBuffer.allocate(BATCH_SIZE * PIXEL_SIZE * INPUT_HEIGHT * INPUT_WIDTH)

    private val testBitmap by lazy { BitmapFactory.decodeResource(context.resources, R.drawable.test_2) }

    fun loadModel() {
        val assetManager = context.assets
        val outputFile = File(context.filesDir.toString(), ONNX_YOLOX_FILE_NAME)
        assetManager.open(ONNX_YOLOX_FILE_NAME).use { inputStream ->
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

        xGenYoloxEngine = loadXGen(XGEN_YOLOX_MODEL_NAME)
    }

    private fun loadXGen(model: String): Long {
        val pbFile = File(context.filesDir, "${model}.pb")
        val dataFile = File(context.filesDir, "${model}.data")
        CoCoPIEUtils.copyAssetsFile(context, pbFile.absolutePath, "${model}.pb")
        CoCoPIEUtils.copyAssetsFile(context, dataFile.absolutePath, "${model}.data")
        return CoCoPIEJNIExporter.CreateOpt(pbFile.absolutePath, dataFile.absolutePath)
    }

    fun loadLabel(): Array<String> {
        BufferedReader(InputStreamReader(context.assets.open(LABEL_YOLOX_FILE_NAME))).use { reader ->
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
//        val t = System.currentTimeMillis()
        val bitmap = imageProxy.toBitmap() // 640x480
        val bw = bitmap.width.toFloat()
        val bh = bitmap.height.toFloat()
        val scale = min(INPUT_WIDTH / bw, INPUT_HEIGHT / bh)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, (bw * scale).roundToInt(), (bh * scale).roundToInt(), true)
        val paddingBitmap = Bitmap.createBitmap(INPUT_WIDTH, INPUT_HEIGHT, Bitmap.Config.ARGB_8888)
        paddingBitmap.applyCanvas {
            drawColor(Color.rgb(114, 114, 114))
            drawBitmap(scaledBitmap, (INPUT_WIDTH - scaledBitmap.width) / 2f, (INPUT_HEIGHT - scaledBitmap.height) / 2f, paint)
        }
//        File(context.cacheDir, "${System.currentTimeMillis()}.jpg").writeBitmap(paddingBitmap)
//        Log.e("YoloX", "toBitmap ${System.currentTimeMillis() - t}")
        return paddingBitmap
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        // NCHW 1x3x640x640
        inputBuffer.rewind()
        val area = INPUT_WIDTH * INPUT_HEIGHT
        val bitmapData = IntArray(area)
        bitmap.getPixels(bitmapData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in 0 until INPUT_HEIGHT - 1) {
            for (j in 0 until INPUT_WIDTH - 1) {
                val idx = INPUT_WIDTH * i + j
                val pixelValue = bitmapData[idx]
                inputBuffer.put(idx, (pixelValue shr 16 and 0xff).toFloat())
                inputBuffer.put(idx + area, (pixelValue shr 8 and 0xff).toFloat())
                inputBuffer.put(idx + area * 2, (pixelValue and 0xff).toFloat())
            }
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun bitmapNormToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        // NCHW 1x3x640x640
        inputBuffer.rewind()
        val area = INPUT_WIDTH * INPUT_HEIGHT
        val bitmapData = IntArray(area)
        bitmap.getPixels(bitmapData, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in 0 until INPUT_HEIGHT - 1) {
            for (j in 0 until INPUT_WIDTH - 1) {
                val idx = INPUT_WIDTH * i + j
                val pixelValue = bitmapData[idx]
                inputBuffer.put(idx, ((pixelValue shr 16 and 0xFF) / 255f - modelMean[0]) / modelStd[0])
                inputBuffer.put(idx + area, ((pixelValue shr 8 and 0xFF) / 255f - modelMean[1]) / modelStd[1])
                inputBuffer.put(idx + area * 2, ((pixelValue and 0xFF) / 255f - modelMean[2]) / modelStd[2])
            }
        }
        inputBuffer.rewind()
        return inputBuffer
    }

    private fun xGenPrevProcess(imageProxy: ImageProxy): FloatBuffer {
        val bitmap = imageToBitmap(imageProxy)
        return bitmapToFloatBuffer(bitmap)
    }

    private fun onnxPrevProcess(imageProxy: ImageProxy): FloatBuffer {
        val bitmap = imageToBitmap(imageProxy)
        return bitmapToFloatBuffer(bitmap)
    }

    fun inference(imageProxy: ImageProxy, engine: Int): ArrayList<Result> {
        var time = System.currentTimeMillis()
        if (engine != ENGINE_ONNX_YOLOX) {
            val floatBuffer = xGenPrevProcess(imageProxy)
            Log.e("YoloX", "XGen PrevProcess:${System.currentTimeMillis() - time}")
            val inputArray = floatBuffer.array()
            time = System.currentTimeMillis()
            val xgenResult = CoCoPIEJNIExporter.Inference(xGenYoloxEngine, arrayOf(inputArray))
            inferenceTime = System.currentTimeMillis() - time
            Log.e("YoloX", "XGen Inference:${inferenceTime}")
            if (xgenResult.isNullOrEmpty()) {
                return ArrayList()
            }

            val outputArray = xgenResult[0] // 71400
            Log.e("YoloX", "XGen Output:" + outputArray.contentToString())
            time = System.currentTimeMillis()
            val result = xGenPostProcess(outputArray)
            Log.e("YoloX", "XGen PostProcess:${System.currentTimeMillis() - time}")
            return result
        } else {
            val floatBuffer = onnxPrevProcess(imageProxy)
            Log.e("YoloX", "ONNX PrevProcess:${System.currentTimeMillis() - time}")
            time = System.currentTimeMillis()
            val inputName = ortSession.inputNames.iterator().next()
            val inputShape = longArrayOf(
                BATCH_SIZE.toLong(),
                PIXEL_SIZE.toLong(),
                INPUT_HEIGHT.toLong(),
                INPUT_WIDTH.toLong()
            ) // NCHW 1x3x640x640
            val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, inputShape)
            val resultTensor = ortSession.run(Collections.singletonMap(inputName, inputTensor))
            val onnxResult = resultTensor.get(0).value as Array<*>
            inferenceTime = System.currentTimeMillis() - time
            Log.e("YoloX", "ONNX Inference:${inferenceTime}")

            val outputArray = onnxResult[0] as Array<FloatArray> // 8400x85
            Log.e("YoloX", "ONNX Output(${outputArray.size}):" + outputArray[0].contentToString())
            time = System.currentTimeMillis()
            val result = onnxPostProcess(outputArray)
            Log.e("YoloX", "ONNX PostProcess:${System.currentTimeMillis() - time}")
            return result
        }
    }

    private fun xGenPostProcess(outputs: FloatArray): ArrayList<Result> {
        val rows = POST_PROCESS_ROW
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
        val rows = POST_PROCESS_ROW
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
                    min(INPUT_WIDTH - 1f, xPos + width / 2f),
                    min(INPUT_HEIGHT - 1f, yPos + height / 2f)
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
                if (confidenceInClass > universalThreshold) {
                    val recognition = Result(detectionClass, confidenceInClass, rectF)
                    results.add(recognition)
                }
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