package com.cocopie.xgen.yolo.example

import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var rectView: ResultView
    private lateinit var engineBtn: Button
    private lateinit var infoView: TextView

    private val dataProcess = YoloX(context = this)

    private var engine = YoloX.ENGINE_XGEN_YOLOX

    @Volatile
    private var totalFps = 0f

    @Volatile
    private var fpsCount = 0

    companion object {
        const val PERMISSION = 1
    }

    private fun updateEngineBtn() {
        engineBtn.text = when (engine) {
            YoloX.ENGINE_XGEN_YOLOX -> "Using XGen"
            else -> "Using ONNX"
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        rectView = findViewById(R.id.rectView)
        engineBtn = findViewById(R.id.engine)
        engineBtn.setOnClickListener {
            engine++
            if (engine > YoloX.ENGINE_ONNX_YOLOX) {
                engine = YoloX.ENGINE_XGEN_YOLOX
            }
            updateEngineBtn()

            infoView.text = "Calculating"
            totalFps = 0f
            fpsCount = 0
        }
        updateEngineBtn()
        infoView = findViewById(R.id.info)

        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setPermissions()

        load()

        setCamera()
    }

    private fun setCamera() {
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER

        val cameraSelector = CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        val preview = Preview.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9).build()
        preview.setSurfaceProvider(previewView.surfaceProvider)

        val analysis = ImageAnalysis.Builder().setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            dataProcess.inference(it, engine).let { result ->
                runOnUiThread {
                    rectView.transform(result)
                    val fps = 1000f / dataProcess.inferenceTime
                    totalFps += fps
                    fpsCount++
                    // Displays the average frame rate of the last 10 frames
                    if (fpsCount == 10) {
                        infoView.text = getString(R.string.time, dataProcess.inferenceTime, totalFps / 10)
                        totalFps = 0f
                        fpsCount = 0
                    }
                }
            }
            it.close()
        }

        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)
    }

    private fun load() {
        dataProcess.loadModel()
        val classes = dataProcess.loadLabel()
        rectView.setClassLabel(classes)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        if (requestCode == PERMISSION) {
            grantResults.forEach {
                if (it != PackageManager.PERMISSION_GRANTED) {
                    finish()
                }
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

    private fun setPermissions() {
        val permissions = ArrayList<String>()
        permissions.add(android.Manifest.permission.CAMERA)
        permissions.forEach {
            if (ActivityCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions.toTypedArray(), PERMISSION)
            }
        }
    }
}