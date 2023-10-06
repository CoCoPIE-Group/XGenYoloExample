package com.cocopie.xgen.yolo.example

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.round

class ResultView(context: Context, attributeSet: AttributeSet) : View(context, attributeSet) {

    private var results: ArrayList<Result>? = null
    private lateinit var classes: Array<String>

    private val textPaint = Paint().also {
        it.textSize = 60f
        it.color = Color.WHITE
    }

    fun transform(results: ArrayList<Result>) {
        val scale = width / YoloX.INPUT_WIDTH.toFloat()
        val diffY = width - height

        results.forEach {
            it.rectF.left *= scale
            it.rectF.right *= scale
            it.rectF.top = it.rectF.top * scale - (diffY / 2f)
            it.rectF.bottom = it.rectF.bottom * scale - (diffY / 2f)
        }
        this.results = results

        invalidate()
    }

    override fun onDraw(canvas: Canvas?) {
        results?.forEach {
            canvas?.drawRect(it.rectF, findPaint(it.classIndex))
            canvas?.drawText(
                classes[it.classIndex] + ", " + round(it.score * 100) + "%",
                it.rectF.left + 10,
                it.rectF.top + 60,
                textPaint
            )
        }
        super.onDraw(canvas)
    }

    fun setClassLabel(classes: Array<String>) {
        this.classes = classes
    }

    private fun findPaint(classIndex: Int): Paint {
        val paint = Paint()
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 10.0f
        paint.strokeCap = Paint.Cap.ROUND
        paint.strokeJoin = Paint.Join.ROUND
        paint.strokeMiter = 100f

        paint.color = when (classIndex) {
            0, 45, 18, 19, 22, 30, 42, 43, 44, 61, 71, 72 -> Color.WHITE
            1, 3, 14, 25, 37, 38, 79 -> Color.BLUE
            2, 9, 10, 11, 32, 47, 49, 51, 52 -> Color.RED
            5, 23, 46, 48 -> Color.YELLOW
            6, 13, 34, 35, 36, 54, 59, 60, 73, 77, 78 -> Color.GRAY
            7, 24, 26, 27, 28, 62, 64, 65, 66, 67, 68, 69, 74, 75 -> Color.BLACK
            12, 29, 33, 39, 41, 58, 50 -> Color.GREEN
            15, 16, 17, 20, 21, 31, 40, 55, 57, 63 -> Color.DKGRAY
            70, 76 -> Color.LTGRAY
            else -> Color.DKGRAY
        }
        return paint
    }
}