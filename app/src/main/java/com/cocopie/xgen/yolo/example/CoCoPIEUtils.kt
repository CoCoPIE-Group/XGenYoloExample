package com.cocopie.xgen.yolo.example

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max
import kotlin.math.min

object CoCoPIEUtils {

    fun copyAssetsFile(context: Context, dstPath: String, srcPath: String) {
        if (File(dstPath).exists()) {
            return
        }
        try {
            val fileNames = context.assets.list(srcPath)
            if (!fileNames.isNullOrEmpty()) {
                val file = File(dstPath)
                if (!file.exists()) {
                    file.mkdirs()
                }
                for (fileName in fileNames) {
                    if (srcPath != "") { // assets 文件夹下的目录
                        copyAssetsFile(context, srcPath + File.separator + fileName, dstPath + File.separator + fileName)
                    } else { // assets 文件夹
                        copyAssetsFile(context, fileName, dstPath + File.separator + fileName)
                    }
                }
            } else {
                val outFile = File(dstPath)
                val aos = context.assets.open(srcPath)
                val fos = FileOutputStream(outFile)
                val buffer = ByteArray(1024)
                var byteCount: Int
                while (aos.read(buffer).also { byteCount = it } != -1) {
                    fos.write(buffer, 0, byteCount)
                }
                fos.flush()
                aos.close()
                fos.close()
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun boxIOU(a: RectF, b: RectF): Float {
        return boxIntersection(a, b) / boxUnion(a, b)
    }

    private fun boxIntersection(a: RectF, b: RectF): Float {
        val w = overlap(
            (a.left + a.right) / 2f, a.right - a.left,
            (b.left + b.right) / 2f, b.right - b.left
        )
        val h = overlap(
            (a.top + a.bottom) / 2f, a.bottom - a.top,
            (b.top + b.bottom) / 2f, b.bottom - b.top
        )
        return if (w < 0 || h < 0) 0f else w * h
    }

    private fun boxUnion(a: RectF, b: RectF): Float {
        val i: Float = boxIntersection(a, b)
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - i
    }

    private fun overlap(x1: Float, w1: Float, x2: Float, w2: Float): Float {
        val l1 = x1 - w1 / 2
        val l2 = x2 - w2 / 2
        val left = max(l1, l2)
        val r1 = x1 + w1 / 2
        val r2 = x2 + w2 / 2
        val right = min(r1, r2)
        return right - left
    }
}

fun File.writeBitmap(bitmap: Bitmap, format: Bitmap.CompressFormat = Bitmap.CompressFormat.JPEG, quality: Int = 100) {
    outputStream().use { out ->
        bitmap.compress(format, quality, out)
        out.flush()
    }
}