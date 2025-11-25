package com.lowasis.mnistnumberapp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import androidx.core.graphics.scale
import androidx.core.graphics.get

class MnistClassifier(
    private val context: Context,
    private val isQuantizedModel: Boolean = false // 모델 타입에 맞춰 설정
) {

    private val interpreter: Interpreter
    private val inputImageSize = 28
    private val numBytesPerChannel = if (isQuantizedModel) 1 else 4

    init {
        val options = Interpreter.Options()
        interpreter = Interpreter(loadModelFile(), options)
    }

    private fun loadModelFile(): ByteBuffer {
        val afd = context.assets.openFd("mnist_model.tflite")
        val inputStream = FileInputStream(afd.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = afd.startOffset
        val declaredLength = afd.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Bitmap -> ByteBuffer (모델 입력 형태)
    fun preprocess(bitmap: Bitmap, invert: Boolean): ByteBuffer {
        // MNIST 모델은 28x28 흑백 이미지를 기대
        val inputSize = 28
        val resized = bitmap.scale(inputSize, inputSize, filter = true) //부드럽게 리사이징 되로고 하기 위하여 True추가.

        val byteBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 1 * 4) // float32
        byteBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val pixel = resized[x, y]
                val r = Color.red(pixel)
                val g = Color.green(pixel)
                val b = Color.blue(pixel)

                // 흑백 변환 (0.0 ~ 1.0)
                var normalized = (r + g + b) / 3.0f / 255.0f

                // 색상 반전: MNIST는 흰 글씨(1.0) / 검은 배경(0.0)
                if(invert) {
                    normalized = 1.0f - normalized
                }

                // [개선] 대비 향상 및 노이즈 제거
                // 반전 후(검은 배경, 흰 글씨) 기준으로, 어두운 회색(종이 배경 등)은 0으로, 밝은 회색(글씨)은 1로 강조
                // 0.2 이하는 0.0(완전 검정)으로, 그 이상은 값을 키워서 선명하게 만듦
                val threshold = 0.2f
                normalized = (normalized - threshold).coerceAtLeast(0f) / (1.0f - threshold)
                normalized = normalized.coerceAtMost(1.0f)

                byteBuffer.putFloat(normalized)
            }
        }
        return byteBuffer
    }

    /**
     * predict: Bitmap 받아서 (예측값, confidence) 반환
     */
    fun predict(bitmap: Bitmap, invert: Boolean = false): Pair<Int, Float> {
        val inputBuffer = preprocess(bitmap, invert)

        return if (isQuantizedModel) {
            val output = Array(1) { ByteArray(10) }
            interpreter.run(inputBuffer, output)
            val scores = FloatArray(10)
            for (i in 0 until 10) {
                val v = output[0][i].toInt() and 0xFF
                scores[i] = v / 255.0f
            }
            val maxIndex = scores.indices.maxByOrNull { scores[it] } ?: -1
            val confidence = if (maxIndex >= 0) scores[maxIndex] else 0f
            Pair(maxIndex, confidence)
        } else {
            val output = Array(1) { FloatArray(10) }
            interpreter.run(inputBuffer, output)
            val probs = output[0]
            val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: -1
            val confidence = if (maxIndex >= 0) probs[maxIndex] else 0f
            Pair(maxIndex, confidence)
        }
    }

    fun close() {
        interpreter.close()
    }
}