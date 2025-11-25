package com.lowasis.mnistnumberapp

import android.R.attr.bitmap
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.lowasis.mnistnumberapp.ui.theme.MnistNumberAppTheme

class MainActivity : ComponentActivity() {
    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val bitmap = result.data?.extras?.get("data") as? Bitmap
            if (bitmap != null) {
                // 액티비티 컨텍스트로 classifier 한번만 생성해서 재사용 권장
                val classifier = MnistClassifier(this, isQuantizedModel = false)
                val (pred, conf) = classifier.predict(bitmap, invert = true) // invert 필요성은 모델에 따라 조정
                classifier.close()

                if(conf > 0.65f) {
                    Toast.makeText(this, "예측: $pred (conf=${"%.2f".format(conf)})", Toast.LENGTH_LONG).show()
                } else {
                    Toast.makeText(this, "모르겠음.", Toast.LENGTH_LONG).show()
                }

            } else {
                Toast.makeText(this, "이미지 없음", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MnistNumberAppTheme {
                CameraButton { openCamera() }
            }
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(intent)
    }
}

@Composable
fun CameraButton(onClick: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center) {
        Button(onClick = onClick) {
            Text("카메라 실행")
        }
    }
}
