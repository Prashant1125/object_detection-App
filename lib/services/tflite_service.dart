import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ObjectDetector {
  late Interpreter _interpreter;
  late List<String> labels;

  ObjectDetector._();

  static Future<ObjectDetector> create() async {
    final detector = ObjectDetector._();
    await detector._loadModel();
    return detector;
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      String labelsData = await rootBundle.loadString('assets/labels.txt');
      labels = labelsData.split('\n');
      print("✅ Model & Labels Loaded Successfully");
    } catch (e) {
      print("❌ Model Loading Failed: $e");
    }
  }

  Future<List<dynamic>?> runModelOnImage(File imageFile) async {
    if (_interpreter == null) {
      print("❌ Model not loaded yet!");
      return null;
    }

    // 📌 सही इनपुट डेटा तैयार करें
    var input = preprocessImage(imageFile);
    var inputTensor = _interpreter.getInputTensor(0);
    var outputTensor = _interpreter.getOutputTensor(0);

    print("✅ Expected Input Shape: ${inputTensor.shape}");
    print("✅ Expected Output Shape: ${outputTensor.shape}");

    // ✅ आउटपुट लिस्ट बनाएं
    var output = List.generate(
        outputTensor.shape[1], (i) => List.filled(outputTensor.shape[2], 0.0));

    // ✅ मॉडल को रन करें
    _interpreter.run(input, output);

    print("✅ Model Output: $output");
    return output;
  }

  Uint8List preprocessImage(File imageFile) {
    img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
    img.Image resizedImage = img.copyResize(image!, width: 224, height: 224);

    Uint8List inputBytes = Uint8List(224 * 224 * 3);
    int index = 0;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        img.Pixel pixel =
            resizedImage.getPixelSafe(x, y); // ✅ `getPixelSafe()` का उपयोग करें

        int red = pixel.r.toInt(); // 🔴 रेड वैल्यू
        int green = pixel.g.toInt(); // 🟢 ग्रीन वैल्यू
        int blue = pixel.b.toInt(); // 🔵 ब्लू वैल्यू

        inputBytes[index++] = red;
        inputBytes[index++] = green;
        inputBytes[index++] = blue;
      }
    }

    return inputBytes;
  }

  void close() {
    _interpreter.close();
  }
}
