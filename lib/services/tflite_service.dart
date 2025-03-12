// object_detector.dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ObjectDetector {
  late Interpreter _interpreter;
  List<String> _labels = [];

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
      _labels = labelsData.split('\n').map((e) => e.trim()).toList();
      print("Model Loaded Successfully");
    } catch (e) {
      print("Model not Loaded $e");
    }
  }

  Future<List<dynamic>?> runModelOnImage(File imageFile) async {
    if (_interpreter == null) {
      print("❌ Model not loaded yet!");
      return null;
    }

    var input = preprocessImage(imageFile);

    var inputTensor = _interpreter.getInputTensor(0);
    var inputShape = inputTensor.shape;
    var outputTensor = _interpreter.getOutputTensor(0);
    var outputShape = outputTensor.shape;

    print(
        "✅ Corrected Input Tensor Shape: $inputShape, Output Tensor Shape: $outputShape");

    if (inputShape.length != 4 ||
        inputShape[1] != 224 ||
        inputShape[2] != 224 ||
        inputShape[3] != 3) {
      print(
          "❌ Input shape is incorrect! Expected [1, 224, 224, 3], but got $inputShape");
      return null;
    }

    var output = List.filled(outputShape[1], 0);
    _interpreter.run(input, output);

    print("✅ Model output: $output");
    return output;
  }

  Float32List preprocessImage(File imageFile) {
    img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
    img.Image resizedImage = img.copyResize(image!, width: 224, height: 224);

    Float32List convertedBytes = Float32List(224 * 224 * 3);
    int index = 0;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        img.Pixel pixel =
            resizedImage.getPixel(x, y); // ✅ Corrected: Pixel object

        convertedBytes[index++] = pixel.r.toDouble() / 255.0; // Normalize Red
        convertedBytes[index++] = pixel.g.toDouble() / 255.0; // Normalize Green
        convertedBytes[index++] = pixel.b.toDouble() / 255.0; // Normalize Blue
      }
    }

    return convertedBytes;
  }

  void close() {
    _interpreter.close();
  }
}
