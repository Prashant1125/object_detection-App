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

      labels = labelsData
          .split('\n')
          .map((label) => label.trim())
          .where((label) => label.isNotEmpty)
          .toList();

      if (labels.isEmpty) {
        throw Exception("❌ Labels file is empty.");
      }

      print("✅ Model & Labels Loaded Successfully");
    } catch (e) {
      print("❌ Model Loading Failed: $e");
    }
  }

  Uint8List? preprocessImage(File imageFile) {
    Uint8List imageBytes = imageFile.readAsBytesSync();

    if (imageBytes.isEmpty) {
      print("❌ Image file is empty.");
      return null;
    }

    img.Image? image = img.decodeImage(imageBytes);
    if (image == null) {
      print("❌ Image decoding failed.");
      return null;
    }

    img.Image resizedImage = img.copyResize(image, width: 224, height: 224);

    Uint8List inputBytes = Uint8List(224 * 224 * 3);
    int index = 0;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        img.Pixel pixel = resizedImage.getPixelSafe(x, y);

        int red = pixel.r.toInt();
        int green = pixel.g.toInt();
        int blue = pixel.b.toInt();

        inputBytes[index++] = red;
        inputBytes[index++] = green;
        inputBytes[index++] = blue;
      }
    }

    return inputBytes;
  }

  Future<List<dynamic>?> runModelOnImage(File imageFile) async {
    if (_interpreter == null) {
      print("❌ Model not loaded yet!");
      return null;
    }

    try {
      // 🔹 Image Processing
      Uint8List? inputBytes = preprocessImage(imageFile);
      if (inputBytes == null || inputBytes.isEmpty) {
        throw Exception("❌ Image processing failed.");
      }

      // 🔹 Input और Output Tensor चेक करें
      Tensor? inputTensor = _interpreter.getInputTensor(0);
      Tensor? outputTensor = _interpreter.getOutputTensor(0);

      if (inputTensor == null || outputTensor == null) {
        throw Exception(
            "❌ Tensor is null. Model may not be initialized properly.");
      }

      print("✅ Expected Input Shape: ${inputTensor.shape}");
      print("✅ Expected Output Shape: ${outputTensor.shape}");

      // 🔹 Input Data Initialization
      var inputShape = inputTensor.shape;
      var outputShape = outputTensor.shape;

      if (inputShape.isEmpty || outputShape.isEmpty) {
        throw Exception("❌ Tensor shape is invalid.");
      }

      var inputData = List.generate(
          inputShape[0],
          (i) => List.generate(
              inputShape[1],
              (j) => List.generate(
                  inputShape[2], (k) => List.filled(inputShape[3], 0))),
          growable: false);

      int index = 0;
      for (int i = 0; i < 224; i++) {
        for (int j = 0; j < 224; j++) {
          inputData[0][0][0][0] = inputBytes[index++];
        }
      }

      // 🔹 Output Data Initialization
      var output = List.generate(
          outputShape[0],
          (i) => List.generate(
              outputShape[1], (j) => List.filled(outputShape[2], 0.0)));

      if (output.isEmpty) {
        throw Exception("❌ Output tensor initialization failed.");
      }

      // 🔹 Run Model
      _interpreter.run(inputData, output);
      print("✅ Model Output: $output");

      return output;
    } catch (e, stackTrace) {
      print("❌ Error in runModelOnImage: $e");
      print(stackTrace);
      return null;
    }
  }

  void close() {
    _interpreter.close();
  }
}
