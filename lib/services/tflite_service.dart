import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ObjectDetector {
  late Interpreter _interpreter;
  late List<String> labels;
  static const int INPUT_SIZE = 300; // Update with your model's input size
  static const double THRESHOLD = 0.5; // Confidence threshold

  ObjectDetector._();

  static Future<ObjectDetector> create() async {
    final detector = ObjectDetector._();
    await detector._loadModel();
    return detector;
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      labels = await _loadLabels('assets/labels.txt');
      print("✅ Model & Labels Loaded Successfully");
    } catch (e) {
      print("❌ Model Loading Failed: $e");
      rethrow;
    }
  }

  Future<List<String>> _loadLabels(String path) async {
    try {
      final data = await rootBundle.loadString(path);
      return data.split('\n').where((label) => label.isNotEmpty).toList();
    } catch (e) {
      throw Exception("Failed to load labels: $e");
    }
  }

  img.Image _resizeAndConvertImage(img.Image image) {
    return img.copyResize(
      image,
      width: INPUT_SIZE,
      height: INPUT_SIZE,
      interpolation: img.Interpolation.nearest,
    );
  }

  Float32List _convertImageToFloat32List(img.Image image) {
    final inputShape = _interpreter.getInputTensor(0).shape;
    final inputSize = inputShape[1]; // Assuming NHWC format

    final resizedImage =
        img.copyResize(image, width: inputSize, height: inputSize);

    final bytes = resizedImage.getBytes(order: img.ChannelOrder.rgb);
    final floatBuffer = Float32List(inputSize * inputSize * 3);

    for (int i = 0; i < bytes.length; i += 3) {
      final r = bytes[i];
      final g = bytes[i + 1];
      final b = bytes[i + 2];

      // Update normalization based on your model's requirements
      floatBuffer[i] = (r / 255.0); // Example normalization
      floatBuffer[i + 1] = (g / 255.0);
      floatBuffer[i + 2] = (b / 255.0);
    }

    return floatBuffer;
  }

  List<Map<String, dynamic>> _processOutput(
    List<dynamic> detectionBoxes,
    List<dynamic> detectionClasses,
    List<dynamic> detectionScores,
  ) {
    final results = <Map<String, dynamic>>[];

    for (int i = 0; i < detectionScores.length; i++) {
      final score = detectionScores[i] as double;
      if (score < THRESHOLD) continue;

      final classIndex = detectionClasses[i] as int;
      final className = labels[classIndex];

      final box = detectionBoxes
          .sublist(i * 4, (i + 1) * 4)
          .map((e) => e as double)
          .toList();

      results.add({
        'class': className,
        'confidence': score,
        'rect': box,
      });
    }

    return results;
  }

  Future<List<Map<String, dynamic>>> detect(File imageFile) async {
    try {
      // Preprocess image
      final imageBytes = await imageFile.readAsBytes();
      final image = img.decodeImage(imageBytes);
      if (image == null) throw Exception("Failed to decode image");

      final processedImage = _resizeAndConvertImage(image);
      final inputBuffer = _convertImageToFloat32List(processedImage);

      // Setup output buffers
      final outputBoxes = List.filled(10 * 4, 0.0).reshape([1, 10, 4]);
      final outputClasses = List.filled(10, 0.0).reshape([1, 10]);
      final outputScores = List.filled(10, 0.0).reshape([1, 10]);
      final outputCount = List.filled(1, 0.0).reshape([1]);

      // Run inference
      _interpreter.run(
        {
          0: inputBuffer.reshape([1, INPUT_SIZE, INPUT_SIZE, 3])
        },
        {
          0: outputBoxes,
          1: outputClasses,
          2: outputScores,
          3: outputCount,
        },
      );

      // Process results
      return _processOutput(
        outputBoxes[0],
        outputClasses[0].map((e) => e.toInt()).toList(),
        outputScores[0],
      );
    } catch (e) {
      print("Detection error: $e");
      return [];
    }
  }

  void close() {
    _interpreter?.close();
  }
}
