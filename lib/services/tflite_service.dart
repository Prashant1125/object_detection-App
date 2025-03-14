import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ObjectDetector {
  Interpreter? _interpreter;
  List<String> _labels = [];
  TensorType? _inputType;
  static const double _confidenceThreshold = 0.5;

  ObjectDetector._();

  static Future<ObjectDetector> create() async {
    final detector = ObjectDetector._();
    await detector._initialize();
    return detector;
  }

  Future<void> _initialize() async {
    await _loadModel();
    await _loadLabels();
    _verifyModelStructure();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
    } catch (e) {
      throw Exception('''Model Error: $e
      1. Ensure assets/model.tflite exists
      2. Verify model compatibility with TFLite 2.x
      3. Check model file integrity''');
    }
  }

  Future<void> _loadLabels() async {
    try {
      final data = await rootBundle.loadString('assets/labels.txt');
      _labels =
          data.split('\n').where((line) => line.trim().isNotEmpty).toList();
      if (_labels.isEmpty) throw Exception('Labels file is empty');
    } catch (e) {
      throw Exception('Label Error: $e');
    }
  }

  void _verifyModelStructure() {
    if (_interpreter == null) throw Exception('Model not initialized');

    final inputTensor = _interpreter!.getInputTensor(0);
    _inputType = inputTensor.type;

    print('''
    === Model Details ===
    Input Shape: ${inputTensor.shape}
    Input Type: ${inputTensor.type}
    ''');
  }

  Future<List<Map<String, dynamic>>> detect(File imageFile) async {
    if (_interpreter == null) throw Exception('Model not initialized');

    try {
      // Step 1: Verify image file
      if (!await imageFile.exists()) throw Exception('Image file not found');

      // Step 2: Process image
      final image = await _processImage(imageFile);

      // Step 3: Create input tensor
      final input = _createInput(image);

      // Step 4: Run inference
      final output = _runInference(input);

      // Step 5: Process results
      return _processOutput(output);
    } catch (e) {
      throw Exception('Detection Failed: ${e.toString()}');
    }
  }

  Future<img.Image> _processImage(File file) async {
    try {
      final bytes = await file.readAsBytes();
      final image = img.decodeImage(bytes);
      if (image == null) throw Exception('Invalid image format');

      final resized = img.copyResize(image, width: 1, height: 1);
      if (resized.width != 1 || resized.height != 1) {
        throw Exception('Resizing failed');
      }
      return resized;
    } catch (e) {
      throw Exception('Image Error: ${e.toString()}');
    }
  }

  dynamic _createInput(img.Image image) {
    try {
      final pixel = image.getPixel(0, 0);

      if (_inputType == TensorType.uint8) {
        return Uint8List.fromList([
          pixel.r.clamp(0, 255).toInt(),
          pixel.g.clamp(0, 255).toInt(),
          pixel.b.clamp(0, 255).toInt(),
        ]);
      }

      return Float32List.fromList([
        (pixel.r / 255.0).clamp(0.0, 1.0),
        (pixel.g / 255.0).clamp(0.0, 1.0),
        (pixel.b / 255.0).clamp(0.0, 1.0),
      ]);
    } catch (e) {
      throw Exception('Input Error: ${e.toString()}');
    }
  }

  List<List<List<double>>> _runInference(dynamic input) {
    try {
      // Initialize output buffer
      final output = List<double>.filled(1 * 12804 * 4, 0.0);

      // Run inference
      _interpreter!.run(
        input is Uint8List
            ? input.reshape([1, 1, 1, 3])
            : (input as Float32List).reshape([1, 1, 1, 3]),
        {0: output},
      );

      // Reshape output
      return List.generate(
          1,
          (i) =>
              List.generate(12804, (j) => output.sublist(j * 4, (j + 1) * 4)));
    } catch (e) {
      throw Exception('Inference Error: ${e.toString()}');
    }
  }

  List<Map<String, dynamic>> _processOutput(List<List<List<double>>> output) {
    try {
      return output[0]
          .map((box) => {
                'rect': {
                  'x': box[0],
                  'y': box[1],
                  'width': box[2],
                  'height': box[3],
                },
                'confidence': _confidenceThreshold,
                'label': _labels.isNotEmpty ? _labels[0] : 'unknown',
              })
          .toList();
    } catch (e) {
      throw Exception('Output Error: ${e.toString()}');
    }
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
