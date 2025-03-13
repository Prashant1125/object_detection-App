import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ObjectDetector {
  late Interpreter _interpreter;
  late List<String> _labels;
  late TfLiteType _inputType;
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
      throw Exception('Model initialization failed: $e');
    }
  }

  Future<void> _loadLabels() async {
    final data = await rootBundle.loadString('assets/labels.txt');
    _labels = data.split('\n').where((line) => line.trim().isNotEmpty).toList();
    if (_labels.isEmpty) throw Exception('No labels found');
  }

  void _verifyModelStructure() {
    final inputTensor = _interpreter.getInputTensor(0);
    _inputType = inputTensor.type;
    print('Input type: $_inputType');
  }

  bool get _isQuantized => _inputType == TfLiteType.uint8;

  Future<List<Map<String, dynamic>>> detect(File imageFile) async {
    try {
      final image = await _processImage(imageFile);
      final input = _createInput(image);
      final output = _runInference(input);
      return _processOutput(output);
    } catch (e) {
      throw Exception('Detection failed: $e');
    }
  }

  Future<img.Image> _processImage(File file) async {
    final bytes = await file.readAsBytes();
    final image = img.decodeImage(bytes);
    if (image == null) throw Exception('Invalid image file');
    return img.copyResize(image, width: 1, height: 1);
  }

  dynamic _createInput(img.Image image) {
    final pixel = image.getPixel(0, 0);
    return _isQuantized
        ? _createQuantizedInput(pixel)
        : _createFloatInput(pixel);
  }

  Uint8List _createQuantizedInput(img.Pixel pixel) {
    return Uint8List.fromList([
      pixel.r.toInt(),
      pixel.g.toInt(),
      pixel.b.toInt(),
    ]);
  }

  Float32List _createFloatInput(img.Pixel pixel) {
    return Float32List.fromList([
      pixel.r / 255.0,
      pixel.g / 255.0,
      pixel.b / 255.0,
    ]);
  }

  List<List<List<double>>> _runInference(dynamic input) {
    final outputBuffer = List<double>.filled(1 * 12804 * 4, 0.0);
    final reshapedOutput = _reshapeOutput(outputBuffer);

    _interpreter.run(
      _isQuantized
          ? (input as Uint8List).reshape([1, 1, 1, 3])
          : (input as Float32List).reshape([1, 1, 1, 3]),
      {0: reshapedOutput},
    );

    return reshapedOutput;
  }

  List<List<List<double>>> _reshapeOutput(List<double> output) {
    return List.generate(1,
        (i) => List.generate(12804, (j) => output.sublist(j * 4, (j + 1) * 4)));
  }

  List<Map<String, dynamic>> _processOutput(List<List<List<double>>> output) {
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
  }

  void dispose() => _interpreter.close();
}
