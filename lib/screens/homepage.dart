import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:obj_detect/services/tflite_service.dart';
import 'bounding_box.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  List<Map<String, dynamic>>? _recognitions;
  final ImagePicker _picker = ImagePicker();
  ObjectDetector? _detector;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    try {
      _detector = await ObjectDetector.create();
    } catch (e) {
      _showError('Detector Init Failed: ${e.toString()}');
    }
  }

  Future<void> _pickImage() async {
    final file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    setState(() {
      _image = File(file.path);
      _recognitions = null;
      _isLoading = true;
    });

    await _detectObjects();
    setState(() => _isLoading = false);
  }

  Future<void> _detectObjects() async {
    if (_image == null || _detector == null) return;

    try {
      final results = await _detector!.detect(_image!);
      setState(() => _recognitions = results);
    } catch (e) {
      _showError(e.toString());
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(message),
      duration: const Duration(seconds: 5),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Object Detector')),
      body: _buildBody(),
      floatingActionButton: FloatingActionButton(
        onPressed: _isLoading ? null : _pickImage,
        tooltip: 'Pick Image',
        child: _isLoading
            ? const CircularProgressIndicator(color: Colors.white)
            : const Icon(Icons.image),
      ),
    );
  }

  Widget _buildBody() {
    return Stack(
      children: [
        if (_image != null)
          Center(child: Image.file(_image!, fit: BoxFit.contain)),
        if (_recognitions != null)
          ..._recognitions!.map((res) => BoundingBox(res)).toList(),
        if (_isLoading) const Center(child: CircularProgressIndicator()),
        if (_image == null)
          const Center(child: Text('Select an image to start detection')),
      ],
    );
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }
}
