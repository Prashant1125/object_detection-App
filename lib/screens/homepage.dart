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
      _showError("Detector init failed: ${e.toString()}");
    }
  }

  Future<void> _pickImage() async {
    final file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    setState(() {
      _image = File(file.path);
      _isLoading = true;
    });

    await _detectObjects();
    setState(() => _isLoading = false);
  }

  // In your HomeScreen
  Future<void> _detectObjects() async {
    if (_image == null || _detector == null) return;

    setState(() => _isLoading = true);

    try {
      final results = await _detector!.detect(_image!);
      setState(() => _recognitions = results);
    } catch (e) {
      print(e.toString());
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Error: ${e.toString()}')));
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('1x1 Model Demo')),
      body: _buildMainContent(),
      floatingActionButton: FloatingActionButton(
        onPressed: _isLoading ? null : _pickImage,
        child: _isLoading
            ? const CircularProgressIndicator()
            : const Icon(Icons.image),
      ),
    );
  }

  Widget _buildMainContent() {
    if (_image == null) {
      return const Center(child: Text('Select an image'));
    }

    return Stack(
      children: [
        Center(child: Image.file(_image!, fit: BoxFit.contain)),
        if (_recognitions != null)
          ..._recognitions!.map((r) => BoundingBox(r)).toList(),
      ],
    );
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }
}
