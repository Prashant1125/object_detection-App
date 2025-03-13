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
  List<dynamic>? _recognitions;
  final picker = ImagePicker();
  ObjectDetector? objectDetector;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    objectDetector = await ObjectDetector.create();
    setState(() {});
  }

  Future<void> _pickImage() async {
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await _detectObjects();
    }
  }

  Future<void> _detectObjects() async {
    if (_image == null || objectDetector == null) return;
    List<dynamic>? results = await objectDetector!.detect(_image!);
    setState(() {
      _recognitions = results;
    });
  }

  @override
  void dispose() {
    objectDetector?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Object Detection")),
      body: Stack(
        children: [
          if (_image != null) Image.file(_image!),
          if (_recognitions != null)
            ..._recognitions!.map((res) => BoundingBox(res)).toList(),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _pickImage,
        child: Icon(Icons.image),
      ),
    );
  }
}
