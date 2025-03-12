import 'package:flutter/material.dart';

class BoundingBox extends StatelessWidget {
  final Map<String, dynamic> recognition;

  BoundingBox(this.recognition);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      left: recognition['boundingBox']['x'],
      top: recognition['boundingBox']['y'],
      width: recognition['boundingBox']['width'],
      height: recognition['boundingBox']['height'],
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.red, width: 2),
        ),
        child: Text(
          "${recognition['label']} (${(recognition['confidence'] * 100).toStringAsFixed(2)}%)",
          style: TextStyle(
            backgroundColor: Colors.black54,
            color: Colors.white,
            fontSize: 12,
          ),
        ),
      ),
    );
  }
}
