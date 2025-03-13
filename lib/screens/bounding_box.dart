import 'package:flutter/material.dart';

class BoundingBox extends StatelessWidget {
  final Map<String, dynamic> detection;

  const BoundingBox(this.detection, {Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final rect = detection['rect'] as Map<String, double>;
    final confidence = detection['confidence'] as double;
    final label = detection['class'] as String;

    return Positioned(
      left: rect['x']! * MediaQuery.of(context).size.width,
      top: rect['y']! * MediaQuery.of(context).size.height,
      width: rect['w']! * MediaQuery.of(context).size.width,
      height: rect['h']! * MediaQuery.of(context).size.height,
      child: Container(
        decoration: BoxDecoration(
          border: Border.all(color: Colors.red, width: 2),
        ),
        child: Text(
          '$label ${(confidence * 100).toStringAsFixed(1)}%',
          style: TextStyle(
            color: Colors.white,
            fontSize: 12,
            backgroundColor: Colors.black.withOpacity(0.7),
          ),
        ),
      ),
    );
  }
}
