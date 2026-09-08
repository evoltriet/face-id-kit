# Third-party notices

Framework source is MIT-licensed. Dependencies and model weights retain their original licenses.

- OpenCV: Apache-2.0. https://github.com/opencv/opencv/blob/4.12.0/LICENSE
- YuNet model and upstream implementation: MIT, copyright Shiqi Yu. Full notice: licenses/YUNET.txt.
- SFace model: Apache-2.0. Full notice: licenses/SFACE.txt.
- Official model repository: https://github.com/opencv/opencv_zoo
- Face detection/recognition API: https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html

The framework's OpenCV adapter uses the public FaceDetectorYN and FaceRecognizerSF APIs. Model binaries are downloaded separately and verified against the SHA-256 values in face_id_kit.models. The MIT license of this framework does not relicense those models or bundled dependency packages.
