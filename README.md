# SmartPose

## Project Overview

SmartPose is an AI-powered pose analysis application.

## Key Features

- Real-time pose detection
- Pose comparison
- Alignment score
- Correction suggestions
## 1. System Architecture

SmartPose follows a real-time computer vision architecture consisting of five major components:

1. **Input Layer:** Captures live webcam frames or accepts uploaded reference images.
2. **Pose Detection Layer:** Uses Google's MediaPipe Pose Landmarker to identify 33 human body landmarks.
3. **Feature Extraction Layer:** Extracts normalized landmark coordinates and calculates biomechanical joint angles.
4. **Pose Analysis Layer:** Compares reference and live joint configurations to calculate pose similarity.
5. **Feedback Layer:** Generates alignment scores, visual indicators and corrective suggestions.

## 2. Computer Vision Pipeline

The application follows this processing pipeline:

Webcam Input → Frame Processing → Pose Detection → Landmark Extraction → Joint Angle Calculation → Pose Comparison → Alignment Score → Visual Feedback.

### Image Processing
- Capture video frames using OpenCV.
- Convert frames into the required image format.
- Process frames using the pretrained MediaPipe model.
- Extract landmark coordinates and visibility information.
- Render skeleton connections over the original video.

## 3. Pose Landmark Detection

SmartPose uses MediaPipe Pose Landmarker, which identifies 33 anatomical landmarks.

Important landmarks include:

| Body Part | Landmarks |
|---|---|
| Face | Nose, Eyes, Ears |
| Upper Body | Shoulders, Elbows, Wrists |
| Torso | Hips |
| Lower Body | Knees, Ankles, Heels |
| Feet | Foot Index |

Landmark coordinates are used for geometric calculations and pose analysis.

## 4. Biomechanical Joint Angle Calculation

Joint angles are calculated using three corresponding body landmarks.

For example:

- Elbow angle: Shoulder – Elbow – Wrist.
- Shoulder angle: Elbow – Shoulder – Hip.
- Knee angle: Hip – Knee – Ankle.

These measurements help identify differences between the reference pose and the user's live posture.

## 5. Pose Similarity Algorithm

The application compares corresponding anatomical joint angles between reference and live poses.

The comparison considers:

- Left and right elbow angles.
- Left and right shoulder angles.
- Left and right hip angles.
- Left and right knee angles.
- Spine verticality.
- Shoulder horizontal alignment.

The differences are used to calculate a weighted overall pose similarity score ranging from 0 to 100%.

## 6. Real-Time Feedback Mechanism

SmartPose analyzes joint deviations and generates corrective instructions.

Examples:

- Raise your left arm.
- Straighten your right leg.
- Adjust your shoulder position.
- Improve your torso alignment.

Visual indicators:
- Green: Properly aligned.
- Amber: Minor deviation.
- Red: Significant deviation.

## 7. Database Architecture

SmartPose uses SQLite for local data storage.

Database responsibilities:

- User account management.
- Authentication records.
- Session information.
- Pose analysis history.
- Captured analysis results.

## 8. Authentication and Security

The application implements:

- User registration and login.
- Password hashing using PBKDF2-HMAC-SHA256.
- Unique per-user salts.
- Input validation.
- Session management.
- Logout functionality.

## 9. Performance Optimization

The application uses:

- Pretrained pose detection instead of custom model training.
- NumPy vectorized operations for geometric calculations.
- Lightweight local processing.
- Streamlit for the interactive interface.
- OpenCV for real-time frame processing.

## 10. Limitations

- Pose detection accuracy depends on lighting conditions.
- Body visibility affects landmark detection.
- Camera angle can influence joint-angle measurements.
- Multiple people in the frame may affect analysis.
- Occluded body parts can reduce comparison reliability.

## 11. Future Enhancements

- Exercise repetition counting.
- Yoga pose classification.
- Advanced posture analysis.
- Personalized fitness recommendations.
- Historical progress visualization.
- Mobile application integration.
- Support for multiple-person pose analysis.
