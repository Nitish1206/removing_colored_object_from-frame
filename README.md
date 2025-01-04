# Object Tracking

This code implements a simple object tracking algorithm in Python using OpenCV. It can track a single colored object in a video stream.

## Dependencies

This code requires the following Python libraries:

* OpenCV
* NumPy
* collections

## How to Run

1. Clone this repository.
2. Install the required libraries using `pip install opencv-python numpy collections`.
3. Save a video file as `video.mp4` in the same directory as the code.
4. Run the script using `python object_tracking.py`.

## Explanation

The code first defines a function to convert a rectangle object to bounding box coordinates. Then, it defines a function to calculate the distance between two points. The main part of the code opens the video capture object and reads frames from the video stream. It then converts the frame to HSV color space and defines the color range for the object to be tracked (green in this example).

The code uses inRange function to create a color mask for the object and applies morphological operations to remove noise and improve the mask. It then finds contours in the mask and identifies the contour with the largest area as the object to track.

The code calculates the moments of the largest contour and uses them to find the center of the object. It then inpaints the object in the original frame and applies Canny edge detection to refine the object boundaries. Finally, it displays the original frame with the tracked object highlighted.

## Note

This is a basic implementation of object tracking and may not work perfectly in all scenarios. You can improve the code by:

* Implementing more sophisticated tracking algorithms like Kalman filters.
* Using a pre-trained object detection model to detect and track multiple objects.
