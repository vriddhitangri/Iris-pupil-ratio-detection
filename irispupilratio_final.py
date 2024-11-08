import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Define iris landmarks
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Adjust scaling factor to process smaller frame for detection at larger distances
        scale_factor = 0.7  # Increase scale_factor to capture farther distance
        small_frame = cv.resize(rgb_frame, (int(img_w * scale_factor), int(img_h * scale_factor)))

        results = face_mesh.process(small_frame)

        if results.multi_face_landmarks:
            # Adjust landmark scaling for smaller frame
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w * scale_factor, img_h * scale_factor]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])

            # Get iris centers and radii
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])

            # Scale coordinates back to original size
            l_cx, l_cy, l_radius = int(l_cx / scale_factor), int(l_cy / scale_factor), int(l_radius / scale_factor)
            r_cx, r_cy, r_radius = int(r_cx / scale_factor), int(r_cy / scale_factor), int(r_radius / scale_factor)

            # Draw iris circles (purplish)
            cv.circle(frame, (l_cx, l_cy), int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, (r_cx, r_cy), int(r_radius), (255, 0, 255), 2, cv.LINE_AA)

            # Pupil detection improvements
            def detect_pupil(roi):
                if roi.size == 0:
                    return None, 0, 0  # Return if ROI is empty
                
                # Convert to grayscale and apply Gaussian Blur
                gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
                gray_blurred = cv.GaussianBlur(gray, (7, 7), 0)

                # Adaptive thresholding based on Otsu's method
                _, thresh = cv.threshold(gray_blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

                # Find contours for the pupil
                contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Filter contours by area to reduce false detections
                    valid_contours = [c for c in contours if 50 < cv.contourArea(c) < 300]  # Adjusted limits
                    if valid_contours:
                        # Find the largest valid contour
                        largest_contour = max(valid_contours, key=cv.contourArea)
                        (x, y), radius = cv.minEnclosingCircle(largest_contour)
                        return (int(x), int(y)), radius, cv.contourArea(largest_contour)

                return None, 0, 0

            # Extract the ROI and detect the left and right pupils
            left_roi = frame[max(0, int(l_cy - l_radius)):min(frame.shape[0], int(l_cy + l_radius)), 
                             max(0, int(l_cx - l_radius)):min(frame.shape[1], int(l_cx + l_radius))]
                             
            right_roi = frame[max(0, int(r_cy - r_radius)):min(frame.shape[0], int(r_cy + r_radius)), 
                              max(0, int(r_cx - r_radius)):min(frame.shape[1], int(r_cx + r_radius))]

            left_pupil_center, left_pupil_radius, left_pupil_area = detect_pupil(left_roi)
            right_pupil_center, right_pupil_radius, right_pupil_area = detect_pupil(right_roi)

            # Draw detected pupil information (red)
            if left_pupil_radius > 0:
                cv.circle(frame, (int(l_cx - l_radius + left_pupil_center[0]), 
                                   int(l_cy - l_radius + left_pupil_center[1])),
                          int(left_pupil_radius), (0, 0, 255), 2)  # Red circle for pupil

            if right_pupil_radius > 0:
                cv.circle(frame, (int(r_cx - r_radius + right_pupil_center[0]), 
                                   int(r_cy - r_radius + right_pupil_center[1])),
                          int(right_pupil_radius), (0, 0, 255), 2)  # Red circle for pupil

           

            # Calculate the pupil-iris ratio (adjusted to more realistic values)
            if left_pupil_radius > 0:
                left_ratio = left_pupil_radius / l_radius  # Changed to pupil-iris ratio
                cv.putText(frame, f'Left Pupil-Iris Ratio: {left_ratio:.2f}', (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

            if right_pupil_radius > 0:
                right_ratio = right_pupil_radius / r_radius  # Changed to pupil-iris ratio
                cv.putText(frame, f'Right Pupil-Iris Ratio: {right_ratio:.2f}', (50, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow('Iris and Pupil Detection', frame)
        if cv.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

cap.release()
cv.destroyAllWindows()
