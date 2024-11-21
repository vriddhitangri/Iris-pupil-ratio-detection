import cv2 as cv 
import mediapipe as mp
import numpy as np

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Function to detect the pupil within the iris region
def detect_pupil(gray_frame, iris_center, iris_radius):
    x, y = iris_center
    roi_size = int(iris_radius * 1.2)  # Slightly larger than iris radius

    # Define the Region of Interest (ROI) around the iris
    #in open cv origin(0,0) is located at top-left
    x1 = max(0, x - roi_size)
    y1 = max(0, y - roi_size)
    x2 = min(gray_frame.shape[1], x + roi_size)
    y2 = min(gray_frame.shape[0], y + roi_size)
    roi = gray_frame[y1:y2, x1:x2]

    # Apply GaussianBlur to reduce noise
    roi_blur = cv.GaussianBlur(roi, (7, 7), 0)

    # Use HoughCircles to detect circles 
    # Uses canny edge detector internally(param1)
    # param2-Determines the number of votes a candidate circle needs to be considered valid

    circles = cv.HoughCircles(roi_blur, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                             param1=50, param2=30,
                             minRadius=int(iris_radius * 0.2),
                             maxRadius=int(iris_radius * 0.6))
    #If circle detected then the variable will have an NX3 array(N-no. of circles) with each row representing a circle((x_xenter,y_center),radius)


    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Assuming the first detected circle is the pupil
        pupil = circles[0][0]
        pupil_x = pupil[0] + x1 #(adjusting coordinates to full image)
        pupil_y = pupil[1] + y1
        pupil_radius = pupil[2]
        return (pupil_x, pupil_y), pupil_radius
    else:
        # Fallback to minMaxLoc if HoughCircles fails
        min_val, _, min_loc, _ = cv.minMaxLoc(roi_blur)
        pupil_center = (min_loc[0] + x1, min_loc[1] + y1)
        pupil_radius = int(iris_radius * 0.3)
        return pupil_center, pupil_radius

cap = cv.VideoCapture(0)


mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        img_h, img_w = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Extract landmarks for left and right irises
            mesh_points = np.array([
                np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                for p in results.multi_face_landmarks[0].landmark
            ])

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Left Iris
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            center_left = (int(l_cx), int(l_cy))
            # Detect pupil in left iris
            pupil_left, pupil_left_radius = detect_pupil(gray, center_left, l_radius)

            # Right Iris
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_right = (int(r_cx), int(r_cy))
            # Detect pupil in right iris
            pupil_right, pupil_right_radius = detect_pupil(gray, center_right, r_radius)

            # Calculate iris/pupil ratios
            left_ratio = l_radius / pupil_left_radius if pupil_left_radius != 0 else 0
            right_ratio = r_radius / pupil_right_radius if pupil_right_radius != 0 else 0

            # Draw Iris ( in green)
            cv.circle(frame, center_left, int(l_radius), (0, 255, 0), 2, cv.LINE_AA) 
            cv.circle(frame, center_right, int(r_radius), (0, 255, 0), 2, cv.LINE_AA)  

            # Draw Pupil(in blue)
            cv.circle(frame, pupil_left, int(pupil_left_radius), (255, 0, 0), 2, cv.LINE_AA)  
            cv.circle(frame, pupil_right, int(pupil_right_radius), (255, 0, 0), 2, cv.LINE_AA)  

            # Display information for Left Eye
            cv.putText(frame, f"Left Iris Radius: {int(l_radius)} px", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f"Left Pupil Radius: {int(pupil_left_radius)} px", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)
            cv.putText(frame, f"Left Ratio: {left_ratio:.2f}", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)

            # Display information for Right Eye
            cv.putText(frame, f"Right Iris Radius: {int(r_radius)} px", (10, 120),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(frame, f"Right Pupil Radius: {int(pupil_right_radius)} px", (10, 150),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)
            cv.putText(frame, f"Right Ratio: {right_ratio:.2f}", (10, 180),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA)


        resized_frame = cv.resize(frame, (800, 600))
        cv.imshow('Iris and Pupil Detection', resized_frame)

        
        key = cv.waitKey(1)
        if key == ord('q'):
            break


cap.release()
cv.destroyAllWindows()
