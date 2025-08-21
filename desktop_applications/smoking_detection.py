import tensorflow as tf
import tensorflow_hub as hub
import torch
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml

# Load custom object detection model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model_SD_large_v2_noiseFree.pt')

class SmokerLabel:
    CONFIDENT_SMOKER = "confident_smoker"
    POSSIBLE_SMOKER = "possible_smoker"
    NON_SMOKER = "nonSmoker"
    CIGARETTE = "cigarette"

def load_models():
    """Load both YOLO and MoveNet models"""
    try:
        object_model = YOLO(MODEL_PATH)
        landmark_model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
        return object_model, landmark_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

def preprocess_image(image_path):
    """Preprocess image for MoveNet"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    orig_height, orig_width = image.shape[:2]

    input_size = 256
    scale = min(input_size / orig_width, input_size / orig_height)
    new_height, new_width = int(orig_height * scale), int(orig_width * scale)

    resized_image = tf.image.resize(image, (new_height, new_width))
    padded_image = tf.image.pad_to_bounding_box(resized_image, 0, 0, input_size, input_size)
    input_image = tf.cast(padded_image, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)

    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    scale_x, scale_y = orig_width / new_width, orig_height / new_height
    shift_flag = (orig_width - orig_height) / 2
    print("Shift flag:", shift_flag)

    return input_image, (scale_x, scale_y), (pad_x, pad_y), orig_height, orig_width, shift_flag

def run_movenet(image_path, landmark_model):
    """Run MoveNet Pose Estimation"""
    input_image, (scale_x, scale_y), (pad_x, pad_y), orig_height, orig_width, shift_flag = preprocess_image(image_path)
    movenet_signature = landmark_model.signatures['serving_default']
    outputs = movenet_signature(input_image)
    keypoints = outputs['output_0'].numpy()
    return keypoints, (scale_x, scale_y), (pad_x, pad_y), orig_height, orig_width, shift_flag

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def non_max_suppression(boxes, iou_threshold=0.5):
    """Perform Non-Maximum Suppression on bounding boxes"""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
    filtered_boxes = []

    while boxes:
        best_box = boxes.pop(0)
        filtered_boxes.append(best_box)

        # Keep boxes only if they don't overlap significantly with the best box
        boxes = [
            box for box in boxes
            if calculate_iou(best_box["coords"], box["coords"]) < iou_threshold
        ]

    return filtered_boxes

def calculate_center(box):
    """Calculate center point of a bounding box"""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def is_point_in_box(point, box, tolerance=20):
    """Check if a point is inside a bounding box with tolerance"""
    x, y = point
    x1, y1, x2, y2 = box
    
    x1 -= tolerance
    y1 -= tolerance
    x2 += tolerance
    y2 += tolerance
    
    return x1 <= x <= x2 and y1 <= y <= y2

def process_pose_landmarks(pose_keypoints, scale_x, scale_y, pad_x, pad_y, shift_flag):
    """Process landmarks for a pose"""
    processed_landmarks = []
    for i in [0, 9, 10]:  # Nose and wrists
        y, x, confidence = pose_keypoints[i * 3:i * 3 + 3]
        if confidence > 0.25:
            adjusted_x = (x * 256 - pad_x) * scale_x
            adjusted_y = ((y * 256 - pad_y) * scale_y) + shift_flag
            processed_landmarks.append({
                "id": i,
                "coords": (int(adjusted_x), int(adjusted_y)),
                "confidence": confidence,
                "original_coords": (x, y)
            })
    return processed_landmarks

def preprocess_all_poses(detected_poses, scale_x, scale_y, pad_x, pad_y, shift_flag):
    """Process all poses at once"""
    processed_poses = []
    for pose_keypoints in detected_poses:
        landmarks = process_pose_landmarks(pose_keypoints, scale_x, scale_y, pad_x, pad_y, shift_flag)
        if landmarks:
            processed_poses.append({
                "landmarks": landmarks,
                "pose_id": len(processed_poses),
                "assigned": False,
                "average_confidence": sum(l["confidence"] for l in landmarks) / len(landmarks)
            })
    return processed_poses

def find_best_person_for_landmarks(landmarks, persons_boxes):
    """Find best matching person for landmarks"""
    best_person = None
    max_valid_landmarks = 0
    
    for person in persons_boxes:
        valid_landmarks = 0
        for landmark in landmarks:
            if is_point_in_box(landmark["coords"], person["coords"]):
                valid_landmarks += 1
        
        if valid_landmarks > max_valid_landmarks:
            max_valid_landmarks = valid_landmarks
            best_person = person
    
    return best_person if max_valid_landmarks > 0 else None

def detect_smoking(image_path, object_model, landmark_model, save_output=True):
    """Main function to detect smoking in an image"""
    try:
        # Load and process image
        image = Image.open(image_path)
        results = object_model(image)  # Pass PIL Image directly to YOLO
        image_np = np.array(image)  # Convert to numpy array after YOLO

        # Get pose estimation
        keypoints, (scale_x, scale_y), (pad_x, pad_y), orig_height, orig_width, shift_flag = run_movenet(image_path, landmark_model)
        
        # Process detections
        persons_boxes = []
        cigarette_boxes = []

        for box in results[0].boxes:
            class_id = int(box.cls)
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_id == 2:  # Smoker class
                persons_boxes.append({
                "coords": (x1, y1, x2, y2),
                "class_id": class_id,
                "confidence": confidence,
                "yolo_smoker": True,  # YOLO predicted smoker
                "pipeline_smoker": False,  # Will be set by cigarette detection
                "final_label": None})
            elif class_id == 1:  # Non-smoker class
                persons_boxes.append({
                "coords": (x1, y1, x2, y2),
                "class_id": class_id,
                "confidence": confidence,
                "yolo_smoker": False,  # YOLO predicted non-smoker
                "pipeline_smoker": False,  # Will be set by cigarette detection
                "final_label": None})
            elif class_id == 0:  # Cigarette class
                cigarette_boxes.append({"coords": (x1, y1, x2, y2), "confidence": confidence})


        # Apply NMS and process poses
        persons_boxes = non_max_suppression(persons_boxes)
        processed_poses = preprocess_all_poses(keypoints[0], scale_x, scale_y, pad_x, pad_y, shift_flag)
        
        # Process each person
        annotated_image = image_np.copy()
        person_landmarks = []

        for person in persons_boxes:
            # Process landmarks and assign poses
            x1, y1, x2, y2 = person["coords"]
            person_center = calculate_center((x1, y1, x2, y2))
            person["landmarks"] = []

            # Find best matching pose
            best_pose = None
            min_distance = float('inf')

            for pose in processed_poses:
                if not pose["assigned"]:
                    valid_landmarks = sum(1 for lm in pose["landmarks"] 
                                    if is_point_in_box(lm["coords"], (x1, y1, x2, y2)))
                    
                    if valid_landmarks > 0:
                        valid_positions = [lm["coords"] for lm in pose["landmarks"] 
                                        if is_point_in_box(lm["coords"], (x1, y1, x2, y2))]
                        avg_x = sum(x for x, _ in valid_positions) / len(valid_positions)
                        avg_y = sum(y for _, y in valid_positions) / len(valid_positions)
                        
                        distance = np.sqrt((person_center[0] - avg_x)**2 + (person_center[1] - avg_y)**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            best_pose = pose

            if best_pose is not None:
                best_person = find_best_person_for_landmarks(best_pose["landmarks"], persons_boxes)
                
                if best_person is None or best_person["coords"] == person["coords"]:
                    person["landmarks"].extend(best_pose["landmarks"])
                    best_pose["assigned"] = True

            person_landmarks.append(person)

            # Plot landmarks for the current person
            person_colors = [
                (255, 0, 0),    # Red
                (0, 255, 0),    # Green
                (0, 0, 255),    # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),   # Cyan
                
            ]
            # Get unique color for current person
            person_color = person_colors[(len(person_landmarks) - 1) % len(person_colors)]

            # Plot landmarks for this person
            for landmark in person["landmarks"]:
                # Draw landmark point
                cv2.circle(annotated_image, landmark["coords"], 5, person_color, -1)
                
                # Add label with landmark ID and confidence
                label_text = f"P{len(person_landmarks)-1}L{landmark['id']} ({landmark['confidence']:.2f})"
                cv2.putText(
                    annotated_image,
                label_text,
                (landmark["coords"][0] + 5, landmark["coords"][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                person_color,
                1)

            # Add visualization for person-landmark association

            # Draw person's bounding box center
            x1, y1, x2, y2 = person["coords"]
            box_center = calculate_center((x1, y1, x2, y2))
            cv2.circle(annotated_image, (int(box_center[0]), int(box_center[1])), 8, person_color, -1)
        
            # Label the box center
            cv2.putText(
                annotated_image,
                f"P{len(person_landmarks)-1}_center",
                (int(box_center[0]) + 5, int(box_center[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                person_color,
                1)

            # Draw lines from box center to each landmark
            for landmark in person["landmarks"]:
                cv2.line(
                    annotated_image, 
                    (int(box_center[0]), int(box_center[1])), 
                    landmark["coords"], 
                    person_color, 
                    1, 
                    cv2.LINE_AA)

        # Process cigarette detections
        assigned_cigarettes = set()

        # First pass: Single person overlap
        for cig_idx, cigarette in enumerate(cigarette_boxes):
            cig_center = calculate_center(cigarette["coords"])
            overlapping_persons = []

            for person in person_landmarks:
                x1, y1, x2, y2 = person["coords"]
                if x1 <= cig_center[0] <= x2 and y1 <= cig_center[1] <= y2:
                    overlapping_persons.append(person)

            if len(overlapping_persons) == 1:
                overlapping_persons[0]["pipeline_smoker"] = True
                overlapping_persons[0]["confident_by_single_cigarette"] = True
                assigned_cigarettes.add(cig_idx)
                print(f"Assigned cigarette {cig_idx} to confident smoker: {overlapping_persons[0]['coords']}")


        # Second pass: Multiple overlaps
        for cig_idx, cigarette in enumerate(cigarette_boxes):
            if cig_idx in assigned_cigarettes:
                continue

            cig_center = calculate_center(cigarette["coords"])
            overlapping_persons = []

            for person in person_landmarks:
                x1, y1, x2, y2 = person["coords"]
                if x1 <= cig_center[0] <= x2 and y1 <= cig_center[1] <= y2:
                    if not person.get("confident_by_single_cigarette", False):
                        overlapping_persons.append(person)

            if overlapping_persons:
                min_distance = float('inf')
                closest_person = None

                for person in overlapping_persons:
                    for landmark in person["landmarks"]:
                        landmark_coords = landmark["coords"]
                        distance = np.sqrt((cig_center[0] - landmark_coords[0])**2 + 
                                        (cig_center[1] - landmark_coords[1])**2)

                        if distance < min_distance:
                            min_distance = distance
                            closest_person = person

                if closest_person:
                    closest_person["pipeline_smoker"] = True
                    assigned_cigarettes.add(cig_idx)

        # Determine final labels
        for person in person_landmarks:
            if person.get("confident_by_single_cigarette", False):
                # If person has a cigarette uniquely overlapping them
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER
            elif person["pipeline_smoker"]== True and person["yolo_smoker"] == True:
                # Both methods agree it's a smoker
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER
            elif person["pipeline_smoker"] == False and person["yolo_smoker"] == False:
                # Both methods agree it's not a smoker
                person["final_label"] = SmokerLabel.NON_SMOKER
            elif person["pipeline_smoker"] != person["yolo_smoker"]:
                # Either pipeline or YOLO detected smoking
                person["final_label"] = SmokerLabel.POSSIBLE_SMOKER
            else:
                # Neither method detected smoking
                person["final_label"] = SmokerLabel.NON_SMOKER

        # Draw annotations
        for person in person_landmarks:
            label = person["final_label"]
            if label == SmokerLabel.CONFIDENT_SMOKER:
                color = (255, 0, 0)  # Red
            elif label == SmokerLabel.POSSIBLE_SMOKER:
                color = (0, 0, 255)  # Blue
            else:  # NON_SMOKER
                color = (0, 255, 0)  # Green
                
            x1, y1, x2, y2 = person["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(annotated_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 4)

        # Draw cigarettes
        for cigarette in cigarette_boxes:
            x1, y1, x2, y2 = cigarette["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 204, 204), 4)  # Light blue
            cv2.putText(annotated_image, "cigarette", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 204, 204), 4)

        # Save or return the annotated image
        if save_output:
            output_path = os.path.join(os.path.dirname(image_path), 'output_' + os.path.basename(image_path))
            Image.fromarray(annotated_image).save(output_path)
            print(f"Saved annotated image to: {output_path}")
            return annotated_image, results
        else:
            return annotated_image, results

    except Exception as e:
        print(f"Error in detect_smoking: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main entry point"""
    try:
        # Load models
        object_model, landmark_model = load_models()
        
        # Get input image path
        image_path = "test.png"
        # Process image
        annotated_image, results = detect_smoking(image_path, object_model, landmark_model)
        
        # Display results
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
