import tensorflow as tf
import tensorflow_hub as hub
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import traceback
import gc
import collections

# Load custom object detection model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_model_SD_large_v2_noiseFree.pt')

class SmokerLabel:
    CONFIDENT_SMOKER = "confident_smoker"
    POSSIBLE_SMOKER = "possible_smoker"
    NON_SMOKER = "nonSmoker"
    CIGARETTE = "cigarette"

# Create temp directory in the project folder
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def get_temp_path():
    """Generate a temporary file path in our custom temp directory"""
    temp_name = f"frame_{time.time_ns()}.jpg"
    return os.path.join(TEMP_DIR, temp_name)

def cleanup_temp_files():
    """Clean up old temporary files"""
    if os.path.exists(TEMP_DIR):
        for file in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, file))
            except Exception:
                pass

# Global cache for models
_model_cache = {
    'object_model': None,
    'landmark_model': None,
    'initialized': False
}
# Global reusable arrays and objects
_reusable_cache = {
    'person_colors': [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan 
    ],
    'persons_boxes': [],
    'cigarette_boxes': [],
    'final_persons': [],
    'final_cigarettes': [],
    'person_landmarks': [],
    'annotated_image': None
}
def load_models():
    """Load both YOLO and MoveNet models"""
    global _model_cache

    # If models are already loaded, return them from cache
    if _model_cache['initialized']:
        return _model_cache['object_model'], _model_cache['landmark_model']
    
    try:
        # Load models only if they haven't been loaded before
        _model_cache['object_model'] = YOLO(MODEL_PATH)
        _model_cache['landmark_model'] = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
        _model_cache['initialized'] = True
        return _model_cache['object_model'], _model_cache['landmark_model']
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
    # print("Shift flag:", shift_flag)

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
    """Process landmarks for a pose

    MoveNet keypoints order:
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    processed_landmarks = []
    for i in range(17):  
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

def has_overlapping_persons(current_person, all_persons):
    """Check if a person has any overlapping persons"""
    current_box = current_person["coords"]
    for other_person in all_persons:
        if other_person["coords"] != current_person["coords"]:  # Skip self
            if calculate_iou(current_box, other_person["coords"]) > 0:  # Using existing calculate_iou function
                return True
    return False

def calculate_helper_weight(person, cigarette, yolo_weight=0.6, landmark_weight=0.4):
    """Calculate helper weight for person-cigarette pair"""
    # Base YOLO weight
    weight = 0
    if person["yolo_smoker"]:
        weight += yolo_weight * person["confidence"]
    
    # Landmark weight
    min_landmark_distance = float('inf')
    cigarette_center = calculate_center(cigarette["coords"])
    
    # Check only landmarks 0 (nose), 9 (left wrist), 10 (right wrist)
    for landmark in person["landmarks"]:
        if landmark["id"] in [0, 9, 10]:
            distance = np.sqrt((cigarette_center[0] - landmark["coords"][0])**2 + 
                             (cigarette_center[1] - landmark["coords"][1])**2)
            min_landmark_distance = min(min_landmark_distance, distance)
    
    if min_landmark_distance != float('inf'):
        # Convert distance to a 0-1 scale (closer = higher weight)
        landmark_score = 1 / (1 + min_landmark_distance/100)  # /100 to normalize distance
        weight += landmark_weight * landmark_score
    
    return weight
def run_yolo_detection(model, image):
    """Run YOLO detection in a separate thread"""
    # Ensure image is in the correct format
    if isinstance(image, np.ndarray):
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = Image.fromarray(image)
    return model(image)

def run_movenet_detection(image_path, model):
    """Run MoveNet detection in a separate thread"""
    return run_movenet(image_path, model)

def convert_cv2_to_pil(cv2_frame):
    """
    Convert a cv2 frame to PIL Image format that YOLO expects.
    Ensures consistent color format and channel ordering.
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image while preserving the color information
    pil_image = Image.fromarray(rgb_frame)

    return pil_image

def detect_smoking(image_input, object_model, landmark_model):
    """Main function to detect smoking in an image"""
    temp_path = None  # Initialize temp_path at the start
    try:

        # Use cached models if not provided
        if object_model is None or landmark_model is None:
            object_model, landmark_model = load_models()

        # Handle different input types
        if isinstance(image_input, str):
            # It's a file path
            image = Image.open(image_input)
            image_np = np.array(image)
        elif isinstance(image_input, Image.Image):
            # It's already a PIL Image
            image = image_input
            image_np = np.array(image)
        else:
            raise ValueError("image_input must be either a file path or a PIL Image object")


        # Clear previous arrays for reuse
        _reusable_cache['persons_boxes'].clear()
        _reusable_cache['cigarette_boxes'].clear()
        _reusable_cache['final_persons'].clear()
        _reusable_cache['final_cigarettes'].clear()
        _reusable_cache['person_landmarks'].clear()
        
        # Reuse the annotated image array
        if _reusable_cache['annotated_image'] is None or _reusable_cache['annotated_image'].shape != image_np.shape:
            _reusable_cache['annotated_image'] = image_np.copy()
        else:
            _reusable_cache['annotated_image'][:] = image_np
            
        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            # For webcam frames, we already have the image in memory
            yolo_future = executor.submit(run_yolo_detection, object_model, image)
            
            # For pose estimation, create a temporary file if needed
            if isinstance(image_input, str):
                movenet_future = executor.submit(run_movenet_detection, image_input, landmark_model)
            else:
                temp_path = get_temp_path()  # Use our custom temp path
                image.save(temp_path)
                movenet_future = executor.submit(run_movenet_detection, temp_path, landmark_model)
            
            results = yolo_future.result()
            keypoints, (scale_x, scale_y), (pad_x, pad_y), orig_height, orig_width, shift_flag = movenet_future.result()

        # Process YOLO detections in parallel
        def process_detection(box):
            class_id = int(box.cls)
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if class_id == 2:  # Smoker class
                return {
                    "type": "person",
                    "data": {
                        "coords": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": confidence,
                        "yolo_smoker": True,
                        "pipeline_smoker": False,
                        "final_label": None,
                        "overlapping_cigarette": None,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": []
                    }
                }
            elif class_id == 1:  # Non-smoker class
                return {
                    "type": "person",
                    "data": {
                        "coords": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": confidence,
                        "yolo_smoker": False,
                        "pipeline_smoker": False,
                        "final_label": None,
                        "overlapping_cigarette": None,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": []
                    }
                }
            elif class_id == 0:  # Cigarette class
                return {
                    "type": "cigarette",
                    "data": {
                        "coords": (x1, y1, x2, y2),
                        "confidence": confidence
                    }
                }
            return None

        # Process detections in parallel and reuse arrays
        with ThreadPoolExecutor(max_workers=4) as executor:
            detection_futures = [executor.submit(process_detection, box) for box in results[0].boxes]
            for future in detection_futures:
                result = future.result()
                if result:
                    if result["type"] == "person":
                        _reusable_cache['persons_boxes'].append(result["data"])
                    else:
                        _reusable_cache['cigarette_boxes'].append(result["data"])

        # Apply NMS and process poses
        _reusable_cache['persons_boxes'] = non_max_suppression(_reusable_cache['persons_boxes'])
        _reusable_cache['cigarette_boxes'] = non_max_suppression(_reusable_cache['cigarette_boxes'])
        
        processed_poses = preprocess_all_poses(keypoints[0], scale_x, scale_y, pad_x, pad_y, shift_flag) #array of landmarks for each person
        
        # Process each person
        annotated_image = _reusable_cache['annotated_image']

        for person in _reusable_cache['persons_boxes']:
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
                best_person = find_best_person_for_landmarks(best_pose["landmarks"], _reusable_cache['persons_boxes'])
                
                if best_person is None or best_person["coords"] == person["coords"]:
                    # Only add landmarks 0 (nose), 9 (left wrist), and 10 (right wrist)
                    selected_landmarks = [lm for lm in best_pose["landmarks"] if lm["id"] in [0, 9, 10]]
                    person["landmarks"].extend(selected_landmarks)
                    best_pose["assigned"] = True

            _reusable_cache['person_landmarks'].append(person)

            person_color = _reusable_cache['person_colors'][(len(_reusable_cache['person_landmarks']) - 1) % len(_reusable_cache['person_colors'])]
            
            # Get scale factor for annotations
            scale = get_scale_factor(annotated_image)
        

            # Plot landmarks for this person
            for landmark in person["landmarks"]:
                # Draw landmark point with scaled radius
                point_radius = max(1, int(5 * scale))
                cv2.circle(annotated_image, landmark["coords"], point_radius, person_color, -1)
                
                # Add label with landmark ID and confidence using scaled font
                label_text = f"P{len(_reusable_cache['person_landmarks'])-1}L{landmark['id']} ({landmark['confidence']:.2f})"
                font_scale = 0.5 * scale
                font_thickness = max(1, int(scale))
                cv2.putText(
                    annotated_image,
                    label_text,
                    (landmark["coords"][0] + 5, landmark["coords"][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    person_color,
                    font_thickness
                )

            # Draw person's bounding box center
            x1, y1, x2, y2 = person["coords"]
            box_center = calculate_center((x1, y1, x2, y2))
            center_radius = max(1, int(8 * scale))  # Scale the 8 pixel radius
            cv2.circle(annotated_image, (int(box_center[0]), int(box_center[1])), center_radius, person_color, -1)
            #cv2.circle(annotated_image, (int(box_center[0]), int(box_center[1])), 8, person_color, -1)
        
            # Label the box center with scaled font
            font_scale = 0.5 * scale
            font_thickness = max(1, int(scale))
            cv2.putText(
                annotated_image,
                f"P{len(_reusable_cache['person_landmarks'])-1}_center",
                (int(box_center[0]) + 5, int(box_center[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                person_color,
                font_thickness)

            # Draw lines from box center to each landmark with scaled thickness
            line_thickness = max(1, int(scale))
            for landmark in person["landmarks"]:
                cv2.line(
                    annotated_image, 
                    (int(box_center[0]), int(box_center[1])), 
                    landmark["coords"], 
                    person_color, 
                    line_thickness, 
                    cv2.LINE_AA)

        # Process persons and cigarettes
        persons_to_remove = set()
        cigarettes_to_remove = set()

        for person_idx, person in enumerate(_reusable_cache['persons_boxes']):
            # Check if person has overlapping persons
            has_overlapping_person = has_overlapping_persons(person, _reusable_cache['persons_boxes'])
            
            # Find all overlapping cigarettes for this person
            overlapping_cigarettes = []
            for cig_idx, cigarette in enumerate(_reusable_cache['cigarette_boxes']):
                if calculate_iou(person["coords"], cigarette["coords"]) > 0:
                    overlapping_cigarettes.append((cig_idx, cigarette))
            
            # Case 1: Person has no cigarettes
            if len(overlapping_cigarettes) == 0:
                # Add person to final_persons as non-smoker
                final_person = {
                    "coords": person["coords"],
                    "class_id": person["class_id"],
                    "confidence": person["confidence"],
                    "yolo_smoker": person["yolo_smoker"],
                    "pipeline_smoker": False,
                    "final_label": None,
                    "overlapping_cigarette": None,
                    "overlapping_person": None,
                    "helper_weight": None,
                    "landmarks": person["landmarks"],
                    "person_id": len(_reusable_cache['final_persons'])  # Assign unique ID
                }
                _reusable_cache['final_persons'].append(final_person)  
                persons_to_remove.add(person_idx)
                continue
            
            # Only process further if person has no overlapping persons
            if not has_overlapping_person:
                # Case 2: Person has multiple cigarettes
                if len(overlapping_cigarettes) > 1:
                    # Sort by confidence and keep the best one
                    overlapping_cigarettes.sort(key=lambda x: x[1]["confidence"], reverse=True)
                    best_cigarette_idx, best_cigarette = overlapping_cigarettes[0]
                    
                    # Create final person entry
                    final_person = {
                        "coords": person["coords"],
                        "class_id": person["class_id"],
                        "confidence": person["confidence"],
                        "yolo_smoker": person["yolo_smoker"],
                        "pipeline_smoker": True,
                        "final_label": None,
                        "overlapping_cigarette": best_cigarette,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": person["landmarks"],
                        "person_id": len(_reusable_cache['final_persons'])  # Assign unique ID
                    }
                    _reusable_cache['final_persons'].append(final_person)

                    # Add cigarette to final_cigarettes
                    _reusable_cache['final_cigarettes'].append({
                        "coords": best_cigarette["coords"],
                        "confidence": best_cigarette["confidence"],
                        "person_id": final_person["person_id"],  # Link to person
                        "cigarette_id": len(_reusable_cache['final_cigarettes'])    # Unique cigarette ID
                    })

                    # Mark all other cigarettes for removal
                    for cig_idx, _ in overlapping_cigarettes[1:]:
                        cigarettes_to_remove.add(cig_idx)
                    
                    persons_to_remove.add(person_idx)

                # Case 3: Person has exactly one cigarette
                elif len(overlapping_cigarettes) == 1:
                    cig_idx, cigarette = overlapping_cigarettes[0]
                    
                    # Create final person entry
                    final_person = {
                        "coords": person["coords"],
                        "class_id": person["class_id"],
                        "confidence": person["confidence"],
                        "yolo_smoker": person["yolo_smoker"],
                        "pipeline_smoker": True,
                        "final_label": None,
                        "overlapping_cigarette": cigarette,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": [],
                        "person_id": len(_reusable_cache['final_persons'])  # Assign unique ID
                    }
                    _reusable_cache['final_persons'].append(final_person)
                    
                    # Add cigarette to final_cigarettes
                    _reusable_cache['final_cigarettes'].append({
                        "coords": cigarette["coords"],
                        "confidence": cigarette["confidence"],
                        "person_id": final_person["person_id"],  # Link to person
                        "cigarette_id": len(_reusable_cache['final_cigarettes'])    # Unique cigarette ID
                    })
                    
                    cigarettes_to_remove.add(cig_idx)
                    persons_to_remove.add(person_idx)

        # Clean up: Remove processed items
        _reusable_cache['persons_boxes'] = [person for idx, person in enumerate(_reusable_cache['persons_boxes']) if idx not in persons_to_remove]
        _reusable_cache['cigarette_boxes'] = [cig for idx, cig in enumerate(_reusable_cache['cigarette_boxes']) if idx not in cigarettes_to_remove]

        overlapping_groups = []  # Will store groups of overlapping persons
        current_group = []

        # First, group overlapping persons
        processed_persons = set()
        for i, person1 in enumerate(_reusable_cache['persons_boxes']):
            if i not in processed_persons:
                current_group = [person1]
                processed_persons.add(i)
                
                # Keep checking for new overlaps until no more found
                idx = 0
                while idx < len(current_group):
                    current_person = current_group[idx]
                    
                    # Check against all unprocessed persons
                    for j, person2 in enumerate(_reusable_cache['persons_boxes']):
                        if j not in processed_persons:
                            iou = calculate_iou(current_person["coords"], person2["coords"])
                            if iou > 0:
                                current_group.append(person2)
                                processed_persons.add(j)
                    idx += 1
                
                if len(current_group) > 1:
                    overlapping_groups.append(current_group)

        # Process each overlapping group
        persons_to_remove = set()
        cigarettes_to_remove = set()

        # Process each overlapping group
        for group in overlapping_groups:
            # Initialize person_cigarette_map for the group FIRST
            person_cigarette_map = {}
            for person in group:
                person_cigarette_map[id(person)] = []
                
            # Find cigarettes overlapping with each person
            for cigarette_idx, cigarette in enumerate(_reusable_cache['cigarette_boxes']):
                for person in group:
                    if calculate_iou(person["coords"], cigarette["coords"]) > 0:
                        person_cigarette_map[id(person)].append({
                            "cigarette": cigarette,
                            "cigarette_idx": cigarette_idx
                        })

            # 1. First collect all cigarettes overlapping with this group
            group_cigarettes = []
            for person in group:
                for cig_data in person_cigarette_map[id(person)]:
                    if cig_data["cigarette_idx"] not in [c["cigarette_idx"] for c in group_cigarettes]:
                        group_cigarettes.append({
                            "cigarette": next(cig for cig in _reusable_cache['cigarette_boxes'] if _reusable_cache['cigarette_boxes'].index(cig) == cig_data["cigarette_idx"]),
                            "cigarette_idx": cig_data["cigarette_idx"]
                        })
                
            # 2. Calculate weights for all person-cigarette pairs
            all_pairs = []
            for person in group:
                for cig_data in group_cigarettes:
                    weight = calculate_helper_weight(person, cig_data["cigarette"])
                    all_pairs.append({
                        "person": person,
                        "cigarette": cig_data["cigarette"],
                        "cigarette_idx": cig_data["cigarette_idx"],
                        "weight": weight
                    })

            # 3. Sort all pairs by weight to get best matches
            all_pairs.sort(key=lambda x: x["weight"], reverse=True)

            # 4. Assign cigarettes to persons (ensuring one cigarette per person)
            assigned_persons = set()
            assigned_cigarettes = set()
                
            for pair in all_pairs:
                person_id = id(pair["person"])
                cig_idx = pair["cigarette_idx"]
                    
                # Skip if either person or cigarette already assigned
                if person_id in assigned_persons or cig_idx in assigned_cigarettes:
                    continue
                    
                # Check if cigarette already in final_cigarettes
                cigarette_exists = any(
                    existing_cig["coords"] == pair["cigarette"]["coords"]
                    for existing_cig in _reusable_cache['final_cigarettes']
                )
                if not cigarette_exists:
                    # Add person as smoker
                    person_exists = any(
                        p["coords"] == pair["person"]["coords"]
                        for p in _reusable_cache['final_persons']
                    )
                    if not person_exists:
                        final_person = {
                            "coords": pair["person"]["coords"],
                            "class_id": pair["person"]["class_id"],
                            "confidence": pair["person"]["confidence"],
                            "yolo_smoker": pair["person"]["yolo_smoker"],
                            "pipeline_smoker": True,
                            "final_label": None,
                            "overlapping_cigarette": pair["cigarette"]["coords"],
                            "overlapping_person": True,
                            "helper_weight": pair["weight"],
                            "landmarks": pair["person"]["landmarks"],
                            "person_id": len(_reusable_cache['final_persons'])
                        }
                        _reusable_cache['final_persons'].append(final_person)
                            
                        # Add cigarette
                        _reusable_cache['final_cigarettes'].append({
                            "coords": pair["cigarette"]["coords"],
                            "confidence": pair["cigarette"]["confidence"],
                            "person_id": final_person["person_id"],
                            "cigarette_id": len(_reusable_cache['final_cigarettes'])
                        })
                            
                        assigned_persons.add(person_id)
                        assigned_cigarettes.add(cig_idx)
                        cigarettes_to_remove.add(cig_idx)
                
            # 5. Process remaining unassigned persons as non-smokers
            for person in group:
                if id(person) not in assigned_persons:
                    person_exists = any(
                        p["coords"] == person["coords"] for p in _reusable_cache['final_persons']
                    )
                        
                    if not person_exists:
                        final_person = {
                            "coords": person["coords"],
                            "class_id": person["class_id"],
                            "confidence": person["confidence"],
                            "yolo_smoker": person["yolo_smoker"],
                            "pipeline_smoker": False,
                            "final_label": None,
                            "overlapping_cigarette": None,
                            "overlapping_person": True,
                            "helper_weight": None,
                            "landmarks": person["landmarks"],
                            "person_id": len(_reusable_cache['final_persons'])
                        }
                        _reusable_cache['final_persons'].append(final_person)
                        
                persons_to_remove.add(id(person))
                

        # Final cleanup
        _reusable_cache['persons_boxes'] = [person for person in _reusable_cache['persons_boxes'] if id(person) not in persons_to_remove]
        _reusable_cache['cigarette_boxes'] = [cig for idx, cig in enumerate(_reusable_cache['cigarette_boxes']) if idx not in cigarettes_to_remove]
        
        # Determine final labels for all persons in final_persons
        for person in _reusable_cache['final_persons']:
            # Case 1: Person has a unique cigarette (processed first in our overlapping logic)
            if (person["pipeline_smoker"] and 
                person["overlapping_cigarette"] and 
                not person["overlapping_person"]):
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER
            
            # Case 2: Person has a cigarette (either unique or shared) and YOLO agrees
            elif person["pipeline_smoker"] and person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER
            
            # Case 3: Person has no cigarette and YOLO agrees they're not smoking
            elif not person["pipeline_smoker"] and not person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.NON_SMOKER

            # Case 4: Pipeline and YOLO disagree
            elif person["pipeline_smoker"] != person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.POSSIBLE_SMOKER

            # Case 5: Default case (should rarely happen)
            else:
                person["final_label"] = SmokerLabel.NON_SMOKER

        # Draw annotations with scaling
        scale = get_scale_factor(annotated_image)
        box_thickness = max(2, int(2 * scale))
        font_scale = 1.3 * scale
        font_thickness = max(2, int(3 * scale))

        # Draw annotations
        for person in _reusable_cache['final_persons']:
            label = person["final_label"]
            if label == SmokerLabel.CONFIDENT_SMOKER:
                color = (255, 0, 0)  # Red
            elif label == SmokerLabel.POSSIBLE_SMOKER:
                color = (0, 0, 255)  # Blue
            else:  # NON_SMOKER
                color = (0, 255, 0)  # Green
                
            x1, y1, x2, y2 = person["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
             # Calculate text position with scaled offset
            text_offset = max(10, int(10 * scale))
            cv2.putText(annotated_image, label, (x1, y1 - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Draw cigarettes with scaled annotations
        for cigarette in _reusable_cache['final_cigarettes']:
            x1, y1, x2, y2 = cigarette["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 204, 204), box_thickness)
            
            # Calculate text position with scaled offset
            text_offset = max(10, int(10 * scale))
            cv2.putText(annotated_image, "cigarette", (x1, y1 - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 204, 204), font_thickness)

        return annotated_image

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

    finally:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def draw_metrics(frame, fps=None, process_time=None):
    """Draw metrics on frame consistently"""
    try:
        if fps is None and process_time is None:
            return frame
            
        metrics_text = []
        if fps is not None:
            metrics_text.append(f"FPS: {fps:.1f}")
        if process_time is not None:
            metrics_text.append(f"Time: {process_time*1000:.1f}ms")
            
        text = " | ".join(metrics_text)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw black background
        cv2.rectangle(frame, 
                    (5, 5), 
                    (15 + text_w, 15 + text_h + baseline), 
                    (0, 0, 0), 
                    -1)
        # Draw text
        cv2.putText(frame, 
                  text,
                  (10, 20), 
                  font, 
                  font_scale, 
                  (0, 255, 0), 
                  thickness)
                  
        return frame
    except Exception as e:
        print(f"Error drawing metrics: {str(e)}")
        return frame

def get_scale_factor(image):
    """Calculate scale factor based on image size"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:  # PIL Image
        width, height = image.size
        
    # Use 4000x2252 as reference size (your static image size)
    reference_width = 4000
    reference_height = 2252
    
    # Calculate scale factor based on diagonal
    current_diagonal = np.sqrt(width**2 + height**2)
    reference_diagonal = np.sqrt(reference_width**2 + reference_height**2)
    
    return current_diagonal / reference_diagonal

def process_webcam_frame(frame, object_model, landmark_model):
    """Process webcam frame exactly like static image path"""
    try:
        original_size = (frame.shape[1], frame.shape[0])
        
        # Save frame to temp file with original quality
        temp_path = get_temp_path()
        cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        try:
            # Process exactly like a static image path
            annotated_frame = detect_smoking(temp_path, object_model, landmark_model)
            
            # Convert back to BGR for display, maintaining original size
            result = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
            
            # Ensure we're at original size
            if result.shape[:2] != frame.shape[:2]:
                result = cv2.resize(result, original_size, interpolation=cv2.INTER_LINEAR)
            
            return result
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        traceback.print_exc()
        return frame

def main():
    """Main entry point for real-time webcam detection"""
    # Initialize variables outside try block
    cap = None
    frame_times = collections.deque(maxlen=30)
    
    try:
        cleanup_temp_files()
        
        print("Loading models...")
        object_model, landmark_model = load_models()
        print("Models loaded successfully!")
        
        # # Test with static image first
        # print("\nTesting with static image...")
        # test_image_path = "test.jpg"
        
        # start_time = time.time()
        # annotated_image = detect_smoking(test_image_path, object_model, landmark_model)
        
        # plt.figure(figsize=(12, 8))
        # plt.imshow(annotated_image)
        # plt.axis('off')
        # plt.show()
        # plt.close()
        
        # input("\nPress Enter to start webcam detection (or Ctrl+C to exit)...")
        
        # Initialize video capture
        cap = cv2.VideoCapture(1)
        
        # Set optimal camera properties for quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Capture at higher resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Capture at higher resolution
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increased buffer for smoother capture
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
        
        # Create window
        cv2.namedWindow('Smoking Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Smoking Detection', 1280, 720)
        
        print("Starting webcam detection pipeline...")
        
        # Performance tracking variables
        fps_update_interval = 10  # Update FPS more frequently
        frame_count = 0
        start_time = time.time()
        skip_count = 0
        last_gc_time = time.time()
        current_fps = 0
        current_frame_time = 0
        last_frame_time = time.time()
        
        while True:
            loop_start = time.time()
            
            # Calculate instantaneous FPS
            current_fps = 1.0 / (loop_start - last_frame_time) if loop_start > last_frame_time else current_fps
            last_frame_time = loop_start
            
            # Periodic garbage collection
            if time.time() - last_gc_time > 10:
                gc.collect()
                last_gc_time = time.time()
            
            # Skip frames if processing is too slow
            if skip_count > 0:
                ret = cap.grab()  # Just grab frame, don't decode
                skip_count -= 1
                continue
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            try:
                
                # Process frame
                annotated_frame = process_webcam_frame(frame, object_model, landmark_model)
                
                # Update performance metrics
                current_frame_time = time.time() - loop_start
                frame_times.append(current_frame_time)
                
                # Adaptive frame skipping
                if current_frame_time > 1/15:  # Target minimum 15 FPS
                    skip_count = min(2, int(current_frame_time * 15))  # Skip up to 2 frames
                
                frame_count += 1
                
                # Calculate rolling average FPS
                if frame_count % fps_update_interval == 0:
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else current_fps
                
                # Draw metrics on every frame
                annotated_frame = draw_metrics(annotated_frame, current_fps, current_frame_time)
                
                cv2.imshow('Smoking Detection', annotated_frame)
                
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                traceback.print_exc()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
    finally:
        # Clean up
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\nCleaned up resources")
        #gc.collect()  # Final cleanup
        
        if frame_times:
            avg_process_time = sum(frame_times) / len(frame_times)
            print(f"\nWebcam Performance Statistics:")
            print(f"Average processing time per frame: {avg_process_time*1000:.2f}ms")
            print(f"Average FPS: {1/avg_process_time:.1f}")
            print(f"Total frames processed: {frame_count}")
            print(f"Frames skipped: {skip_count}")
            if 'frame' in locals():
                print(f"Final frame size: {frame.shape[1]}x{frame.shape[0]}")

if __name__ == "__main__":
    main()
