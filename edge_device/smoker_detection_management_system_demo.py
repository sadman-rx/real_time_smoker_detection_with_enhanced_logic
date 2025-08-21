import os
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont # Ensure ImageDraw and ImageFont are imported at the top
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import traceback
import gc
import collections
from picamera2 import Picamera2
import onnxruntime as ort
import tflite_runtime.interpreter as tflite
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import threading
import shutil # For deleting directories
import sys # Import sys for potential use (e.g., sys.version)

# Global variables for display
root = None
panel = None

# Model path
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'altf4_v2_smoking_detection_model.onnx')
MOVENET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'movenet_multipose_lightning.tflite')

class SmokerLabel:
   CONFIDENT_SMOKER = "confident_smoker"
   POSSIBLE_SMOKER = "possible_smoker"
   NON_SMOKER = "nonSmoker"
   CIGARETTE = "cigarette"

# Create temp directory in the project folder
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
if not os.path.exists(TEMP_DIR):
   os.makedirs(TEMP_DIR)

# Directory for saving smoker frames
SAVED_FRAMES_BASE_DIR = os.path.join(os.path.dirname(__file__), 'saved_smoker_frames')
if not os.path.exists(SAVED_FRAMES_BASE_DIR):
    os.makedirs(SAVED_FRAMES_BASE_DIR)

# Global queue for inter-thread communication (processed frames for UI)
frame_queue = Queue(maxsize=2) # Limit queue size to prevent lag

# Global variable to control the detection loop
detection_running = False
picam2_instance = None # To hold the Picamera2 instance

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

def load_models():
    """Load both ONNX and TFLite models"""
    global _model_cache
   
    if not _model_cache['initialized']:
        try:
            # Load YOLO ONNX model
            yolo_path = os.path.join(os.path.dirname(__file__), YOLO_MODEL_PATH)
            providers = ['CPUExecutionProvider']  # Use CPU provider for Raspberry Pi
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4  # Use all Pi cores for YOLO
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _model_cache['object_model'] = ort.InferenceSession(yolo_path, providers=providers, sess_options=options)
           
            # Load MoveNet TFLite model
            movenet_path = os.path.join(os.path.dirname(__file__), MOVENET_MODEL_PATH)
            interpreter = tflite.Interpreter(model_path=movenet_path)
            interpreter.allocate_tensors()
           
            # Get input and output details for MoveNet
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            _model_cache['input_details'] = input_details
            _model_cache['output_details'] = output_details
            _model_cache['landmark_model'] = interpreter
           
            _model_cache['initialized'] = True
            print("Models loaded successfully")
            print(f"YOLO Model: {YOLO_MODEL_PATH}")
            print(f"MoveNet Model: {MOVENET_MODEL_PATH}")
            # Warmup runs
            dummy_yolo = np.random.rand(1, 3, 640, 640).astype(np.float32) # YOLO input size
            _model_cache['object_model'].run(None, {'images': dummy_yolo})
            
            input_details = interpreter.get_input_details()[0]
            dummy_movenet = np.random.randint(0, 255, input_details['shape'], dtype=np.uint8)
            interpreter.set_tensor(input_details['index'], dummy_movenet)
            interpreter.invoke()
           
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            traceback.print_exc()
            return None, None
   
    return _model_cache['object_model'], _model_cache['landmark_model']

def preprocess_image(image_path):
    """Preprocess image for movenet model input"""
    try:
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from {image_path}")
        else:
            image = image_path.copy()
       
        # Convert RGBA to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        pad_factor = int((orig_width - orig_height))
        
        # Apply white padding (255 for each channel)
        padded_image = cv2.copyMakeBorder(
            image,
            top= 0,
            bottom=pad_factor,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
            )
        
        # Convert to PIL Image for consistent resizing
        pil_image = Image.fromarray(padded_image)
       
        # Resize using PIL (using old-style constants)
        input_size = 256  # Standard size for MoveNet
        resized_image = pil_image.resize((input_size, input_size), Image.NEAREST)
       
        # Convert back to numpy array and ensure uint8
        input_image = np.array(resized_image, dtype=np.uint8)
       
        # Add batch dimension and ensure uint8
        input_image = np.expand_dims(input_image, axis=0).astype(np.uint8)
       
        # Calculate scale factors
        movenet_image_scale = orig_width / input_size
       
        return input_image, orig_height, orig_width, movenet_image_scale
       
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        traceback.print_exc()
        return None, (1, 1), (0, 0), 0, 0, False

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

def process_pose_landmarks(pose_keypoints, orig_height, orig_width, scale):
    """Process pose keypoints with proper error handling"""
    try:
        if pose_keypoints is None or len(pose_keypoints.shape) < 2:
            print("Invalid pose keypoints shape")
            return []
        processed_poses = []
        input_size = 256  # MoveNet input size
        
        # Handle different MoveNet output shapes
        if len(pose_keypoints.shape) == 4:  # Shape [1, 1, 17, 3]
            pose_keypoints = pose_keypoints.reshape(-1, 51)  # Reshape to [N, 51]
        if len(pose_keypoints.shape) == 2:  # Shape [N, 51]
            num_poses = pose_keypoints.shape[0]
            keypoints_per_pose = 17
            
            for i in range(num_poses):
                pose = pose_keypoints[i]  # Get single pose
                keypoints = []
                
                for j in range(keypoints_per_pose):
                    # Get coordinates in 256x256 space
                    y = float(pose[j * 3]) * input_size  # y-coordinate
                    x = float(pose[j * 3 + 1]) * input_size  # x-coordinate
                    confidence = float(pose[j * 3 + 2])  # confidence score
                    
                    # Scale back to original dimensions
                    x = x * scale
                    y = y * scale
                    
                    # Remove padding effect (clamp y to original height)
                    y = min(y, orig_height - 1)
                    
                    # Ensure within image bounds
                    x = max(0, min(x, orig_width - 1))
                    y = max(0, min(y, orig_height - 1))
                    
                    keypoints.append({
                        "id": j,
                        "coords": (int(x), int(y)),
                        "confidence": confidence
                    })
                
                if keypoints:  # Only add if we have valid keypoints
                    processed_poses.append({
                        "landmarks": keypoints,
                        "assigned": False
                    })
        elif len(pose_keypoints.shape) == 3:  # Shape [1, 6, 56]
            num_poses = pose_keypoints.shape[1]
            keypoints_per_pose = 17
            
            for i in range(num_poses):
                pose = pose_keypoints[0, i]  # Get single pose
                if np.any(pose):  # Check if pose contains any valid points
                    keypoints = []
                    for j in range(keypoints_per_pose):
                        y = float(pose[j * 3]) * input_size
                        x = float(pose[j * 3 + 1]) * input_size
                        confidence = float(pose[j * 3 + 2])
                        
                        # Scale back to original dimensions
                        x = x * scale
                        y = y * scale
                        
                        # Remove padding effect
                        y = min(y, orig_height - 1)
                        
                        # Ensure within image bounds
                        x = max(0, min(x, orig_width - 1))
                        y = max(0, min(y, orig_height - 1))
                        
                        keypoints.append({
                            "id": j,
                            "coords": (int(x), int(y)),
                            "confidence": confidence
                        })
                    
                    if keypoints:
                        processed_poses.append({
                            "landmarks": keypoints,
                            "assigned": False
                        })
        return processed_poses
            
       
    except Exception as e:
        print(f"Error processing landmarks: {str(e)}")
        return []

def analyze_pose(pose_data):
    """Analyze pose data to detect smoking-related gestures"""
    try:
        keypoints = pose_data['keypoints']
       
        # Get relevant keypoint positions
        nose = next((kp for kp in keypoints if kp['id'] == 0), None)
        left_wrist = next((kp for kp in keypoints if kp['id'] == 9), None)
        right_wrist = next((kp for kp in keypoints if kp['id'] == 10), None)
        left_elbow = next((kp for kp in keypoints if kp['id'] == 7), None)
        right_elbow = next((kp for kp in keypoints if kp['id'] == 8), None)
       
        if nose and (left_wrist or right_wrist) and (left_elbow or right_elbow):
            # Check if either hand is near the face
            nose_pos = nose['coords']
            smoking_gesture = False
           
            # Define distance thresholds
            face_threshold = 50  # pixels for hand-to-face distance
            elbow_angle_threshold = 60  # degrees for elbow bend
           
            # Check left arm
            if left_wrist and left_elbow:
                left_dist = np.sqrt(np.sum(np.square(np.array(nose_pos) - np.array(left_wrist['coords']))))
                if left_dist < face_threshold:
                    # Check elbow angle
                    left_shoulder = next((kp for kp in keypoints if kp['id'] == 5), None)
                    if left_shoulder:
                        angle = calculate_angle(
                            left_shoulder['coords'],
                            left_elbow['coords'],
                            left_wrist['coords']
                        )
                        if angle < elbow_angle_threshold:
                            smoking_gesture = True
           
            # Check right arm
            if right_wrist and right_elbow:
                right_dist = np.sqrt(np.sum(np.square(np.array(nose_pos) - np.array(right_wrist['coords']))))
                if right_dist < face_threshold:
                    # Check elbow angle
                    right_shoulder = next((kp for kp in keypoints if kp['id'] == 6), None)
                    if right_shoulder:
                        angle = calculate_angle(
                            right_shoulder['coords'],
                            right_elbow['coords'],
                            right_wrist['coords']
                        )
                        if angle < elbow_angle_threshold:
                            smoking_gesture = True
           
            return smoking_gesture
           
    except Exception as e:
        print(f"Error analyzing pose: {e}")
        traceback.print_exc()
       
    return False

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
   
    ba = a - b
    bc = c - b
   
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
   
    return np.degrees(angle)

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

def run_yolo_detection(model, image_np):
    """Run YOLO detection using ONNX runtime"""
    try:
        target_size = 640
        original_height, original_width = image_np.shape[:2]
        padding_flag = int((original_width - original_height))
        # Apply white padding (255 for each channel)
        padded_image = cv2.copyMakeBorder(
            image_np,
            top= 0,
            bottom=padding_flag,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
            )
        padded_image, compression_factor = safe_resize_image(padded_image, target_size, original_width)
        # Normalize and prepare input tensor
        input_image = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, 0)
        # Get input name and run inference
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_tensor})
        # Process predictions
        predictions = outputs[0][0].transpose()  # Shape: (8400, 7)
       
        # Get confidence scores
        confidence_scores = predictions[:, 4:]
        max_conf = np.max(confidence_scores, axis=1)
        class_ids = np.argmax(confidence_scores, axis=1)
       
        # Filter by confidence threshold
        conf_threshold = 0.25
        valid_detections = max_conf > conf_threshold
       
        boxes = predictions[valid_detections, :4]
        scores = max_conf[valid_detections]
        classes = class_ids[valid_detections]
        def non_max_suppression_yolo(boxes, scores, iou_threshold, class_id):
            """Perform Non-Maximum Suppression"""
            if len(boxes) == 0:
                return np.array([])
               
            # Convert to corners format
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
           
            # Convert to corner coordinates
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
           
            # Compute areas
            areas = (x2 - x1) * (y2 - y1)
           
            # For person class, pre-filter small boxes
            if class_id == 1:  # person class
                area_threshold = np.max(areas) * 0.5  # Keep boxes at least 50% of max area
                valid_indices = areas >= area_threshold
                if np.any(valid_indices):
                    boxes = boxes[valid_indices]
                    scores = scores[valid_indices]
                    areas = areas[valid_indices]
                    x1 = x1[valid_indices]
                    y1 = y1[valid_indices]
                    x2 = x2[valid_indices]
                    y2 = y2[valid_indices]
           
            # Sort by confidence score
            order = scores.argsort()[::-1]
           
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
               
                if order.size == 1:
                    break
                   
                # Compute IoU
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
               
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
               
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= iou_threshold)[0]
                order = order[inds + 1]
               
            return np.array(keep)
        # Apply NMS per class
        final_detections = []
       
        for class_id in np.unique(classes):
            class_mask = classes == class_id
            if not np.any(class_mask):
                continue
               
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
           
            # Apply NMS
            keep_indices = non_max_suppression_yolo(class_boxes, class_scores, 0.45, class_id)
           
            for idx in keep_indices:
                x, y, w, h = class_boxes[idx]
                score = class_scores[idx]
               
                # Convert to corner format
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
               
                # Create detection object
                detection = type('Detection', (), {
                    'cls': np.array([class_id]),
                    'conf': np.array([score]),
                    'xyxy': np.array([[x1, y1, x2, y2]])
                })
                final_detections.append(detection)
       
        # Create results object
        results = type('Results', (), {'boxes': final_detections})()
        return [results]
       
    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        traceback.print_exc()
        return None

def run_movenet(input_tensor, interpreter):
    """Run MoveNet model inference"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
       
        # Check if dynamic shape model
        is_dynamic_shape = input_details[0]['shape_signature'][2] == -1
        if is_dynamic_shape:
            interpreter.resize_tensor_input(
                input_details[0]['index'],
                input_tensor.shape,
                strict=True)
            interpreter.allocate_tensors()
       
        # Set input tensor (already uint8 from preprocessing)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        print(f"Error in MoveNet detection: {e}")
        return None

def get_scale_factor(image):
    """Calculate scale factor based on image size"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:  # PIL Image
        width, height = image.size
    # Use 1920x1080 as reference size (your camera resolution)
    reference_width = 1920
    reference_height = 1080
    
    # Calculate scale factor based on diagonal
    current_diagonal = np.sqrt(width**2 + height**2)
    reference_diagonal = np.sqrt(reference_width**2 + reference_height**2)
    
    return current_diagonal / reference_diagonal

def safe_resize_image(image, target_size, original_width):
    """Safely resize an image using PIL"""  
    try:
        # If image is a numpy array with more than 3 dimensions, remove the batch dimension
        if isinstance(image, np.ndarray) and len(image.shape) > 3:
            image = np.squeeze(image)  # Remove batch dimension
           
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Convert float32 to uint8 if necessary
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
       
        # Resize using PIL (using old-style constants)
        resized = image.resize((target_size, target_size), Image.BILINEAR)
        compression_factor = original_width / target_size       
        # Convert back to numpy array
        return np.array(resized), compression_factor      
       
    except Exception as e:
        print(f"Error in safe_resize_image: {str(e)}")
        traceback.print_exc()
        return None, None

def is_frame_valid(frame):
    """Check if the captured frame is valid"""
    return frame is not None and frame.size > 0

def initialize_camera():
    """Robust camera initialization for Pi Camera v3"""
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            print(f"\nCamera initialization attempt {attempt + 1}/{max_attempts}")
            
            # Create camera instance
            picam2 = Picamera2()
            
            # Simple configuration without tuning file
            config = picam2.create_preview_configuration(
                main={
                    "size": (1920, 1080),  # Reduced resolution for better performance
                    "format": "RGB888"
                },
                controls={
                    "FrameRate": 30,
                    "AwbEnable": True,
                    "AeEnable": True
                },
                queue=False,
                buffer_count=2
            )
            
            # Apply configuration
            picam2.configure(config)
            
            # Start camera with brief delay
            picam2.start()
            time.sleep(2)  # Allow camera to stabilize
            
            print("Camera initialized successfully")
            return picam2
            
        except Exception as e:
            print(f"Camera initialization failed: {str(e)}")
            attempt += 1
            time.sleep(1)  # Wait before retrying
            
            # Clean up if camera was partially initialized
            if 'picam2' in locals():
                try:
                    picam2.close()
                except:
                    pass
    print("Failed to initialize camera after multiple attempts")
    return None

def run_full_detection_pipeline(frame_input, object_model, landmark_model):
    """
    Executes the full smoking detection pipeline on a single frame.
    Combines YOLO and MoveNet inference, post-processing, and annotation.
    """
    try:
        # Ensure frame_input is a numpy array in RGB format for consistent processing
        if isinstance(frame_input, np.ndarray):
            if frame_input.shape[-1] == 4: # RGBA
                image_np = cv2.cvtColor(frame_input, cv2.COLOR_RGBA2RGB)
            elif frame_input.shape[-1] == 3: # BGR (from camera usually)
                image_np = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Unsupported frame_input format. Expected 3 or 4 channels.")
        elif isinstance(frame_input, Image.Image):
            image_np = np.array(frame_input.convert("RGB"), dtype=np.uint8)
        else:
            raise ValueError("frame_input must be a numpy array or PIL Image.")
        # Clear previous arrays for reuse
        _reusable_cache['persons_boxes'].clear()
        _reusable_cache['cigarette_boxes'].clear()
        _reusable_cache['final_persons'].clear()
        _reusable_cache['final_cigarettes'].clear()
        _reusable_cache['person_landmarks'].clear()
    
        if _reusable_cache['annotated_image'] is None or _reusable_cache['annotated_image'].shape != image_np.shape:
            _reusable_cache['annotated_image'] = image_np.copy()
        else:
            _reusable_cache['annotated_image'][:] = image_np # Reset content
        # Run YOLO detection
        yolo_results = run_yolo_detection(object_model, image_np)
        # For pose estimation
        input_movenet_tensor, orig_height, orig_width, movenet_image_scale = preprocess_image(image_np)
        
        # Run pose estimation
        pose_keypoints = run_movenet(input_movenet_tensor, landmark_model)
        # Process YOLO detections
        if yolo_results and yolo_results[0] and yolo_results[0].boxes:
            for box in yolo_results[0].boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                scale_yolo = image_np.shape[1] / 640 # YOLO was trained on 640x640
                x1, y1, x2, y2 = int(x1 * scale_yolo), int(y1 * scale_yolo), int(x2 * scale_yolo), int(y2 * scale_yolo)
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, image_np.shape[1] - 1))
                y1 = max(0, min(y1, image_np.shape[0] - 1))
                x2 = max(0, min(x2, image_np.shape[1] - 1))
                y2 = max(0, min(y2, image_np.shape[0] - 1))
               
                if class_id == 2: # Smoker class
                    _reusable_cache['persons_boxes'].append({"coords": (x1, y1, x2, y2), "class_id": class_id, "confidence": confidence, "yolo_smoker": True, "pipeline_smoker": False, "final_label": None, "overlapping_cigarette": None, "overlapping_person": None, "helper_weight": None, "landmarks": []})
                elif class_id == 1: # Non-smoker class
                    _reusable_cache['persons_boxes'].append({"coords": (x1, y1, x2, y2), "class_id": class_id, "confidence": confidence, "yolo_smoker": False, "pipeline_smoker": False, "final_label": None, "overlapping_cigarette": None, "overlapping_person": None, "helper_weight": None, "landmarks": []})
                elif class_id == 0: # Cigarette class
                    _reusable_cache['cigarette_boxes'].append({"coords": (x1, y1, x2, y2), "confidence": confidence})
        _reusable_cache['persons_boxes'] = non_max_suppression(_reusable_cache['persons_boxes'])
        _reusable_cache['cigarette_boxes'] = non_max_suppression(_reusable_cache['cigarette_boxes'])
        processed_poses = []
        if pose_keypoints is not None:
            if len(pose_keypoints.shape) == 4:
                poses = pose_keypoints.reshape(-1, 51)
            else:
                poses = pose_keypoints
            processed_poses = process_pose_landmarks(poses, orig_height, orig_width, movenet_image_scale)
        
        annotated_image = _reusable_cache['annotated_image']
        for person in _reusable_cache['persons_boxes']:
            x1, y1, x2, y2 = person["coords"]
            person_center = calculate_center((x1, y1, x2, y2))
            person["landmarks"] = []
            best_pose = None
            min_distance = float('inf')
            if processed_poses:
                for pose in processed_poses:
                    if not pose.get("assigned", False):
                        valid_landmarks = []
                        for landmark in pose["landmarks"]:
                            if is_point_in_box(landmark["coords"], (x1, y1, x2, y2)):
                                valid_landmarks.append(landmark)
                           
                        if valid_landmarks:
                            avg_x = sum(lm["coords"][0] for lm in valid_landmarks) / len(valid_landmarks)
                            avg_y = sum(lm["coords"][1] for lm in valid_landmarks) / len(valid_landmarks)
                            distance = np.sqrt((person_center[0] - avg_x)**2 + (person_center[1] - avg_y)**2)
                               
                            if distance < min_distance:
                                min_distance = distance
                                best_pose = pose
                if best_pose is not None:
                    best_person = find_best_person_for_landmarks(best_pose["landmarks"], _reusable_cache['persons_boxes'])
                       
                    if best_person is None or best_person["coords"] == person["coords"]:
                        selected_landmarks = [lm for lm in best_pose["landmarks"] if lm["id"] in [0, 9, 10]]
                        person["landmarks"].extend(selected_landmarks)
                        best_pose["assigned"] = True
            _reusable_cache['person_landmarks'].append(person)
        
        persons_to_remove = set()
        cigarettes_to_remove = set()
        for person_idx, person in enumerate(_reusable_cache['persons_boxes']):
            has_overlapping_person = has_overlapping_persons(person, _reusable_cache['persons_boxes'])
            overlapping_cigarettes = []
            for cig_idx, cigarette in enumerate(_reusable_cache['cigarette_boxes']):
                if calculate_iou(person["coords"], cigarette["coords"]) > 0:
                    overlapping_cigarettes.append((cig_idx, cigarette))
       
            if len(overlapping_cigarettes) == 0:
                final_person = {
                    "coords": person["coords"], "class_id": person["class_id"], "confidence": person["confidence"],
                    "yolo_smoker": person["yolo_smoker"], "pipeline_smoker": False, "final_label": None,
                    "overlapping_cigarette": None, "overlapping_person": None, "helper_weight": None,
                    "landmarks": person["landmarks"], "person_id": len(_reusable_cache['final_persons'])
                }
                _reusable_cache['final_persons'].append(final_person)
                persons_to_remove.add(person_idx)
                continue
       
            if not has_overlapping_person:
                if len(overlapping_cigarettes) > 1:
                    overlapping_cigarettes.sort(key=lambda x: x[1]["confidence"], reverse=True)
                    best_cigarette_idx, best_cigarette = overlapping_cigarettes[0]
               
                    final_person = {
                        "coords": person["coords"], "class_id": person["class_id"], "confidence": person["confidence"],
                        "yolo_smoker": person["yolo_smoker"], "pipeline_smoker": True, "final_label": None,
                        "overlapping_cigarette": best_cigarette, "overlapping_person": None, "helper_weight": None,
                        "landmarks": person["landmarks"], "person_id": len(_reusable_cache['final_persons'])
                    }
                    _reusable_cache['final_persons'].append(final_person)
                    _reusable_cache['final_cigarettes'].append({
                        "coords": best_cigarette["coords"], "confidence": best_cigarette["confidence"],
                        "person_id": final_person["person_id"], "cigarette_id": len(_reusable_cache['final_cigarettes'])
                    })
                    for cig_idx, _ in overlapping_cigarettes[1:]:
                        cigarettes_to_remove.add(cig_idx)
               
                    persons_to_remove.add(person_idx)
                elif len(overlapping_cigarettes) == 1:
                    cig_idx, cigarette = overlapping_cigarettes[0]
               
                    final_person = {
                        "coords": person["coords"], "class_id": person["class_id"], "confidence": person["confidence"],
                        "yolo_smoker": person["yolo_smoker"], "pipeline_smoker": True, "final_label": None,
                        "overlapping_cigarette": cigarette, "overlapping_person": None, "helper_weight": None,
                        "landmarks": [], "person_id": len(_reusable_cache['final_persons'])
                    }
                    _reusable_cache['final_persons'].append(final_person)
               
                    _reusable_cache['final_cigarettes'].append({
                        "coords": cigarette["coords"], "confidence": cigarette["confidence"],
                        "person_id": final_person["person_id"], "cigarette_id": len(_reusable_cache['final_cigarettes'])
                    })
               
                    cigarettes_to_remove.add(cig_idx)
                    persons_to_remove.add(person_idx)
        _reusable_cache['persons_boxes'] = [person for idx, person in enumerate(_reusable_cache['persons_boxes']) if idx not in persons_to_remove]
        _reusable_cache['cigarette_boxes'] = [cig for idx, cig in enumerate(_reusable_cache['cigarette_boxes']) if idx not in cigarettes_to_remove]
   
        overlapping_groups = []
        processed_persons_in_group = set()
        for i, person1 in enumerate(_reusable_cache['persons_boxes']):
            if i not in processed_persons_in_group:
                current_group = [person1]
                processed_persons_in_group.add(i)
           
                idx = 0
                while idx < len(current_group):
                    current_person = current_group[idx]
               
                    for j, person2 in enumerate(_reusable_cache['persons_boxes']):
                        if j not in processed_persons_in_group:
                            iou = calculate_iou(current_person["coords"], person2["coords"])
                            if iou > 0:
                                current_group.append(person2)
                                processed_persons_in_group.add(j)
                    idx += 1
           
                if len(current_group) > 1:
                    overlapping_groups.append(current_group)
        persons_to_remove_final = set()
        cigarettes_to_remove_final = set()
        for group in overlapping_groups:
            person_cigarette_map = {}
            for person in group:
                person_cigarette_map[id(person)] = []
           
            group_cigarettes = []
            for cigarette_idx, cigarette in enumerate(_reusable_cache['cigarette_boxes']):
                for person in group:
                    if calculate_iou(person["coords"], cigarette["coords"]) > 0:
                        person_cigarette_map[id(person)].append({
                            "cigarette": cigarette,
                            "cigarette_idx": cigarette_idx
                        })
            
            for person in group:
                for cig_data in person_cigarette_map[id(person)]:
                    if cig_data["cigarette_idx"] not in [c["cigarette_idx"] for c in group_cigarettes]:
                        group_cigarettes.append({
                            "cigarette": next(cig for cig in _reusable_cache['cigarette_boxes'] if _reusable_cache['cigarette_boxes'].index(cig) == cig_data["cigarette_idx"]),
                            "cigarette_idx": cig_data["cigarette_idx"]
                        })
           
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
            all_pairs.sort(key=lambda x: x["weight"], reverse=True)
            assigned_persons = set()
            assigned_cigarettes = set()
           
            for pair in all_pairs:
                person_id = id(pair["person"])
                cig_idx = pair["cigarette_idx"]
               
                if person_id in assigned_persons or cig_idx in assigned_cigarettes:
                    continue
               
                cigarette_exists = any(
                    existing_cig["coords"] == pair["cigarette"]["coords"]
                    for existing_cig in _reusable_cache['final_cigarettes']
                )
                if not cigarette_exists:
                    person_exists = any(
                        p["coords"] == pair["person"]["coords"]
                        for p in _reusable_cache['final_persons']
                    )
                    if not person_exists:
                        final_person = {
                            "coords": pair["person"]["coords"], "class_id": pair["person"]["class_id"],
                            "confidence": pair["person"]["confidence"], "yolo_smoker": pair["person"]["yolo_smoker"],
                            "pipeline_smoker": True, "final_label": None,
                            "overlapping_cigarette": pair["cigarette"]["coords"], "overlapping_person": True,
                            "helper_weight": pair["weight"], "landmarks": pair["person"]["landmarks"],
                            "person_id": len(_reusable_cache['final_persons'])
                        }
                        _reusable_cache['final_persons'].append(final_person)
                       
                        _reusable_cache['final_cigarettes'].append({
                            "coords": pair["cigarette"]["coords"], "confidence": pair["cigarette"]["confidence"],
                            "person_id": final_person["person_id"], "cigarette_id": len(_reusable_cache['final_cigarettes'])
                        })
                       
                        assigned_persons.add(person_id)
                        assigned_cigarettes.add(cig_idx)
                        cigarettes_to_remove_final.add(cig_idx)
           
            for person in group:
                if id(person) not in assigned_persons:
                    person_exists = any(
                        p["coords"] == person["coords"] for p in _reusable_cache['final_persons']
                    )
                    if not person_exists:
                        final_person = {
                            "coords": person["coords"], "class_id": person["class_id"],
                            "confidence": person["confidence"], "yolo_smoker": person["yolo_smoker"],
                            "pipeline_smoker": False, "final_label": None,
                            "overlapping_cigarette": None, "overlapping_person": True,
                            "helper_weight": None, "landmarks": person["landmarks"],
                            "person_id": len(_reusable_cache['final_persons'])
                        }
                        _reusable_cache['final_persons'].append(final_person)
                   
                    persons_to_remove_final.add(id(person))
                   
        _reusable_cache['persons_boxes'] = [person for person in _reusable_cache['persons_boxes'] if id(person) not in persons_to_remove_final]
        _reusable_cache['cigarette_boxes'] = [cig for idx, cig in enumerate(_reusable_cache['cigarette_boxes']) if idx not in cigarettes_to_remove_final]
   
        has_smoker_detected = False
        for person in _reusable_cache['final_persons']:
            if (person["pipeline_smoker"] and person["overlapping_cigarette"] and not person["overlapping_person"]):
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER       
            elif person["pipeline_smoker"] and person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER
            elif not person["pipeline_smoker"] and not person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.NON_SMOKER
            elif person["pipeline_smoker"] != person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.POSSIBLE_SMOKER
            else:
                person["final_label"] = SmokerLabel.NON_SMOKER
            if person["final_label"] in [SmokerLabel.CONFIDENT_SMOKER, SmokerLabel.POSSIBLE_SMOKER]:
                has_smoker_detected = True
        scale = get_scale_factor(image_np) # Use image_np for scale calculation
        box_thickness = max(1, int(2 * scale))
        font_scale = max(0.3, 1* scale)
        font_thickness = max(1, int(2 * scale))
        for person in _reusable_cache['final_persons']:
            label = person.get("final_label", SmokerLabel.NON_SMOKER)
           
            if label == SmokerLabel.CONFIDENT_SMOKER:
                color = (255, 0, 0)  # Bright Red for confident smoker (BGR)
            elif label == SmokerLabel.POSSIBLE_SMOKER:
                color = (0, 0, 255) # Blue for possible smoker (BGR)
            else:  # NON_SMOKER
                color = (0, 255, 0)  # Green for non-smoker (BGR)
               
            x1, y1, x2, y2 = person["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
            text_offset = max(10, int(10 * scale))
            cv2.putText(annotated_image, label, (x1, y1 - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
        for cigarette in _reusable_cache['final_cigarettes']:
            x1, y1, x2, y2 = cigarette["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), box_thickness)              
            text_offset = max(10, int(10 * scale))
            cv2.putText(annotated_image, "cigarette", (x1, y1 - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)
        metrics = {
            'num_persons': len(_reusable_cache['final_persons']),
            'num_cigarettes': len(_reusable_cache['final_cigarettes']),
            'smokers': [p for p in _reusable_cache['final_persons'] if p.get("final_label") in
                    [SmokerLabel.CONFIDENT_SMOKER, SmokerLabel.POSSIBLE_SMOKER]],
            'has_smoker_detected': has_smoker_detected
        }
              
        return annotated_image, metrics
    except Exception as e:
        print(f"Error in run_full_detection_pipeline: {str(e)}")
        traceback.print_exc()
        return None, None

# List to keep track of saved folder timestamps
saved_folder_timestamps = collections.deque(maxlen=15)

def save_smoker_frames(frame):
    """
    Saves frames with human-readable timestamp format: YYYY-MM-DD_HH-MM-SS[AM/PM]
    """
    # Get current time in new format
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%I-%M-%S%p")  # Example: 2025-07-17_11-35-49PM
    
    folder_path = os.path.join(SAVED_FRAMES_BASE_DIR, timestamp_str)
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        saved_folder_timestamps.append(folder_path)
    
    # Rest of the function remains the same...
    existing_frames = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))],
        key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    
    # Enforce 20-frame limit
    while len(existing_frames) >= 20:
        oldest_frame = existing_frames.pop(0)
        try:
            os.remove(os.path.join(folder_path, oldest_frame))
        except Exception as e:
            print(f"Error deleting old frame {oldest_frame}: {e}")
    
    # Save new frame
    frame_filename = os.path.join(folder_path, 
                                f"frame_{now.strftime('%H%M%S_%f')}.jpg")
    try:
        cv2.imwrite(frame_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    except Exception as e:
        print(f"Error saving frame: {e}")
    
    # Maintain folder limit
    while len(saved_folder_timestamps) > 15:
        oldest_folder = saved_folder_timestamps.popleft()
        if os.path.exists(oldest_folder):
            try:
                shutil.rmtree(oldest_folder)
            except Exception as e:
                print(f"Error deleting folder {oldest_folder}: {e}")
    
    # Maintain only 15 most recent folders (as per your original code)
    while len(saved_folder_timestamps) > 15:
        oldest_folder = saved_folder_timestamps.popleft()
        if os.path.exists(oldest_folder):
            try:
                shutil.rmtree(oldest_folder)
                print(f"Deleted oldest folder: {oldest_folder}")
            except Exception as e:
                print(f"Error deleting folder {oldest_folder}: {e}")

class SmokingDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Smoking Detection System")
        master.geometry("1200x820") # Adjusted window size for better layout to accommodate side panel
        master.resizable(False, False) # Prevent resizing
        master.configure(bg='#1B1C1D')

        self.object_model = None
        self.landmark_model = None
        self.picam2 = None
        self.detection_thread = None
        self.update_frame_id = None # To store the after() job ID

        self.cameras = [] # List to store dummy camera data
        self.camera_feed_labels = {} # To store references to small camera feed labels

        self.create_widgets()
        self.load_models_async() # Load models asynchronously

    def create_widgets(self):
        # --- Dark Mode: Configure ttk styles ---
        style = ttk.Style()
        style.theme_use('clam') # 'clam' is a good base theme for customization
        # General dark theme for Frames
        style.configure('TFrame', background='#3C3C3C') # Darker grey for frames
        # Style for Labels
        style.configure('TLabel', background='#3C3C3C', foreground='#E0E0E0') # Light grey text on dark background
        # Style for Buttons
        style.configure('TButton',
                        font=('Arial', 12),
                        padding=10,
                        background='#555555', # Darker grey for buttons
                        foreground='#FFFFFF', # White text on buttons
                        borderwidth=0, # Remove default border for a flatter look
                        focusthickness=3,
                        focuscolor='none'
                       )
        style.map('TButton',
                  background=[('active', '#666666'), ('disabled', '#444444')], # Hover and disabled states
                  foreground=[('disabled', '#AAAAAA')] # Disabled text color
                 )
        # Style for Scrollbars (in logs window)
        style.configure('Vertical.TScrollbar', background='#555555', troughcolor='#3C3C3C', borderwidth=0)
        style.map('Vertical.TScrollbar',
                  background=[('active', '#777777')]
                 )
        
        # Style for the video frame border
        style.configure('VideoFrame.TFrame', background='#3C3C3C', bordercolor='#555555', relief="groove", borderwidth=2)
        
        # --- Main Layout using Grid ---
        self.master.grid_rowconfigure(1, weight=1) # Row 1 (content area) expands vertically
        self.master.grid_columnconfigure(1, weight=1) # Column 1 (main video) expands horizontally

        # Header Frame (Logo and Title)
        header_frame = ttk.Frame(self.master, padding="10")
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew") # Spans both columns
        header_frame.configure(style='TFrame') # Apply the TFrame style

        try:
            self.logo_image = Image.open(os.path.join(os.path.dirname(__file__), "Team AltF4.png"))
            self.logo_image = self.logo_image.resize((50, 50), Image.LANCZOS)
            self.logo_photo = ImageTk.PhotoImage(self.logo_image)
            self.logo_label = ttk.Label(header_frame, image=self.logo_photo)
            self.logo_label.pack(side="left", padx=10)
            self.logo_label.configure(style='TLabel') # Apply TLabel style to logo label
        except FileNotFoundError:
            print("Logo file 'Team AltF4.png' not found. Skipping logo display.")
            self.logo_label = None
        except Exception as e:
            print(f"Error loading logo: {e}")
            self.logo_label = None

        self.title_label = ttk.Label(header_frame, text="Real-time Smoker Detection", font=("Arial", 24, "bold"))
        self.title_label.pack(side="left", expand=True)
        self.title_label.configure(foreground='#ADD8E6', background='#3C3C3C') # Light blue for title, ensure background matches frame

        # --- Left Panel for Camera Feeds ---
        # Increased width for the left panel to accommodate larger camera feeds
        self.left_panel_frame = ttk.Frame(self.master, padding="10", width=250) # Adjusted width
        self.left_panel_frame.grid(row=1, column=0, sticky="ns", padx=10, pady=10)
        self.left_panel_frame.configure(style='TFrame')
        self.left_panel_frame.grid_propagate(False) # Prevent frame from resizing to content

        self.camera_feeds_container = ttk.Frame(self.left_panel_frame, style='TFrame')
        self.camera_feeds_container.pack(fill="both", expand=True, pady=(0, 10))

        # "More Cameras" button moved to the bottom-left corner
        self.more_cameras_button = ttk.Button(self.left_panel_frame, text="More Cameras", command=self.view_all_cameras_dummy, style='TButton')
        self.more_cameras_button.pack(side="bottom", pady=(0, 10)) # Packed to bottom
        self.more_cameras_button.config(state=tk.ACTIVE) # Initially disabled

        # --- Main Video Feed and Controls Frame ---
        self.main_content_frame = ttk.Frame(self.master, padding="10")
        self.main_content_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)
        self.main_content_frame.configure(style='TFrame')
        self.main_content_frame.grid_rowconfigure(0, weight=1) # Video frame expands
        self.main_content_frame.grid_columnconfigure(0, weight=1) # Video frame expands

        # Video Feed Frame
        video_frame = ttk.Frame(self.main_content_frame, borderwidth=2, relief="groove")
        video_frame.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
        video_frame.configure(style='VideoFrame.TFrame')
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill="both", expand=True)
        self.video_label.configure(background='#1E1E1E') # Very dark background for the video area itself
        
        # Status Label
        self.status_label = ttk.Label(self.main_content_frame, text="Status: Ready", font=("Arial", 12))
        self.status_label.grid(row=1, column=0, pady=5, sticky="ew")
        self.status_label.configure(foreground='#FFFFFF', background='#2B2B2B') # White for status text, background matches master

        # Control Buttons Frame
        button_frame = ttk.Frame(self.main_content_frame, padding="10")
        button_frame.grid(row=2, column=0, pady=10, sticky="ew")
        button_frame.configure(style='TFrame')
        
        # Center buttons in the button_frame and add space for "Add Camera"
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        button_frame.grid_columnconfigure(2, weight=1)
        button_frame.grid_columnconfigure(3, weight=1) # New column for Add Camera button


        self.start_button = ttk.Button(button_frame, text="Start Detection", command=self.start_detection, style='TButton')
        self.start_button.grid(row=0, column=0, padx=10, pady=5)
        self.end_button = ttk.Button(button_frame, text="End Detection", command=self.stop_detection, state=tk.DISABLED, style='TButton')
        self.end_button.grid(row=0, column=1, padx=10, pady=5)
        self.log_button = ttk.Button(button_frame, text="View Logs", command=self.view_logs, style='TButton')
        self.log_button.grid(row=0, column=2, padx=10, pady=5)
        # "Add Camera" button moved here
        self.add_camera_button = ttk.Button(button_frame, text="Add Camera", command=self.add_camera_panel, style='TButton')
        self.add_camera_button.grid(row=0, column=3, padx=10, pady=5)


        # Set up a protocol for when the window is closed
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_models_async(self):
        self.status_label.config(text="Status: Loading models...")
        self.start_button.config(state=tk.DISABLED)
        self.end_button.config(state=tk.DISABLED)
        self.log_button.config(state=tk.DISABLED)
        self.add_camera_button.config(state=tk.DISABLED) # Disable during model loading
        self.more_cameras_button.config(state=tk.DISABLED)

        def _load():
            global _model_cache
            try:
                self.object_model, self.landmark_model = load_models()
                if self.object_model and self.landmark_model:
                    self.master.after(100, lambda: self.status_label.config(text="Status: Models loaded. Ready to start."))
                    self.master.after(100, lambda: self.start_button.config(state=tk.NORMAL))
                    self.master.after(100, lambda: self.log_button.config(state=tk.NORMAL))
                    self.master.after(100, lambda: self.add_camera_button.config(state=tk.NORMAL)) # Enable after loading
                else:
                    self.master.after(100, lambda: self.status_label.config(text="Status: Model loading failed!"))
                    messagebox.showerror("Error", "Failed to load detection models. Check console for details.")
            except Exception as e:
                self.master.after(100, lambda: self.status_label.config(text=f"Status: Error loading models: {e}"))
                messagebox.showerror("Error", f"An error occurred during model loading: {e}")
            finally:
                # Ensure buttons are re-enabled or disabled appropriately even on error
                if not (self.object_model and self.landmark_model):
                    self.master.after(100, lambda: self.add_camera_button.config(state=tk.NORMAL)) # Allow adding cameras even if detection models fail
        threading.Thread(target=_load, daemon=True).start()

    def add_camera_panel(self):
        """Opens a new window to add camera details and saves them to the camera list."""
        # Create the add camera window
        add_camera_window = tk.Toplevel(self.master)
        add_camera_window.title("Add New Camera")
        add_camera_window.geometry("400x350")
        add_camera_window.configure(bg='#2B2B2B')
        add_camera_window.resizable(False, False)

        # Center the new window relative to the main window
        add_camera_window.update_idletasks()
        x = self.master.winfo_x() + (self.master.winfo_width() // 2) - (add_camera_window.winfo_width() // 2)
        y = self.master.winfo_y() + (self.master.winfo_height() // 2) - (add_camera_window.winfo_height() // 2)
        add_camera_window.geometry(f"+{x}+{y}")

        # Create the form frame
        form_frame = ttk.Frame(add_camera_window, padding="20", style='TFrame')
        form_frame.pack(fill="both", expand=True)

        # Configure grid weights for form frame
        form_frame.grid_columnconfigure(0, weight=1)
        form_frame.grid_columnconfigure(1, weight=3)

        # Camera Name
        ttk.Label(form_frame, text="Camera Name:", style='TLabel').grid(
            row=0, column=0, sticky="w", pady=5, padx=5)
        self.camera_name_entry = ttk.Entry(form_frame, width=30)
        self.camera_name_entry.grid(row=0, column=1, sticky="ew", pady=5, padx=5)
        self.camera_name_entry.focus_set()  # Set focus to name field by default

        # Location
        ttk.Label(form_frame, text="Location:", style='TLabel').grid(
            row=1, column=0, sticky="w", pady=5, padx=5)
        self.camera_location_entry = ttk.Entry(form_frame, width=30)
        self.camera_location_entry.grid(row=1, column=1, sticky="ew", pady=5, padx=5)

        # Resolution
        ttk.Label(form_frame, text="Resolution:", style='TLabel').grid(
            row=2, column=0, sticky="w", pady=5, padx=5)
        self.camera_resolution_entry = ttk.Entry(form_frame, width=30)
        self.camera_resolution_entry.grid(row=2, column=1, sticky="ew", pady=5, padx=5)
        self.camera_resolution_entry.insert(0, "1920x1080")  # Default value

        # Other Metadata
        ttk.Label(form_frame, text="Other Metadata:", style='TLabel').grid(
            row=3, column=0, sticky="w", pady=5, padx=5)
        self.camera_metadata_entry = ttk.Entry(form_frame, width=30)
        self.camera_metadata_entry.grid(row=3, column=1, sticky="ew", pady=5, padx=5)

        # Button frame for better layout
        button_frame = ttk.Frame(form_frame, style='TFrame')
        button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))

        # Save button
        save_button = ttk.Button(
            button_frame, 
            text="Save Camera", 
            command=lambda: self.save_camera_data(add_camera_window), 
            style='TButton'
        )
        save_button.pack(side=tk.LEFT, padx=5)

        # Cancel button
        cancel_button = ttk.Button(
            button_frame,
            text="Cancel",
            command=add_camera_window.destroy,
            style='TButton'
        )
        cancel_button.pack(side=tk.LEFT, padx=5)

        # Bind Enter key to save action
        add_camera_window.bind('<Return>', lambda e: self.save_camera_data(add_camera_window))
        
    def save_camera_data(self, window):
        """Saves the dummy camera data and updates the display."""
        camera_name = self.camera_name_entry.get().strip()
        location = self.camera_location_entry.get().strip()
        resolution = self.camera_resolution_entry.get().strip()
        metadata = self.camera_metadata_entry.get().strip()

        if not camera_name:
            messagebox.showwarning("Input Error", "Camera Name cannot be empty!", parent=window)
            return

        # Add dummy camera data
        new_camera = {
            "name": camera_name,
            "location": location,
            "resolution": resolution,
            "metadata": metadata,
            "id": len(self.cameras)  # Simple unique ID
        }
        self.cameras.append(new_camera)
        print(f"Added dummy camera: {new_camera}")
        self.update_camera_feeds_display()
        window.destroy()  # Close the add camera window


    def update_camera_feeds_display(self):
        """Updates the display of small camera feeds on the left panel."""
        # Clear existing feeds
        for widget in self.camera_feeds_container.winfo_children():
            widget.destroy()
        
        # Display up to 4 cameras
        for i, camera in enumerate(self.cameras[:4]):  # Only show first 3 cameras
            camera_frame = ttk.Frame(self.camera_feeds_container, borderwidth=1, relief="solid", style='TFrame')
            camera_frame.pack(pady=5, padx=5, fill="x")

            # Create darker dummy image for the camera feed (RGB: 40,40,40 - very dark gray)
            dummy_image = Image.new('RGB', (220, 120), color=(40, 40, 40))
            d = ImageDraw.Draw(dummy_image)
            
            try:
                # Try to load Arial, fallback to default if not found
                font = ImageFont.truetype("arial.ttf", 18)  # Slightly smaller font for top-left
            except IOError:
                font = ImageFont.load_default()
            
            # Draw camera name in top-left corner with light gray text (RGB: 180,180,180)
            text = f"{camera['name']}"
            d.text((10, 10),  # Positioned 10px from left and top
                text, 
                fill=(180, 180, 180), 
                font=font)
            
            # Add subtle "No Signal" text in the center (even darker gray)
            d.text((70, 50),  # Centered position
                "NO SIGNAL", 
                fill=(80, 80, 80),  # Dark gray
                font=font)
            
            photo = ImageTk.PhotoImage(dummy_image)
            
            # Label to display the dummy camera feed
            camera_label = ttk.Label(camera_frame, image=photo)
            camera_label.image = photo  # Keep a reference
            camera_label.pack(fill="both", expand=True)

        
        
    def view_all_cameras_dummy(self):
        """Dummy function for 'More Cameras' button."""
        messagebox.showinfo("More Cameras", "This feature is not yet implemented. It would show all camera feeds.")

    def start_detection(self):
        global detection_running, picam2_instance
        if detection_running:
            return
        self.status_label.config(text="Status: Initializing camera and starting detection...")
        self.start_button.config(state=tk.DISABLED)
        self.end_button.config(state=tk.NORMAL)
        self.log_button.config(state=tk.DISABLED)
        self.add_camera_button.config(state=tk.DISABLED) # Disable when detection starts
        self.more_cameras_button.config(state=tk.DISABLED)

        def _run_detection_loop():
            global detection_running, picam2_instance
            detection_running = True
            picam2_instance = initialize_camera()
            if not picam2_instance:
                self.master.after(100, lambda: self.status_label.config(text="Status: Camera initialization failed!"))
                self.master.after(100, lambda: self.start_button.config(state=tk.NORMAL))
                self.master.after(100, lambda: self.end_button.config(state=tk.DISABLED))
                self.master.after(100, lambda: self.log_button.config(state=tk.NORMAL))
                self.master.after(100, lambda: self.add_camera_button.config(state=tk.NORMAL)) # Re-enable add camera
                messagebox.showerror("Error", "Failed to initialize camera. Please check connections and permissions.")
                detection_running = False
                return
            self.master.after(100, lambda: self.status_label.config(text="Status: Detection running..."))
            frame_count = 0
            fps_start_time = time.time()
            
            # Use ThreadPoolExecutor for parallel inference
            executor = ThreadPoolExecutor(max_workers=2) # One for YOLO, one for MoveNet
            try:
                while detection_running:
                    loop_start = time.time()
                    
                    frame = picam2_instance.capture_array() # Captures in BGR format
                    
                    if frame is None or not is_frame_valid(frame):
                        print("Invalid frame captured, retrying...")
                        time.sleep(0.01)
                        continue
                    
                    # Run the full detection pipeline
                    processed_frame, metrics = run_full_detection_pipeline(
                        frame, self.object_model, self.landmark_model
                    )
                    
                    if processed_frame is not None:
                        # Put processed frame into queue for UI update
                        try:
                            frame_queue.put_nowait((processed_frame, metrics))
                        except Empty:
                            pass # Queue is full, skip this frame to avoid blocking
                        # Save frames if smoker detected
                        if metrics and metrics.get('has_smoker_detected', False):
                            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                            save_smoker_frames(processed_frame)
                        # Calculate FPS
                        frame_count += 1
                        current_time = time.time()
                        if current_time - fps_start_time >= 1.0:
                            fps = frame_count
                            process_time = (current_time - loop_start)
                            # Update status label with FPS and processing time
                            self.master.after(0, lambda f=fps, pt=process_time: 
                                #self.status_label.config(text=f"Status: Detection running... FPS: {f} | Process Time: {pt:.3f}s"))
                                self.status_label.config(text=f"Status: Detection running... "))

                            frame_count = 0
                            fps_start_time = current_time
                    
                    # Periodic cleanup of old temp files
                    if frame_count % 300 == 0:
                        cleanup_temp_files()
                        gc.collect()
            except Exception as e:
                print(f"Detection loop error: {str(e)}")
                traceback.print_exc()
                self.master.after(100, lambda: self.status_label.config(text=f"Status: Error during detection: {e}"))
                messagebox.showerror("Error", f"An error occurred during detection: {e}")
            finally:
                detection_running = False
                if picam2_instance:
                    picam2_instance.close()
                    picam2_instance = None
                executor.shutdown(wait=True)
                self.master.after(100, lambda: self.status_label.config(text="Status: Detection stopped."))
                self.master.after(100, lambda: self.start_button.config(state=tk.NORMAL))
                self.master.after(100, lambda: self.end_button.config(state=tk.DISABLED))
                self.master.after(100, lambda: self.log_button.config(state=tk.NORMAL))
                self.master.after(100, lambda: self.add_camera_button.config(state=tk.NORMAL)) # Re-enable add camera
                self.master.after(100, lambda: self.update_camera_feeds_display()) # Re-check more cameras button state
                cleanup_temp_files()
        self.detection_thread = threading.Thread(target=_run_detection_loop, daemon=True)
        self.detection_thread.start()
        self.update_video_feed() # Start updating UI from queue

    def update_video_feed(self):
        # This method will be called periodically by master.after()
        try:
            # Check if there's a processed frame in the queue
            processed_frame, metrics = frame_queue.get_nowait() # Non-blocking get
            if processed_frame is not None:
                # Convert OpenCV BGR image to PIL Image (it's already RGB from run_full_detection_pipeline)
                img = Image.fromarray(processed_frame)
                # Get current dimensions of the video_label
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                # Ensure label has a valid size before resizing
                # winfo_width/height can return 1 before the widget is fully rendered
                if label_width == 1 or label_height == 1:
                    # If not ready, reschedule and return
                    if self.update_frame_id:
                        self.master.after_cancel(self.update_frame_id)
                    self.update_frame_id = self.master.after(100, self.update_video_feed) # Retry after a short delay
                    return
                # Calculate aspect ratio of the image
                img_width, img_height = img.size
                img_aspect = img_width / img_height
                # Calculate aspect ratio of the label
                label_aspect = label_width / label_height
                new_width, new_height = label_width, label_height
                # Resize image while maintaining aspect ratio
                if img_aspect > label_aspect:
                    # Image is wider than label, fit to width
                    new_height = int(label_width / img_aspect)
                else:
                    # Image is taller than label, fit to height
                    new_width = int(label_height * img_aspect)
                # Resize the image using PIL with a high-quality filter
                img = img.resize((new_width, new_height), Image.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=self.photo)
                self.video_label.image = self.photo  # Keep reference
                
                # Create overlay label if it doesn't exist
                if not hasattr(self, 'camera_label_overlay'):
                    self.camera_label_overlay = ttk.Label(self.video_label, 
                                                        text="Primary Camera",
                                                        font=("Arial", 12, "bold"),
                                                        foreground="white",
                                                        background="black")
                    self.camera_label_overlay.place(x=10, y=10)  # Top-left position
                
        except Empty:
            pass # No new frame in queue, just continue
        except Exception as e:
            print(f"Error updating frame: {e}")
            traceback.print_exc()
        finally:
            # Schedule the next update only if detection is still running
            if detection_running:
                self.update_frame_id = self.master.after(10, self.update_video_feed) # Update frequently for smooth video

    def stop_detection(self):
        global detection_running, picam2_instance
        if not detection_running:
            return
        
        detection_running = False
        self.status_label.config(text="Status: Stopping detection...")
        self.end_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED) # Disable start until fully stopped
        self.log_button.config(state=tk.DISABLED)
        self.add_camera_button.config(state=tk.DISABLED) # Disable during stopping
        self.more_cameras_button.config(state=tk.DISABLED)

        # Cancel any pending UI updates
        if self.update_frame_id:
            self.master.after_cancel(self.update_frame_id)
            self.update_frame_id = None

    def on_closing(self):
        # Ensure detection is stopped when the window is closed
        self.stop_detection()
        # Give a small delay for threads to clean up before destroying root
        self.master.after(500, self.master.destroy)

    def view_logs(self):
        log_window = tk.Toplevel(self.master)
        log_window.title("Smoker Detection Logs")
        log_window.geometry("1000x700")  # Larger default size
        log_window.configure(bg='#2B2B2B')
        
        # Make the window resizable
        log_window.resizable(True, True)
        
        # Configure grid weights
        log_window.grid_rowconfigure(0, weight=1)
        log_window.grid_columnconfigure(0, weight=1)
        
        # Frame for log content with scrollbar
        log_frame = ttk.Frame(log_window, padding="10")
        log_frame.grid(row=0, column=0, sticky="nsew")
        
        # Canvas with scrollbar
        canvas = tk.Canvas(log_frame, bg='#3C3C3C', highlightthickness=0)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=canvas.yview)
        
        # Pack them
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Create frame inside canvas
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        # Load content
        self.load_saved_frames(inner_frame)
        
        # Add mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Close button
        close_button = ttk.Button(
            log_window, 
            text="Close", 
            command=log_window.destroy,
            style='TButton'
        )
        close_button.grid(row=1, column=0, pady=10)

    def load_saved_frames(self, parent_frame):
        """Loads and displays saved frames in a 2-column grid layout with formatted timestamps"""
        # Clear existing content
        for widget in parent_frame.winfo_children():
            widget.destroy()

        if not os.path.exists(SAVED_FRAMES_BASE_DIR) or not os.listdir(SAVED_FRAMES_BASE_DIR):
            no_logs_label = ttk.Label(parent_frame, text="No smoker detection logs found.", font=("Arial", 14))
            no_logs_label.pack(pady=20)
            no_logs_label.configure(foreground='#AAAAAA', background='#3C3C3C')
            return

        # Configure parent frame grid for 2 columns
        parent_frame.grid_columnconfigure(0, weight=1)
        parent_frame.grid_columnconfigure(1, weight=1)

        # Sort folders by modification time (newest first)
        log_folders = sorted(
            [d for d in os.listdir(SAVED_FRAMES_BASE_DIR) 
            if os.path.isdir(os.path.join(SAVED_FRAMES_BASE_DIR, d))],
            key=lambda x: os.path.getmtime(os.path.join(SAVED_FRAMES_BASE_DIR, x)),
            reverse=True
        )[:6]  # Show up to 6 most recent folders (3 rows of 2 columns)

        for i, folder_name in enumerate(log_folders):
            folder_path = os.path.join(SAVED_FRAMES_BASE_DIR, folder_name)
            
            # Determine grid position (row, column)
            row = i // 2
            col = i % 2
            
            # Create log entry frame
            log_entry_frame = ttk.Frame(parent_frame, padding=10)
            log_entry_frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
            log_entry_frame.configure(style='TFrame')
            
            # Parse and format timestamp for display
            try:
                # Handle both old (20250717_211356) and new (2025-07-17_11-35-49PM) formats
                if "-" in folder_name:  # New format
                    date_part, time_part = folder_name.split("_")
                    dt_obj = datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%I-%M-%S%p")
                    display_time = dt_obj.strftime("%Y-%m-%d %I:%M:%S %p")  # 2025-07-17 11:35:49 PM
                else:  # Old format
                    dt_obj = datetime.strptime(folder_name, "%Y%m%d_%H%M%S")
                    display_time = dt_obj.strftime("%Y-%m-%d %I:%M:%S %p")  # Convert to new format
            except ValueError:
                display_time = folder_name.replace("_", " ")  # Fallback if parsing fails
            
            # Display formatted timestamp
            timestamp_label = ttk.Label(
                log_entry_frame, 
                text=f"Detection: {display_time}",
                font=("Arial", 12, "bold")
            )
            timestamp_label.pack(anchor="w", pady=(0, 5))
            
            # Frame for images with 2x2 grid
            images_frame = ttk.Frame(log_entry_frame)
            images_frame.pack(fill="both", expand=True)
            
            # Configure grid for images
            images_frame.grid_rowconfigure(0, weight=1)
            images_frame.grid_rowconfigure(1, weight=1)
            images_frame.grid_columnconfigure(0, weight=1)
            images_frame.grid_columnconfigure(1, weight=1)
            
            # Get frames and show the 4 newest
            frames = sorted(
                [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))],
                key=lambda x: os.path.getmtime(os.path.join(folder_path, x)),
                reverse=True
            )[:4]  # Show 4 most recent frames per folder

            # Display images in 2x2 grid
            for j, frame_file in enumerate(frames):
                try:
                    img_path = os.path.join(folder_path, frame_file)
                    
                    # Determine position in 2x2 grid
                    img_row = j // 2
                    img_col = j % 2
                    
                    # Create image container frame
                    img_container = ttk.Frame(images_frame, padding=2)
                    img_container.grid(row=img_row, column=img_col, sticky="nsew")
                    
                    # Load with PIL and create thumbnail
                    img = Image.open(img_path)
                    img.thumbnail((300, 225), Image.LANCZOS)  # 4:3 aspect ratio
                    
                    # Convert to PhotoImage and display
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(
                        img_container, 
                        image=photo,
                        compound="center"
                    )
                    img_label.image = photo  # Keep reference
                    img_label.pack(fill="both", expand=True)
                    
                    # Add border
                    img_container.configure(style='VideoFrame.TFrame')
                    
                    # Force garbage collection
                    del img
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error loading {frame_file}: {e}")
                    error_label = ttk.Label(images_frame, text="[Image Error]", foreground="red")
                    error_label.grid(row=img_row, column=img_col)

# Main execution block
if __name__ == "__main__":
    cleanup_temp_files() # Clean up at startup
    root = tk.Tk()
    app = SmokingDetectionApp(root)
    root.mainloop()
