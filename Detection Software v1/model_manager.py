#!/usr/bin/env python3
"""
Model Manager - This module handles loading and managing YOLO and RF-DETR models.

Supported Model Types:
1. .pt files
2. .pth files

Key Features:
- Automatic model type detection based on file extension
- GPU acceleration when CUDA is available
- Model warm-up to eliminate first-inference latency
- Detailed timing breakdown for performance analysis
- Thread-safe model loading with PyQt signals
- Memory management with model unloading

Model Loading Process:
1. Auto-detect model type from file extension
2. Load model using appropriate method
3. Move model to optimal device (GPU/CPU)
4. Perform warm-up inference cycles
5. Emit signals to notify GUI of status

The warm-up process runs multiple dummy inferences to:
- Initialize CUDA kernels (if using GPU)
- Load model weights into memory
- Eliminate JIT compilation overhead
- Provide consistent inference timing

Date: 2025-07-29
Version: 1.0
"""

import os
import time
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from PyQt6.QtCore import QObject, pyqtSignal
from logger_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

class ModelManager(QObject):
    """
    Model Manager Class to manages loading, warming up, and running inference on AI models.
    
    Signals:
        model_loaded(str, str): Emitted when model loads successfully (path, type)
        model_error(str): Emitted when model loading fails (error_message)
        model_warming_started(): Emitted when model warm-up begins
        model_warming_completed(): Emitted when model warm-up finishes
    
    Attributes:
        current_model: The loaded AI model object
        current_model_path: Path to the currently loaded model file
        current_model_type: Type of current model ('yolo' or 'rfdetr')
        is_warmed_up: Whether model has completed warm-up process
        device: PyTorch device (cuda or cpu) for inference
    """
    model_loaded = pyqtSignal(str, str)        # model_path, model_type
    model_error = pyqtSignal(str)              # error_message
    model_warming_started = pyqtSignal()       # warm-up started
    model_warming_completed = pyqtSignal()     # warm-up completed
    
    def __init__(self):
        super().__init__()
        # Model state tracking
        self.current_model = None           # The loaded model object
        self.current_model_path = None      # Path to model file
        self.current_model_type = None      # 'yolo' or 'rfdetr'
        self.is_warmed_up = False          # Whether warm-up is complete
        
        # Determine optimal device for inference (GPU preferred for speed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ModelManager initialized with device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
            logger.debug(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            logger.warning("CUDA not available - using CPU for inference (slower performance)")
    
    def load_trained_model(self, checkpoint_path: str, actual_classes: int = 2):
        """
        Load an RF-DETR model with 2 classes.
        
        Args:
            checkpoint_path (str): Path to the .pth checkpoint file
            actual_classes (int): Number of classes the model was trained on (default: 2)
            
        Returns:
            RF-DETR model object ready for inference
            
        Raises:
            Exception: If RF-DETR components are not available or loading fails
        """
        try:
            # Import RF-DETR components (must be available in environment)
            from rfdetr import RFDETRBase
            
            # Get base model and configuration
            model = RFDETRBase()
            config = model.get_model_config()
            
            # Change config to match the size of the detector head of the fine-tuned model
            config.num_classes = actual_classes - 1
            config.pretrain_weights = None
            
            # Create model with correct configuration
            pytorch_model = model.get_model(config)
            actual_model = pytorch_model.model
            
            # Load the fine-tuned model into the actual model
            checkpoint_data = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
            actual_model.load_state_dict(checkpoint_data['model'])
            
            # Update the rfdetr wrapper
            pytorch_model.model = actual_model
            model.model = pytorch_model
            
            # Set to evaluation mode
            model.model.model.eval()
            
            return model
            
        except Exception as e:
            logger.error(f"RF-DETR model loading failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to load RF-DETR model: {str(e)}")
    
    def load_yolo_model(self, model_path: str):
        try:
            # Try to import ultralytics YOLO
            from ultralytics import YOLO
            
            model = YOLO(model_path)
            model.to(self.device)
            
            return model
            
        except ImportError:
            # Fallback to torch.hub if ultralytics not available
            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                model.to(self.device)
                return model
            except Exception as e:
                raise Exception(f"Failed to load YOLO model with torch.hub: {str(e)}")
        except Exception as e:
            logger.error(f"YOLO model loading failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to load YOLO model: {str(e)}")
    
    def load_model(self, model_path: str, model_type: Optional[str] = None) -> bool:
        if not os.path.exists(model_path):
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            self.model_error.emit(error_msg)
            return False
        
        # Auto-detect model type if not specified
        if model_type is None:
            if model_path.endswith('.pt'):
                model_type = 'yolo'
            elif model_path.endswith('.pth'):
                model_type = 'rfdetr'
            else:
                error_msg = f"Unknown model type for file: {model_path}"
                logger.error(error_msg)
                self.model_error.emit(error_msg)
                return False
        
        try:
            logger.info(f"Loading {model_type.upper()} model: {model_path}")
            start_time = time.time()
            
            if model_type.lower() == 'yolo':
                self.current_model = self.load_yolo_model(model_path)
            elif model_type.lower() == 'rfdetr':
                self.current_model = self.load_trained_model(model_path)
            else:
                error_msg = f"Unsupported model type: {model_type}"
                logger.error(error_msg)
                self.model_error.emit(error_msg)
                return False
            
            load_time = time.time() - start_time
            
            self.current_model_path = model_path
            self.current_model_type = model_type.lower()
            
            logger.info(f"{model_type.upper()} model loaded successfully in {load_time:.2f}s")
            logger.debug(f"Model details - Device: {self.device}, Path: {model_path}")
            self.model_loaded.emit(model_path, model_type)
            
            # Perform model warm-up to eliminate first-inference overhead
            self._warm_up_model()
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model {model_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.model_error.emit(error_msg)
            return False
    
    def _warm_up_model(self):
        """
        Pre-warm the model to eliminate first-inference overhead and ensure consistent timing.
        
        Model warm-up is critical for industrial applications because:
        1. First inference is often 2-10x slower due to GPU kernel initialization
        2. CUDA memory allocation happens on first use
        3. PyTorch JIT compilation occurs during initial runs
        4. Model weights may need to be loaded into GPU memory
        """
        if self.current_model is None:
            return
        
        try:
            logger.info("Starting model warm-up process...")
            self.model_warming_started.emit()
            
            # Create a dummy image for warm-up
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            
            # Run multiple warm-up inferences to ensure all systems are initialized
            warm_up_start = time.time()
            for i in range(3):
                _ = self.predict(dummy_image)  # Run inference and discard results
            
            warm_up_time = time.time() - warm_up_start
            self.is_warmed_up = True
            
            logger.info(f"Model warm-up completed in {warm_up_time:.2f}s (3 inference cycles)")
            logger.debug(f"Model ready for production inference with consistent timing")
            self.model_warming_completed.emit()
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {str(e)} - proceeding without warm-up")
            # Don't emit error signal - warm-up failure shouldn't prevent usage
            self.is_warmed_up = False
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if self.current_model is None:
            logger.error("Inference attempted with no model loaded")
            raise Exception("No model loaded")
        
        logger.debug(f"Starting inference with {self.current_model_type} model")
        
        timing_breakdown = {}
        start_total = time.time()
        
        try:
            if self.current_model_type == 'yolo':
                return self._predict_yolo(image, timing_breakdown)
            elif self.current_model_type == 'rfdetr':
                return self._predict_rfdetr(image, timing_breakdown)
            else:
                raise Exception(f"Unknown model type: {self.current_model_type}")
                
        except Exception as e:
            timing_breakdown['total_time'] = time.time() - start_total
            logger.error(f"Inference failed: {str(e)}", exc_info=True)
            return {
                'detections': [],
                'timing': timing_breakdown,
                'error': str(e)
            }
    
    def _predict_yolo(self, image: np.ndarray, timing_breakdown: Dict[str, float]) -> Dict[str, Any]:
        start_time = time.time()
        
        # Preprocessing
        preprocess_start = time.time()
        # YOLO expects RGB image
        timing_breakdown['preprocessing'] = time.time() - preprocess_start
        
        # Inference
        inference_start = time.time()
        results = self.current_model(image)
        timing_breakdown['inference'] = time.time() - inference_start
        
        # Postprocessing
        postprocess_start = time.time()
        detections = []
        
        if hasattr(results, '__len__') and len(results) > 0:
            result = results[0]
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bounding box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name if available
                    class_name = str(class_id)
                    if hasattr(self.current_model, 'names') and class_id in self.current_model.names:
                        class_name = self.current_model.names[class_id]
                    
                    detections.append({
                        'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        timing_breakdown['postprocessing'] = time.time() - postprocess_start
        timing_breakdown['total_time'] = time.time() - start_time
        
        logger.debug(f"YOLO inference completed: {len(detections)} detections in {timing_breakdown['total_time']*1000:.1f}ms")
        if detections:
            logger.info(f"YOLO detected {len(detections)} objects")
            for det in detections:
                logger.debug(f"  {det['class_name']}: {det['confidence']:.3f} confidence")
        
        return {
            'detections': detections,
            'timing': timing_breakdown,
            'model_type': 'yolo'
        }
    
    def _predict_rfdetr(self, image: np.ndarray, timing_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """
        Run inference using RF-DETR model and parse supervision.Detections results.
        
        RF-DETR returns a supervision.Detections object with attributes:
        - xyxy: Bounding boxes in (x1, y1, x2, y2) format
        - confidence: Detection confidence scores  
        - class_id: Class indices
        """
        start_time = time.time()
        
        # Preprocessing
        preprocess_start = time.time()
        # RF-DETR expects RGB image - no preprocessing needed as image is already RGB
        timing_breakdown['preprocessing'] = time.time() - preprocess_start
        
        # Inference
        inference_start = time.time()
        with torch.no_grad():
            results = self.current_model.predict(image)
        timing_breakdown['inference'] = time.time() - inference_start
        
        # Postprocessing - Handle supervision.Detections object
        postprocess_start = time.time()
        detections = []
        
        # RF-DETR returns a supervision.Detections object
        # The supervision.Detections object has specific attributes we need to access correctly
        if results is not None and len(results) > 0:
            # Extract detection arrays from supervision.Detections object
            # Each attribute contains numpy arrays with detection data
            boxes = results.xyxy  # Shape: (N, 4) - bounding boxes in [x1, y1, x2, y2] format
            confidences = results.confidence if results.confidence is not None else np.ones(len(boxes))
            class_ids = results.class_id if results.class_id is not None else np.zeros(len(boxes), dtype=int)
            
            # Convert each detection to our standardized format for consistency with YOLO
            # This ensures both model types produce identical output structures for the GUI
            for i in range(len(boxes)):
                box = boxes[i]  # Extract individual bounding box [x1, y1, x2, y2]
                confidence = float(confidences[i]) if confidences is not None else 1.0
                class_id = int(class_ids[i]) if class_ids is not None else 0
                
                # Map numeric class IDs to human-readable names for industrial context
                # Based on RF-DETR model training: specific defect types
                class_name = f"class_{class_id}"  # Default fallback name
                if class_id == 0:
                    class_name = "crushed edge"    # Class 0: Crushed edge defect
                elif class_id == 1:
                    class_name = "split"           # Class 1: Split defect
                
                # Create standardized detection dictionary matching YOLO output format
                # This ensures compatibility with the rest of the application pipeline
                detections.append({
                    'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],  # Convert to native Python floats
                    'confidence': confidence,    # Detection confidence score (0.0-1.0)
                    'class_id': class_id,       # Numeric class identifier  
                    'class_name': class_name    # Human-readable class name
                })
        
        timing_breakdown['postprocessing'] = time.time() - postprocess_start
        timing_breakdown['total_time'] = time.time() - start_time
        
        logger.debug(f"RF-DETR inference completed: {len(detections)} detections in {timing_breakdown['total_time']*1000:.1f}ms")
        if detections:
            logger.info(f"RF-DETR detected {len(detections)} objects")
            for det in detections:
                logger.debug(f"  {det['class_name']}: {det['confidence']:.3f} confidence")
        
        return {
            'detections': detections,
            'timing': timing_breakdown,
            'model_type': 'rfdetr'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        if self.current_model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'path': self.current_model_path,
            'type': self.current_model_type,
            'device': str(self.device)
        }
    
    def unload_model(self):
        self.current_model = None
        self.current_model_path = None
        self.current_model_type = None
        self.is_warmed_up = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
        
        logger.info("Model unloaded and resources freed")

if __name__ == '__main__':
    # Test the model manager
    model_manager = ModelManager()
    
    # Test loading models from the models folder
    models_folder = "models"
    if os.path.exists(models_folder):
        for model_file in os.listdir(models_folder):
            model_path = os.path.join(models_folder, model_file)
            print(f"\nTesting model: {model_file}")
            
            if model_manager.load_model(model_path):
                print(f"Model info: {model_manager.get_model_info()}")
                
                # Test with a dummy image
                dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                try:
                    result = model_manager.predict(dummy_image)
                    print(f"Prediction result: {len(result.get('detections', []))} detections")
                    print(f"Timing: {result.get('timing', {})}")
                except Exception as e:
                    print(f"Prediction error: {e}")
                
                model_manager.unload_model()
            else:
                print(f"Failed to load model: {model_file}")
    else:
        print("Models folder not found")