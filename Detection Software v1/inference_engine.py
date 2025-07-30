#!/usr/bin/env python3
"""
Inference Engine - This module handles the complete inference pipeline for defect detection,
from image preprocessing through result visualization. It provides detailed
timing analysis and standardized result formatting for both YOLO and RF-DETR models.

Inference Pipeline:
1. Image Preprocessing: Format conversion, validation, and preparation
2. Model Inference: Run AI model prediction with timing
3. Result Postprocessing: Filter, sort, and classify detections
4. Visualization: Create annotated images with bounding boxes
5. Result Packaging: Format results for database storage and GUI display

Detection Classification Logic:
- Any detection above confidence threshold = DEFECT DETECTED
- No detections = GOOD
- Both class 0 and class 1 detections are considered defects
- Quality status drives production statistics and OEE calculation

Date: 2025-07-29
Version: 1.0
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex
from PyQt6.QtGui import QPainter, QPen, QFont, QColor
import cv2
from logger_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

class InferenceEngine(QObject):
    """
    This engine coordinates the entire process from raw image input to
    final annotated results. 
    
    Signals:
        inference_completed(dict): Emitted when inference succeeds with full results
        inference_error(str): Emitted when inference fails with error message
    
    The results dictionary contains:
    - original_image: Input image array
    - annotated_image: Image with detection bounding boxes
    - detections: List of detection objects with bbox, confidence, class
    - detection_summary: Overall quality status and counts
    - timing: Detailed breakdown of processing times
    - model_type: Type of AI model used
    - timestamp: When inference was performed
    
    Thread Safety:
    Uses QMutex to ensure only one inference runs at a time, preventing
    resource conflicts and ensuring consistent results.
    """
    inference_completed = pyqtSignal(dict)  # Complete results package
    inference_error = pyqtSignal(str)       # Error message
    
    def __init__(self, model_manager):
        super().__init__()
        self.model_manager = model_manager  # Reference to model management system
        self.mutex = QMutex()              # Thread safety for concurrent access
        
    def run_inference(self, image: np.ndarray, image_path: str = None, confidence_threshold: float = 0.3) -> Dict[str, Any]:
        self.mutex.lock()
        try:
            if self.model_manager.current_model is None:
                error_msg = "No model loaded for inference"
                logger.error("Inference requested but no model loaded")
                self.inference_error.emit(error_msg)
                return {'error': error_msg}
            
            logger.debug(f"Starting inference with confidence threshold: {confidence_threshold}")
            
            # Start timing
            total_start = time.time()
            
            # Detailed timing breakdown
            timing_breakdown = {
                'image_preprocessing': 0.0,
                'model_inference': 0.0,
                'result_postprocessing': 0.0,
                'visualization_preparation': 0.0,
                'total_time': 0.0
            }
            
            # Image preprocessing
            preprocess_start = time.time()
            processed_image = self._preprocess_image(image)
            timing_breakdown['image_preprocessing'] = time.time() - preprocess_start
            
            # Model inference
            inference_start = time.time()
            model_results = self.model_manager.predict(processed_image)
            timing_breakdown['model_inference'] = time.time() - inference_start
            
            # Extract model timing if available
            if 'timing' in model_results:
                model_timing = model_results['timing']
                timing_breakdown.update(model_timing)
            
            # Result postprocessing
            postprocess_start = time.time()
            processed_results = self._postprocess_results(model_results, image.shape, confidence_threshold)
            timing_breakdown['result_postprocessing'] = time.time() - postprocess_start
            
            # Visualization preparation
            viz_start = time.time()
            annotated_image = self._prepare_visualization(image, processed_results['detections'])
            timing_breakdown['visualization_preparation'] = time.time() - viz_start
            
            # Total time
            timing_breakdown['total_time'] = time.time() - total_start
            
            # Compile final results
            final_results = {
                'image_path': image_path,
                'original_image': image,
                'annotated_image': annotated_image,
                'detections': processed_results['detections'],
                'detection_summary': processed_results['summary'],
                'model_type': self.model_manager.current_model_type,
                'model_path': self.model_manager.current_model_path,
                'timing': timing_breakdown,
                'timestamp': time.time()
            }
            
            # Log inference completion with summary
            summary = final_results.get('detection_summary', {})
            total_time_ms = timing_breakdown['total_time'] * 1000
            logger.info(f"Inference completed in {total_time_ms:.1f}ms - Status: {summary.get('quality_status', 'UNKNOWN')}")
            if summary.get('has_defects', False):
                logger.warning(f"Defects detected: {summary.get('defect_count', 0)} detections found")
            logger.debug(f"Timing breakdown - Preprocessing: {timing_breakdown['image_preprocessing']*1000:.1f}ms, Inference: {timing_breakdown['model_inference']*1000:.1f}ms, Postprocessing: {timing_breakdown['result_postprocessing']*1000:.1f}ms")
            
            self.inference_completed.emit(final_results)
            return final_results
            
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.inference_error.emit(error_msg)
            return {'error': error_msg}
        finally:
            self.mutex.unlock()
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image
        elif len(image.shape) == 2:
            # Convert grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image format: {image.shape}")
    
    def _postprocess_results(self, model_results: Dict[str, Any], image_shape: Tuple[int, int, int], confidence_threshold: float = 0.3) -> Dict[str, Any]:
        """
        This function converts raw AI model outputs into a consistent format
        regardless of the underlying model type (YOLO vs RF-DETR). 
        
        Processing Steps:
        1. Filter detections by confidence threshold
        2. Sort detections by confidence (highest first)
        3. Count detections by class type
        4. Determine overall quality status
        5. Generate summary statistics
        
        Quality Logic for Industrial Defect Detection:
        - ANY detection above threshold = DEFECT DETECTED
        - NO detections = GOOD part
        - Both class 0 and class 1 are considered defect types
        
        Args:
            model_results: Raw outputs from AI model
            image_shape: Dimensions of input image (H, W, C)
            confidence_threshold: Minimum confidence for valid detection (0.0-1.0)
            
        Returns:
            Dictionary with 'detections' list and 'summary' statistics
        """
        detections = model_results.get('detections', [])
        
        # Filter detections by confidence threshold (use parameter)
        filtered_detections = [
            det for det in detections 
            if det.get('confidence', 0) >= confidence_threshold
        ]
        
        # Sort by confidence
        filtered_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Classification summary
        class_counts = {}
        total_detections = len(filtered_detections)
        
        for detection in filtered_detections:
            class_name = detection.get('class_name', 'unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # ANY detection indicates a defect has been found
        has_defects = total_detections > 0
        defect_count = total_detections if has_defects else 0
        good_count = 0  # No good parts counted if ANY defects are detected
        
        # Determine overall classification
        quality_status = "DEFECT DETECTED" if has_defects else "GOOD"
        
        summary = {
            'total_detections': total_detections,
            'defect_count': defect_count,
            'good_count': good_count,
            'class_counts': class_counts,
            'has_defects': has_defects,
            'quality_status': quality_status,
            'confidence_threshold': confidence_threshold
        }
        
        return {
            'detections': filtered_detections,
            'summary': summary
        }
    
    def _prepare_visualization(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        # Create a copy for annotation
        annotated = image.copy()
        
        # Use red color for all defects
        defect_color = (255, 0, 0)
        
        for detection in detections:
            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                continue
                
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class_name', 'unknown')
            
            # Draw bounding box only (no text labels)
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), defect_color, 3)  # Thicker line
            
            # Optional: Add small corner marks for better visibility
            corner_size = 10
            # Top-left corner
            cv2.line(annotated, (x1, y1), (x1 + corner_size, y1), defect_color, 4)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_size), defect_color, 4)
            # Top-right corner  
            cv2.line(annotated, (x2, y1), (x2 - corner_size, y1), defect_color, 4)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_size), defect_color, 4)
            # Bottom-left corner
            cv2.line(annotated, (x1, y2), (x1 + corner_size, y2), defect_color, 4)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_size), defect_color, 4)
            # Bottom-right corner
            cv2.line(annotated, (x2, y2), (x2 - corner_size, y2), defect_color, 4)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_size), defect_color, 4)
        
        return annotated
    
    def get_timing_report(self, timing_data: Dict[str, float]) -> str:
        if not timing_data:
            return "No timing data available"
        
        report = "=== INFERENCE TIMING BREAKDOWN ===\n"
        
        # Order for display
        timing_order = [
            ('image_preprocessing', 'Image Preprocessing'),
            ('preprocessing', 'Model Preprocessing'),
            ('inference', 'Model Inference'),
            ('model_inference', 'Total Model Time'),
            ('postprocessing', 'Model Postprocessing'),
            ('result_postprocessing', 'Result Processing'),
            ('visualization_preparation', 'Visualization Prep'),
            ('total_time', 'TOTAL TIME')
        ]
        
        for key, label in timing_order:
            if key in timing_data:
                time_ms = timing_data[key] * 1000
                report += f"{label:<25}: {time_ms:>7.2f} ms\n"
        
        # Calculate FPS
        total_time = timing_data.get('total_time', 0)
        if total_time > 0:
            fps = 1.0 / total_time
            report += f"{'Theoretical FPS':<25}: {fps:>7.2f}\n"
        
        return report
    
    def get_detection_report(self, results: Dict[str, Any]) -> str:
        if 'detection_summary' not in results:
            return "No detection results available"
        
        summary = results['detection_summary']
        detections = results.get('detections', [])
        
        report = "=== DETECTION RESULTS ===\n"
        report += f"Total Detections: {summary['total_detections']}\n"
        report += f"Defects Found: {summary['defect_count']}\n"
        report += f"Good Parts: {summary['good_count']}\n"
        report += f"Quality Status: {summary['quality_status']}\n"
        report += f"Confidence Threshold: {summary['confidence_threshold']}\n\n"
        
        if detections:
            report += "Individual Detections:\n"
            for i, det in enumerate(detections, 1):
                bbox = det.get('bbox', [])
                conf = det.get('confidence', 0)
                class_name = det.get('class_name', 'unknown')
                
                report += f"{i}. {class_name} (conf: {conf:.3f}) "
                report += f"at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]\n"
        
        return report

class InferenceWorker(QThread):
    def __init__(self, inference_engine, image, image_path=None):
        super().__init__()
        self.inference_engine = inference_engine
        self.image = image
        self.image_path = image_path
        
    def run(self):
        self.inference_engine.run_inference(self.image, self.image_path)

if __name__ == '__main__':
    # Test the inference engine
    import sys
    sys.path.append('.')
    
    from model_manager import ModelManager
    from camera_offline import CameraOffline
    
    # Initialize components
    model_manager = ModelManager()
    inference_engine = InferenceEngine(model_manager)
    camera = CameraOffline()
    
    # Test with available models and images
    models_folder = "models"
    import os
    
    if os.path.exists(models_folder) and camera.get_image_count() > 0:
        # Load first available model
        for model_file in os.listdir(models_folder):
            model_path = os.path.join(models_folder, model_file)
            if model_manager.load_model(model_path):
                print(f"Loaded model: {model_file}")
                break
        
        # Run inference on first test image
        if model_manager.current_model is not None:
            test_image = camera.get_current_image()
            if test_image is not None:
                print("Running test inference...")
                results = inference_engine.run_inference(test_image, camera.get_current_image_path())
                
                if 'error' not in results:
                    print(inference_engine.get_timing_report(results.get('timing', {})))
                    print(inference_engine.get_detection_report(results))
                else:
                    print(f"Inference error: {results['error']}")
            else:
                print("No test image available")
        else:
            print("No model loaded")
    else:
        print("No models or test images found")