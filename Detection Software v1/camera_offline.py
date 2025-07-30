#!/usr/bin/env python3
"""
Camera Offline Module - This module simulates an industrial camera by cycling through 
test images stored in a local folder. 

Key Features:
- Loads images from 'test images' folder automatically
- Supports common industrial image formats (JPG, PNG, BMP, TIFF)
- Provides navigation methods (next, previous, random, by index)
- Emits PyQt signals when images are loaded
- Converts images to consistent RGB format for processing
- Handles image display conversion to QPixmap for GUI

This simulation approach allows:
- Testing detection algorithms with known images
- Demonstrating the system without camera hardware
- Consistent results for development and debugging
- Easy switching between different test image sets

Date: 2025-07-29
Version: 1.0
"""

import os
import random
from typing import List, Optional
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QObject, pyqtSignal
import cv2
import numpy as np

class CameraOffline(QObject):
    """    
    This class mimics the behavior of a real industrial camera by cycling through
    a folder of test images. 
    
    Signals:
        image_loaded(str, np.ndarray): Emitted when a new image is loaded
                                      Parameters: (image_path, image_array)
    
    Attributes:
        test_images_folder (str): Path to folder containing test images
        image_files (list): List of valid image file paths
        current_index (int): Index of currently loaded image
        current_image (np.ndarray): Currently loaded image as numpy array
        current_image_path (str): Path to currently loaded image file
    """
    image_loaded = pyqtSignal(str, np.ndarray)  # Signal: (image_path, image_array)
    
    def __init__(self, test_images_folder: str = "test images"):
        super().__init__()
        self.test_images_folder = test_images_folder
        self.image_files = []           # List of valid image file paths
        self.current_index = 0          # Index of currently displayed image
        self.current_image = None       # Current image as numpy array (RGB format)
        self.current_image_path = None  # Full path to current image file
        
        # Automatically discover and load available test images
        self.load_image_list()
    
    def load_image_list(self) -> None:
        if not os.path.exists(self.test_images_folder):
            print(f"Warning: Test images folder '{self.test_images_folder}' not found")
            return
            
        # Get all image files from the test images folder
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        self.image_files = []
        
        for filename in os.listdir(self.test_images_folder):
            if filename.lower().endswith(supported_formats):
                self.image_files.append(os.path.join(self.test_images_folder, filename))
        
        self.image_files.sort()  # Sort for consistent ordering
        print(f"Found {len(self.image_files)} test images")
        
        if self.image_files:
            self.load_current_image()
    
    def load_current_image(self) -> bool:
        if not self.image_files:
            print("No images available")
            return False
            
        image_path = self.image_files[self.current_index]
        
        try:
            # Load image using OpenCV for consistent format
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return False
                
            # Convert BGR to RGB for consistent processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.current_image = image_rgb
            self.current_image_path = image_path
            
            # Emit signal with image data
            self.image_loaded.emit(image_path, image_rgb)
            
            print(f"Loaded image: {os.path.basename(image_path)} [{image_rgb.shape}]")
            return True
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False
    
    def get_next_image(self) -> bool:
        if not self.image_files:
            return False
            
        self.current_index = (self.current_index + 1) % len(self.image_files)
        return self.load_current_image()
    
    def get_previous_image(self) -> bool:
        if not self.image_files:
            return False
            
        self.current_index = (self.current_index - 1) % len(self.image_files)
        return self.load_current_image()
    
    def get_random_image(self) -> bool:
        if not self.image_files:
            return False
            
        self.current_index = random.randint(0, len(self.image_files) - 1)
        return self.load_current_image()
    
    def get_image_by_index(self, index: int) -> bool:
        if not self.image_files or index < 0 or index >= len(self.image_files):
            return False
            
        self.current_index = index
        return self.load_current_image()
    
    def get_current_image(self) -> Optional[np.ndarray]:
        return self.current_image
    
    def get_current_image_path(self) -> Optional[str]:
        return self.current_image_path
    
    def get_current_image_name(self) -> Optional[str]:
        if self.current_image_path:
            return os.path.basename(self.current_image_path)
        return None
    
    def get_image_count(self) -> int:
        return len(self.image_files)
    
    def get_current_index(self) -> int:
        return self.current_index
    
    def simulate_trigger(self) -> bool:
        """
        Simulate an industrial camera trigger event.
        
        Returns:
            bool: True if image successfully loaded, False if no images available
        """
        return self.get_next_image()
    
    def get_image_info(self) -> dict:
        if self.current_image is None:
            return {}
            
        return {
            'path': self.current_image_path,
            'name': self.get_current_image_name(),
            'shape': self.current_image.shape,
            'index': self.current_index,
            'total_images': len(self.image_files)
        }
    
    def numpy_to_qpixmap(self, image: np.ndarray) -> QPixmap:
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        
        # Convert numpy array to QImage then to QPixmap
        from PyQt6.QtGui import QImage
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(q_image)
    
    def get_current_qpixmap(self) -> Optional[QPixmap]:
        if self.current_image is not None:
            return self.numpy_to_qpixmap(self.current_image)
        return None

if __name__ == '__main__':
    # Test the camera offline module
    camera = CameraOffline()
    
    print(f"Total images: {camera.get_image_count()}")
    
    if camera.get_image_count() > 0:
        print(f"Current image: {camera.get_current_image_name()}")
        print(f"Image info: {camera.get_image_info()}")
        
        # Test navigation
        print("\nTesting navigation:")
        camera.get_next_image()
        print(f"Next image: {camera.get_current_image_name()}")
        
        camera.get_previous_image()
        print(f"Previous image: {camera.get_current_image_name()}")
        
        camera.get_random_image()
        print(f"Random image: {camera.get_current_image_name()}")
    else:
        print("No test images found!")