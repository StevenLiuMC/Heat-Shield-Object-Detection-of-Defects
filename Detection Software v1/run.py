#!/usr/bin/env python3

"""
GUI Startup.
Checks dependencies and starts the application with proper error handling.

Date: 2025-07-29
Version: 1.0

[run.py]
   └──➜ [main.py]
           ├──➜ [camera_offline.py]        # Camera simulator, load test image
           ├──➜ [model_manager.py]         # Model loader (supports YOLO and RF-DETR)
           │       └──➜ [logger_config.py] # Log configurator, providing a unified logging interface for all modules
           ├──➜ [inference_engine.py]      # Inference Pipeline
           │       └──➜ [logger_config.py]
           ├──➜ [database_manager.py]      # Database operator, recording production cycle, OEE, downtime information, etc.
           │       └──➜ [logger_config.py]
           ├──➜ [export_manager.py]        # Data exporter, export CSV / JSON / HTML reports
           │       └──➜ [database_manager.py]
           └──➜ [logger_config.py]         # The GUI itself also records logs
"""

import sys
import os
import importlib.util
import traceback

def check_dependencies():
    """
    Verify that all required Python packages are installed and importable.
    
    Returns:
        bool: True if all dependencies are available, False if any are missing
    """
    required_packages = {
        'PyQt6': 'PyQt6',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'Pillow'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                importlib.import_module(package)
            print(f"[OK] {package} is available")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"[MISSING] {package} is missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_files():
    """
    Verify that all core application files are present in the current directory.
    
    Checks for essential Python modules that the application needs to run:
    - main.py: Main application window and GUI
    - camera_offline.py: Simulated camera functionality
    - model_manager.py: AI model loading and management
    - inference_engine.py: Detection processing pipeline
    - database_manager.py: Production data storage
    
    Returns:
        bool: True if all required files exist, False if any are missing
    """
    required_files = [
        'main.py',
        'camera_offline.py',
        'model_manager.py',
        'inference_engine.py',
        'database_manager.py'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} found")
        else:
            missing_files.append(file)
            print(f"[MISSING] {file} missing")
    
    if missing_files:
        print(f"\nMissing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_test_data():
    """
    Verify that test images and model folders exist, creating them if necessary.
    
    Checks for:
    - 'test images' folder: Contains sample images for camera simulation
    - 'models' folder: Contains AI model files (.pt and .pth)
       
    This function doesn't return a value as missing test data is not critical
    for application startup (folders can be empty initially).
    """
    test_images_folder = "test images"
    models_folder = "models"
    
    # Check test images
    if os.path.exists(test_images_folder):
        image_files = [f for f in os.listdir(test_images_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"[OK] Found {len(image_files)} test images")
    else:
        print("[WARNING] Test images folder not found - creating empty folder")
        os.makedirs(test_images_folder, exist_ok=True)
    
    # Check models
    if os.path.exists(models_folder):
        model_files = [f for f in os.listdir(models_folder) 
                      if f.lower().endswith(('.pt', '.pth'))]
        print(f"[OK] Found {len(model_files)} model files")
        if len(model_files) == 0:
            print("[WARNING] No model files found - you'll need to load models manually")
    else:
        print("[WARNING] Models folder not found - creating empty folder")
        os.makedirs(models_folder, exist_ok=True)

def main():
    """
    Main startup function that performs comprehensive system checks before launching the GUI.
    
    Startup sequence:
    1. Check Python version compatibility (3.8+ required)
    2. Verify all required dependencies are installed
    3. Confirm all application files are present
    4. Check/create test data folders
    5. Launch the main application window
    6. Load first available test image if present
    
    Returns:
        int: 0 for successful startup, 1 for errors
    """
    print("Project GUI - Startup Check")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"[ERROR] Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return 1
    else:
        print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    print("\nChecking dependencies...")
    if not check_dependencies():
        print("\n[FAILED] Dependency check failed")
        return 1
    
    print("\nChecking files...")
    if not check_files():
        print("\n[FAILED] File check failed")
        return 1
    
    print("\nChecking test data...")
    check_test_data()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All checks passed! Starting application...")
    print("=" * 50)
    
    try:
        # Import and run the main application
        from main import MainWindow
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPalette, QColor
        
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Set dark palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        app.setPalette(palette)
        
        # Create and show main window
        window = MainWindow()
        window.show()
               
        print("[STARTED] Application started successfully!")
        print("Tips:")
        print("   - Load a model using 'Load Model' button")
        print("   - Start a new shift using 'Start New Shift' button")
        print("   - Use 'TRIGGER DETECTION' to run inference")
        
        return app.exec()
        
    except Exception as e:
        print(f"\n[ERROR] Application startup failed:")
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())