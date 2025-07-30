#!/usr/bin/env python3
"""
This is the main application file. It provides a PyQt6-based GUI for:
- Loading and managing AI models (YOLO and RF-DETR)
- Simulating camera input with test images
- Running defect detection inference
- Tracking production metrics and OEE (Overall Equipment Effectiveness)
- Managing work shifts and downtime events
- Displaying real-time statistics and performance metrics

The application uses a modular architecture with separate components for:
- Camera simulation (camera_offline.py)
- Model management (model_manager.py) 
- Inference processing (inference_engine.py)
- Database operations (database_manager.py)

Date: 2025-07-29
Version: 1.0
"""

import sys
import os
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                             QTextEdit, QFrame, QGridLayout, QGroupBox,
                             QProgressBar, QLCDNumber, QFileDialog, QMessageBox,
                             QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QLineEdit)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPalette, QColor

from camera_offline import CameraOffline
from model_manager import ModelManager
from inference_engine import InferenceEngine
from database_manager import DatabaseManager
from logger_config import get_logger

# Initialize logging for this module
logger = get_logger(__name__)


class ModelLoadingWorker(QThread):
    """
    Background worker thread for loading AI models without freezing the GUI.
    
    Model loading can take several seconds, especially for large models.
    This worker runs in a separate thread to keep the GUI responsive during loading.
    
    Signals:
        loading_finished: Emitted when loading completes with (success, file_path)
    """
    loading_finished = pyqtSignal(bool, str)  # success, file_path
    
    def __init__(self, model_manager, file_path):
        super().__init__()
        self.model_manager = model_manager
        self.file_path = file_path
    
    def run(self):
        """Load the model in a separate thread"""
        try:
            success = self.model_manager.load_model(self.file_path)
            self.loading_finished.emit(success, self.file_path)
        except Exception as e:
            print(f"Model loading error in worker thread: {e}")
            self.loading_finished.emit(False, self.file_path)


def truncate_text(text, max_length=30):
    """Utility function to truncate long text strings for display in the GUI"""  
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

class ShiftDialog(QDialog):
    """
    Dialog window for starting a new production shift.
    
    Collects essential shift information:
    - Shift number for identification
    - Number of operators working
    - Designed cycle time (target time per part)
    - Planned operation time (shift duration)
    
    This data is used for calculating OEE metrics and production statistics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start New Shift")
        self.setModal(True)  # Block interaction with main window until closed
        self.resize(300, 200)
        
        layout = QFormLayout(self)
        
        self.shift_number = QSpinBox()
        self.shift_number.setRange(1, 999)
        self.shift_number.setValue(1)
        layout.addRow("Shift Number:", self.shift_number)
        
        self.operators_count = QSpinBox()
        self.operators_count.setRange(1, 20)
        self.operators_count.setValue(1)
        layout.addRow("Number of Operators:", self.operators_count)
        
        self.designed_cycle_time = QDoubleSpinBox()
        self.designed_cycle_time.setRange(1.0, 300.0)
        self.designed_cycle_time.setValue(30.0)
        self.designed_cycle_time.setSuffix(" seconds")
        layout.addRow("Designed Cycle Time:", self.designed_cycle_time)
        
        self.operation_time = QDoubleSpinBox()
        self.operation_time.setRange(1.0, 24.0)
        self.operation_time.setValue(8.0)
        self.operation_time.setSuffix(" hours")
        layout.addRow("Planned Operation Time:", self.operation_time)
        
        buttons_layout = QHBoxLayout()
        
        start_button = QPushButton("Start Shift")
        start_button.clicked.connect(self.accept)
        buttons_layout.addWidget(start_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        layout.addRow(buttons_layout)

class SettingsDialog(QDialog):
    def __init__(self, parent=None, current_threshold=0.25):
        super().__init__(parent)
        self.setWindowTitle("Detection Settings")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QFormLayout(self)
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.1, 0.9)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setDecimals(2)
        self.confidence_threshold.setValue(current_threshold)
        layout.addRow("Confidence Threshold:", self.confidence_threshold)
        
        buttons_layout = QHBoxLayout()
        
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        buttons_layout.addWidget(apply_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        layout.addRow(buttons_layout)

class DowntimeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Downtime")
        self.setModal(True)
        self.resize(350, 200)
        
        layout = QFormLayout(self)
        
        self.reason = QLineEdit()
        self.reason.setPlaceholderText("e.g., Machine maintenance, Material shortage")
        layout.addRow("Reason:", self.reason)
        
        self.description = QTextEdit()
        self.description.setMaximumHeight(80)
        self.description.setPlaceholderText("Additional details (optional)")
        layout.addRow("Description:", self.description)
        
        buttons_layout = QHBoxLayout()
        
        log_button = QPushButton("Log Downtime")
        log_button.clicked.connect(self.accept)
        buttons_layout.addWidget(log_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(cancel_button)
        
        layout.addRow(buttons_layout)

class MainWindow(QMainWindow):
    """
    This is the primary GUI class that coordinates all system components:
    - Camera simulation for image acquisition
    - AI model management and inference
    - Production tracking and OEE calculations
    - Database operations for persistent storage
    - Real-time display of metrics and statistics
    
    The window is organized into three main panels:
    - Left: Camera view with detection results overlay
    - Middle: Controls and detection information
    - Right: Statistics, timers, and shift management
    """
    def __init__(self):
        super().__init__()
        logger.info("Initializing GUI application")
        
        # Initialize all system components in proper order
        self.init_components()  # Create backend objects
        self.init_ui()          # Build the user interface
        self.init_timer()       # Start update timers
        self.connect_signals()  # Wire up event handlers
        
        # Timing and state tracking variables
        self.cycle_start_time = None        # When current cycle began
        self.last_cycle_end_time = None     # When previous cycle completed (for real cycle time)
        self.current_downtime_id = None     # Active downtime event ID in database
        self.confidence_threshold = 0.25    # Minimum confidence for valid detections
        self.downtime_start_time = None     # When current downtime began
        self.model_loading_worker = None    # Background thread for model loading
        
        logger.info("GUI application initialized successfully")
        
    def init_components(self):
        """
        Initialize all backend components that power the application.
        
        Creates instances of:
        - CameraOffline: Simulates camera by cycling through test images
        - ModelManager: Handles loading and managing AI models (YOLO/RF-DETR)
        - InferenceEngine: Runs detection inference with detailed timing
        - DatabaseManager: Manages SQLite database for production data
        
        These components communicate via PyQt signals/slots for loose coupling.
        """
        self.camera = CameraOffline()                                    # Image acquisition simulation
        self.model_manager = ModelManager()                              # AI model management
        self.inference_engine = InferenceEngine(self.model_manager)     # Detection processing
        self.database = DatabaseManager()                               # Data persistence
        
    def init_ui(self):
        self.setWindowTitle("Defect Detection System")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1200, 700)  # Flexible minimum size
        
        # Clean Industrial Theme - Minimal Colors, High Contrast
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                border: 2px solid #ffffff;
                border-radius: 5px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                background-color: #3c3c3c;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: 2px solid #cccccc;
                border-radius: 5px;
                padding: 12px 20px;
                font-weight: bold;
                font-size: 14px;
                color: #ffffff;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
                border-color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
            QPushButton#triggerButton {
                background-color: #2d5a2d;
                border: 3px solid #ffffff;
                font-size: 18px;
                font-weight: bold;
                min-height: 40px;
                color: #ffffff;
            }
            QPushButton#triggerButton:hover {
                background-color: #3d6a3d;
            }
            QPushButton#triggerButton:pressed {
                background-color: #1d4a1d;
            }
            QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            QLCDNumber {
                background-color: #1a1a1a;
                color: #00ff00;
                border: 2px solid #cccccc;
                border-radius: 5px;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
                background-color: #1a1a1a;
                font-weight: bold;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #4a7c4a;
                border-radius: 3px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Left panel - Image display (prioritized)
        left_panel = self.create_image_panel()
        main_layout.addWidget(left_panel, 3)  # More space for image
        
        # Middle panel - Controls and detection results
        middle_panel = self.create_controls_panel()
        main_layout.addWidget(middle_panel, 1)
        
        # Right panel - Statistics and timers
        right_panel = self.create_statistics_panel()
        main_layout.addWidget(right_panel, 1)
        
    def create_image_panel(self):
        """Create the main image display panel - maximum size for defect visibility"""
        image_widget = QWidget()
        # Let panel size itself appropriately
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(5, 5, 5, 5)  # Minimal margins
        
        # Large image display area
        image_group = QGroupBox("CAMERA VIEW")
        image_layout_inner = QVBoxLayout(image_group)
        image_layout_inner.setContentsMargins(10, 10, 10, 10)  # Minimal padding
        
        # Maximum size image display
        self.image_label = QLabel("Start New Shift.\nTrigger Detection to See Results.")
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                background-color: #1a1a1a;
                color: #cccccc;
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                padding: 20px;
            }
        """)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setMinimumSize(800, 600)  # Minimum size for visibility, can grow as needed
        image_layout_inner.addWidget(self.image_label)
        
        image_layout.addWidget(image_group)
        return image_widget
    
    def create_controls_panel(self):
        """Create the middle controls and detection results panel"""
        controls_widget = QWidget()
        # Let panel size itself based on content
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        
        # Model and settings controls
        setup_group = QGroupBox("SETUP AND CONFIGURATION")
        setup_layout = QGridLayout(setup_group)
        setup_layout.setContentsMargins(15, 10, 15, 15)
        setup_layout.setSpacing(10)
        
        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_model)
        setup_layout.addWidget(self.load_model_button, 0, 0)
        
        
        controls_layout.addWidget(setup_group)
        
        # Detection Results Panel
        results_group = QGroupBox("DETECTION RESULTS")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(15, 10, 15, 15)
        
        
        # Status display
        status_layout = QGridLayout()
        status_layout.setSpacing(8)
        
        # Status
        status_label = QLabel("Status:")
        status_label.setStyleSheet("font-size: 16px; color: #ffffff;")
        status_layout.addWidget(status_label, 0, 0)
        self.detection_status_label = QLabel("No Model Loaded")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff8800;")
        status_layout.addWidget(self.detection_status_label, 0, 1)
        
        
        results_layout.addLayout(status_layout)
        
        # Individual detections list
        detections_label = QLabel("Individual Detections:")
        detections_label.setStyleSheet("font-size: 16px; color: #ffffff;")
        detections_label.setStyleSheet("""
            QLabel {
                font-weight: bold; 
                margin: 15px 0 5px 0;
                font-size: 14px;
            }
        """)
        results_layout.addWidget(detections_label)
        
        self.detections_list = QTextEdit()
        self.detections_list.setMinimumHeight(150)  # More space for detection details
        self.detections_list.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 2px solid #cccccc;
                border-radius: 5px;
                font-size: 16px;
                padding: 5px;
            }
        """)
        self.detections_list.setPlainText("No detections yet.")
        results_layout.addWidget(self.detections_list)
        
        controls_layout.addWidget(results_group)
        
        # Main TRIGGER button
        self.trigger_button = QPushButton("TRIGGER DETECTION")
        self.trigger_button.setObjectName("triggerButton")
        self.trigger_button.clicked.connect(self.trigger_detection)
        controls_layout.addWidget(self.trigger_button)
        
        # Add stretch to push everything up
        controls_layout.addStretch()
        
        return controls_widget
        
    def create_statistics_panel(self):
        """Create the right statistics and timers panel"""
        stats_widget = QWidget()
        # Let panel size itself based on content
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        # Combined Timers - shift time at top, cycle and downtime below
        timers_group = QGroupBox("TIMERS")
        timers_layout = QVBoxLayout(timers_group)
        timers_layout.setContentsMargins(15, 10, 15, 15)
        
        # Shift Time
        shift_time_layout = QHBoxLayout()
        shift_time_layout.setContentsMargins(0, 0, 0, 0)
        shift_time_layout.setSpacing(10)
        
        shift_time_label = QLabel("SHIFT TIME:")
        shift_time_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #ffffff;")
        shift_time_layout.addStretch()  # Center content
        shift_time_layout.addWidget(shift_time_label)
        
        self.shift_time_display = QLabel("0:00")
        self.shift_time_display.setStyleSheet("""
            QLabel {
                font-family: 'Consolas', 'Monaco', 'Lucida Console', 'Liberation Mono', 'DejaVu Sans Mono', monospace;
                font-size: 28px;
                font-weight: normal;
                color: #ffffff;
                padding: 0px;
                margin: 0px;
            }
        """)
        shift_time_layout.addStretch()  # Center content
        shift_time_layout.addWidget(self.shift_time_display)
        shift_time_layout.addStretch()  # Center content
        
        timers_layout.addLayout(shift_time_layout)
        
        # Cycle Timer
        cycle_label = QLabel("Cycle Timer")
        cycle_label.setStyleSheet("font-size: 16px; color: #ffffff; margin-top: 10px;")
        timers_layout.addWidget(cycle_label)
        
        self.cycle_timer_lcd = QLCDNumber(8)
        self.cycle_timer_lcd.setDigitCount(8)
        self.cycle_timer_lcd.display("00:00.00")
        timers_layout.addWidget(self.cycle_timer_lcd)
        
        # Downtime Timer
        downtime_label = QLabel("Downtime Timer")
        downtime_label.setStyleSheet("font-size: 16px; color: #ffffff; margin-top: 10px;")
        timers_layout.addWidget(downtime_label)
        
        self.downtime_timer_lcd = QLCDNumber(8)
        self.downtime_timer_lcd.setDigitCount(8)
        self.downtime_timer_lcd.display("00:00.00")
        self.downtime_timer_lcd.setStyleSheet("QLCDNumber { color: #ff8800; }")  # Keep orange for downtime distinction
        timers_layout.addWidget(self.downtime_timer_lcd)
        
        stats_layout.addWidget(timers_group)
        
        # OEE Metrics
        oee_group = QGroupBox("OEE METRICS")
        oee_layout = QGridLayout(oee_group)
        oee_layout.setContentsMargins(15, 10, 15, 15)
        oee_label_style = "font-size: 16px; color: #ffffff;"
        oee_value_style = "font-size: 16px; color: #ffffff;"
        
        # Availability
        availability_label = QLabel("Availability:")
        availability_label.setStyleSheet(oee_label_style)
        oee_layout.addWidget(availability_label, 0, 0)
        self.availability_bar = QProgressBar()
        self.availability_bar.setRange(0, 100)
        self.availability_bar.setValue(0)
        self.availability_bar.setFormat("")
        oee_layout.addWidget(self.availability_bar, 0, 1)
        self.availability_label = QLabel("0.0%")
        self.availability_label.setStyleSheet(oee_value_style)
        oee_layout.addWidget(self.availability_label, 0, 2)
        
        # Performance
        performance_label = QLabel("Performance:")
        performance_label.setStyleSheet(oee_label_style)
        oee_layout.addWidget(performance_label, 1, 0)
        self.performance_bar = QProgressBar()
        self.performance_bar.setRange(0, 100)
        self.performance_bar.setValue(0)
        self.performance_bar.setFormat("")
        oee_layout.addWidget(self.performance_bar, 1, 1)
        self.performance_label = QLabel("0.0%")
        self.performance_label.setStyleSheet(oee_value_style)
        oee_layout.addWidget(self.performance_label, 1, 2)
        
        # Quality
        quality_label = QLabel("Quality:")
        quality_label.setStyleSheet(oee_label_style)
        oee_layout.addWidget(quality_label, 2, 0)
        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setFormat("")
        oee_layout.addWidget(self.quality_bar, 2, 1)
        self.quality_label = QLabel("0.0%")
        self.quality_label.setStyleSheet(oee_value_style)
        oee_layout.addWidget(self.quality_label, 2, 2)
        
        # Overall OEE
        oee_layout.addWidget(QLabel("Overall OEE:"), 3, 0)
        self.oee_lcd = QLCDNumber(5)
        self.oee_lcd.setDigitCount(5)
        self.oee_lcd.display("00.0")
        oee_layout.addWidget(self.oee_lcd, 3, 1, 1, 2)
        
        stats_layout.addWidget(oee_group)
        
        # Production Statistics
        prod_stats_group = QGroupBox("PRODUCTION STATISTICS")
        prod_stats_layout = QGridLayout(prod_stats_group)
        prod_stats_layout.setContentsMargins(15, 10, 15, 15)
        prod_label_style = "font-size: 16px; color: #ffffff;"
        prod_value_style = "font-size: 16px; color: #ffffff;"
        
        # Parts Made
        parts_made_label = QLabel("Parts Made:")
        parts_made_label.setStyleSheet(prod_label_style)
        prod_stats_layout.addWidget(parts_made_label, 0, 0)
        self.parts_made_label = QLabel("0")
        self.parts_made_label.setStyleSheet(prod_value_style)
        prod_stats_layout.addWidget(self.parts_made_label, 0, 1)
        
        # Defects Found
        defects_found_label = QLabel("Defects Found:")
        defects_found_label.setStyleSheet(prod_label_style)
        prod_stats_layout.addWidget(defects_found_label, 1, 0)
        self.defects_found_label = QLabel("0")
        self.defects_found_label.setStyleSheet(prod_value_style)
        prod_stats_layout.addWidget(self.defects_found_label, 1, 1)
        
        # Defect Ratio
        defect_ratio_label = QLabel("Defect Ratio:")
        defect_ratio_label.setStyleSheet(prod_label_style)
        prod_stats_layout.addWidget(defect_ratio_label, 2, 0)
        self.defect_ratio_label = QLabel("0.0%")
        self.defect_ratio_label.setStyleSheet(prod_value_style)
        prod_stats_layout.addWidget(self.defect_ratio_label, 2, 1)
        
        # Cycle Time Average
        cycle_avg_label = QLabel("Avg Cycle Time:")
        cycle_avg_label.setStyleSheet(prod_label_style)
        prod_stats_layout.addWidget(cycle_avg_label, 3, 0)
        self.cycle_avg_label = QLabel("0.0s")
        self.cycle_avg_label.setStyleSheet(prod_value_style)
        prod_stats_layout.addWidget(self.cycle_avg_label, 3, 1)
        
        # Average Model Inference Time
        avg_inference_label = QLabel("Avg Model Inference:")
        avg_inference_label.setStyleSheet(prod_label_style)
        prod_stats_layout.addWidget(avg_inference_label, 4, 0)
        self.avg_inference_label = QLabel("0.0 ms")
        self.avg_inference_label.setStyleSheet(prod_value_style)
        prod_stats_layout.addWidget(self.avg_inference_label, 4, 1)
        
        stats_layout.addWidget(prod_stats_group)
        
        
        # Shift Information
        shift_group = QGroupBox("SHIFT MANAGEMENT")
        shift_layout = QVBoxLayout(shift_group)
        shift_layout.setContentsMargins(15, 10, 15, 15)
        
        self.shift_button = QPushButton("Start New Shift")
        self.shift_button.clicked.connect(self.start_new_shift)
        shift_layout.addWidget(self.shift_button)
        
        self.end_shift_button = QPushButton("End Current Shift")
        self.end_shift_button.clicked.connect(self.end_current_shift)
        self.end_shift_button.setStyleSheet("QPushButton { background-color: #7c4a4a; }")  # Different color to distinguish
        shift_layout.addWidget(self.end_shift_button)
        
        self.downtime_button = QPushButton("Log Downtime")
        self.downtime_button.clicked.connect(self.toggle_downtime)
        shift_layout.addWidget(self.downtime_button)
        
        stats_layout.addWidget(shift_group)
        
        # Add stretch to push everything to top
        stats_layout.addStretch()
        
        return stats_widget
    
    def init_timer(self):
        self.cycle_timer = QTimer()
        self.cycle_timer.timeout.connect(self.update_cycle_timer)
        self.cycle_timer.start(10)  # Update every 10ms
        self.cycle_start_time = time.time()
        
        # GUI update timer (for real-time display only, no OEE calculations)
        self.gui_timer = QTimer()
        self.gui_timer.timeout.connect(self.update_gui_display)
        self.gui_timer.start(1000)  # Update GUI every second
        
    def connect_signals(self):
        # Camera signals
        self.camera.image_loaded.connect(self.on_image_loaded)
        
        # Model manager signals
        self.model_manager.model_loaded.connect(self.on_model_loaded)
        self.model_manager.model_error.connect(self.on_model_error)
        self.model_manager.model_warming_started.connect(self.on_model_warming_started)
        self.model_manager.model_warming_completed.connect(self.on_model_warming_completed)
        
        # Inference engine signals
        self.inference_engine.inference_completed.connect(self.on_inference_completed)
        self.inference_engine.inference_error.connect(self.on_inference_error)
        
        # Database signals - only calculate OEE when data actually changes
        self.database.data_updated.connect(self.update_statistics_with_logging)
        self.database.error_occurred.connect(self.on_database_error)
        
    def update_cycle_timer(self):
        # Update cycle timer (only if not in downtime)
        if self.cycle_start_time is not None and self.current_downtime_id is None:
            current_cycle_time = time.time() - self.cycle_start_time
            minutes = int(current_cycle_time // 60)
            seconds = current_cycle_time % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
            self.cycle_timer_lcd.display(time_str)
        
        # Update downtime timer (only if in downtime)
        if self.downtime_start_time is not None and self.current_downtime_id is not None:
            downtime_duration = time.time() - self.downtime_start_time
            minutes = int(downtime_duration // 60)
            seconds = downtime_duration % 60
            time_str = f"{minutes:02d}:{seconds:05.2f}"
            self.downtime_timer_lcd.display(time_str)
    
    def reset_cycle_timer(self):
        self.cycle_start_time = time.time()
    
    def open_settings(self):
        dialog = SettingsDialog(self, self.confidence_threshold)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.confidence_threshold = dialog.confidence_threshold.value()
            # Update the display
            self.confidence_threshold_label.setText(f"{self.confidence_threshold}")
            self.detection_status_label.setText(f"Threshold Updated: {self.confidence_threshold}")
            self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "models", 
            "Model Files (*.pt *.pth);;All Files (*)"
        )
        
        if file_path:
            logger.info(f"User selected model file: {file_path}")
            
            # Prevent multiple concurrent loading operations
            if self.model_loading_worker is not None and self.model_loading_worker.isRunning():
                logger.warning("Model loading already in progress, ignoring new request")
                QMessageBox.warning(self, "Loading in Progress", "A model is already being loaded. Please wait.")
                return
            
            # Show loading status immediately (GUI stays responsive)
            self.detection_status_label.setText("Loading model...")
            self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff8800;")
            self.trigger_button.setEnabled(False)
            self.trigger_button.setText("LOADING...")
            self.load_model_button.setEnabled(False)  # Prevent multiple clicks
            
            logger.info("Starting model loading in background thread")
            # Start loading in worker thread
            self.model_loading_worker = ModelLoadingWorker(self.model_manager, file_path)
            self.model_loading_worker.loading_finished.connect(self.on_model_loading_finished)
            self.model_loading_worker.start()
    
    def on_model_loading_finished(self, success, file_path):
        """Handle completion of model loading worker thread"""
        # Re-enable the load model button
        self.load_model_button.setEnabled(True)
        
        if success:
            # Success handled by existing signal handlers (model_loaded, model_warming_started, etc.)
            print(f"Model loaded successfully: {file_path}")
        else:
            # Handle failure
            self.detection_status_label.setText("Model Load Failed")
            self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
            self.trigger_button.setEnabled(False)
            self.trigger_button.setText("TRIGGER DETECTION")
        
        # Clean up worker reference
        self.model_loading_worker = None
    
    def trigger_detection(self):
        # Check if there's an active shift
        current_shift = self.database.get_current_shift()
        if not current_shift:
            self.update_camera_view_message("Start New Shift.\nTrigger Detection to See Results.")
            QMessageBox.warning(self, "No Active Shift", "Please start a new shift before triggering detection.")
            return
        
        if self.model_manager.current_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded. Please load a model first.")
            return
        
        # Calculate actual cycle time (from last cycle end to this trigger)
        current_time = time.time()
        if self.last_cycle_end_time is not None:
            # This is the realistic operator cycle time
            actual_operator_cycle_time = current_time - self.last_cycle_end_time
            logger.info(f"Detection triggered - Operator cycle time: {actual_operator_cycle_time:.2f}s")
        else:
            # First cycle - no previous end time
            actual_operator_cycle_time = None
            logger.info("Detection triggered - First cycle of shift")
        
        # Store for calculation
        self.current_operator_cycle_time = actual_operator_cycle_time
        
        # Reset GUI cycle timer (for display purposes)
        self.reset_cycle_timer()
        
        # Get next test image
        if self.camera.simulate_trigger():
            current_image = self.camera.get_current_image()
            if current_image is not None:
                # Run inference with current confidence threshold
                self.inference_engine.run_inference(
                    current_image, 
                    self.camera.get_current_image_path(),
                    self.confidence_threshold
                )
            else:
                self.detection_status_label.setText("No Image Available")
                self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff8800;")
        else:
            self.detection_status_label.setText("Image Acquisition Failed")
            self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
    
    def start_new_shift(self):
        # Check if there's already an active shift
        current_shift = self.database.get_current_shift()
        if current_shift:
            # Prevent starting multiple shifts - show warning
            QMessageBox.warning(
                self, 
                "Active Shift Already Running", 
                f"Shift #{current_shift['shift_number']} is currently active.\n\n"
                f"Please end the current shift before starting a new one.\n"
                f"Use 'End Current Shift' button to close the active shift."
            )
            logger.warning(f"Attempted to start new shift while Shift #{current_shift['shift_number']} is still active")
            return
        
        dialog = ShiftDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            shift_id = self.database.start_new_shift(
                dialog.shift_number.value(),
                dialog.operators_count.value(),
                dialog.designed_cycle_time.value()
            )
            
            if shift_id > 0:
                # Reset cycle timer to zero when new shift starts
                self.cycle_start_time = None
                self.cycle_timer_lcd.display("00:00.00")
                # Reset last cycle end time so first cycle is treated as truly first
                self.last_cycle_end_time = None
                logger.info(f"New shift #{dialog.shift_number.value()} started successfully")
                self.detection_status_label.setText(f"New Shift #{dialog.shift_number.value()} Started")
                self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
                self.update_statistics()
            else:
                logger.error("Failed to start new shift")
                QMessageBox.warning(self, "Error", "Failed to start new shift")
    
    def end_current_shift(self):
        """End the current active shift"""
        # Check if there's an active shift
        current_shift = self.database.get_current_shift()
        if not current_shift:
            QMessageBox.information(self, "No Active Shift", "There is no active shift to end.")
            return
        
        # Confirm shift ending
        reply = QMessageBox.question(
            self, 
            "End Shift", 
            f"Are you sure you want to end Shift #{current_shift['shift_number']}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # End any active downtime first
            if self.current_downtime_id is not None:
                self.database.end_downtime(self.current_downtime_id)
                self.current_downtime_id = None
                self.downtime_button.setText("Log Downtime")
                self.downtime_button.setStyleSheet("")
                self.downtime_start_time = None
            
            # End the shift in database
            if self.database.end_current_shift():
                # Log shift end with final metrics
                final_stats = self.database.get_production_stats()
                final_oee = self.database.calculate_oee_metrics()
                
                logger.info(f"Shift #{current_shift['shift_number']} ended")
                logger.info(f"Final shift metrics - Parts: {final_stats.get('total_cycles', 0)}, Defects: {final_stats.get('total_defects', 0)}, OEE: {final_oee.get('oee', 0):.1%}")
                
                # Reset all timers and states
                self.cycle_start_time = None
                self.last_cycle_end_time = None
                self.cycle_timer_lcd.display("00:00.00")
                self.downtime_timer_lcd.display("00:00.00")
                
                # Update status
                self.detection_status_label.setText("Shift Ended")
                self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff8800;")
                
                # Update camera view to show message
                self.update_camera_view_message("Start New Shift.\nTrigger Detection to See Results.")
                
                # Update statistics to reflect ended shift
                self.update_statistics()
                
                print(f"Shift #{current_shift['shift_number']} ended successfully")
            else:
                QMessageBox.warning(self, "Error", "Failed to end the current shift")
    
    def toggle_downtime(self):
        """
        Handle downtime logging with immediate cycle timer stopping.
        
        This method implements a key industrial UX improvement: the cycle timer
        stops immediately when the operator indicates there's a problem, rather
        than continuing to run while they fill out the downtime form.
        
        Industrial Logic:
        1. Operator notices a problem → clicks "Log Downtime" 
        2. Cycle timer stops IMMEDIATELY (problem time = button press time)
        3. Operator fills out downtime form (cycle timer remains stopped)
        4. Form submission completes the downtime logging process
        
        This prevents artificial inflation of cycle times due to form-filling delays.
        """
        # Check if there's an active shift before allowing downtime logging
        current_shift = self.database.get_current_shift()
        if not current_shift:
            # No active shift - prevent downtime logging
            QMessageBox.warning(
                self, 
                "No Active Shift", 
                "Please start a shift first to log downtime.\n\n"
                "Downtime can only be logged during an active production shift.\n"
                "Use 'Start New Shift' button to begin a shift."
            )
            logger.warning("Attempted to log downtime without an active shift")
            return
        
        if self.current_downtime_id is None:
            # Stop cycle timer IMMEDIATELY when downtime button is pressed
            self.cycle_timer_lcd.display("00:00.00")
            self.cycle_start_time = None
            
            # Show downtime dialog for operator to provide details
            dialog = DowntimeDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                reason = dialog.reason.text().strip()
                description = dialog.description.toPlainText().strip()
                
                if reason:
                    # Log downtime to database with provided reason
                    self.current_downtime_id = self.database.start_downtime(reason, description)
                    if self.current_downtime_id > 0:
                        logger.warning(f"Downtime started: {reason}")
                        # Enter downtime mode - update UI to reflect downtime state
                        self.downtime_button.setText("End Downtime")
                        self.downtime_button.setStyleSheet("background-color: #7c4a4a;")
                        self.detection_status_label.setText(f"Downtime: {truncate_text(reason, 20)}")
                        self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff8800;")
                        
                        # Start downtime timer
                        self.downtime_start_time = time.time()
                        self.downtime_timer_lcd.display("00:00.00")
                    else:
                        # Database error - restore normal operation
                        self.cycle_start_time = time.time()
                        QMessageBox.warning(self, "Error", "Failed to start downtime")
                else:
                    # No reason provided - restore normal operation
                    self.cycle_start_time = time.time()
                    QMessageBox.warning(self, "Warning", "Please enter a reason for downtime")
            else:
                # User canceled dialog - restore normal operation
                # This handles the case where operator accidentally clicked downtime
                self.cycle_start_time = time.time()
        else:
            # End downtime
            if self.database.end_downtime(self.current_downtime_id):
                logger.info("Downtime ended - waiting for next trigger detection to resume cycle timing")
                # End downtime mode
                self.downtime_button.setText("Log Downtime")
                self.downtime_button.setStyleSheet("")
                self.detection_status_label.setText("Downtime Ended - Ready for Detection")
                self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
                
                # Reset downtime timer but don't restart cycle timer
                # Cycle will start when user manually triggers detection
                self.downtime_timer_lcd.display("00:00.00")
                self.downtime_start_time = None  # Stop downtime timer
                self.cycle_start_time = None  # Don't auto-start cycle - wait for manual trigger
                self.current_downtime_id = None
                self.update_statistics()
            else:
                QMessageBox.warning(self, "Error", "Failed to end downtime")
    
    def on_image_loaded(self, image_path, image_array):
        # Check if there's an active shift before displaying image
        current_shift = self.database.get_current_shift()
        if not current_shift:
            # No active shift - show message instead of image
            self.update_camera_view_message("Start New Shift.\nTrigger Detection to See Results.")
            return
        
        # Display image in GUI
        pixmap = self.camera.get_current_qpixmap()
        if pixmap:
            # Scale to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
        
        # Update status in detection panel
        self.detection_status_label.setText("Image Loaded")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
    
    def on_model_loaded(self, model_path, model_type):
        self.detection_status_label.setText(f"Model: {truncate_text(model_type.upper(), 25)}")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
    
    def on_model_error(self, error_message):
        self.detection_status_label.setText("Model Error")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
        QMessageBox.critical(self, "Model Error", error_message)
    
    def on_model_warming_started(self):
        self.detection_status_label.setText("Warming up model...")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff8800;")
        # Disable trigger button during warm-up
        self.trigger_button.setEnabled(False)
        self.trigger_button.setText("WARMING UP...")
    
    def on_model_warming_completed(self):
        self.detection_status_label.setText("Ready")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        # Re-enable trigger button after warm-up
        self.trigger_button.setEnabled(True)
        self.trigger_button.setText("TRIGGER DETECTION")
    
    def on_inference_completed(self, results):
        # Mark cycle end time for next cycle calculation
        cycle_end_time = time.time()
        self.last_cycle_end_time = cycle_end_time
        
        # Check if this is a real cycle (not the first baseline cycle)
        is_real_cycle = hasattr(self, 'current_operator_cycle_time') and self.current_operator_cycle_time is not None
        
        if is_real_cycle:
            cycle_time_to_use = self.current_operator_cycle_time
            print(f"Real cycle - using operator cycle time: {cycle_time_to_use:.3f}s")
            
            # Add the cycle time to results for database logging
            if 'timing' not in results:
                results['timing'] = {}
            results['timing']['gui_cycle_time'] = cycle_time_to_use
        else:
            print("First cycle - using designed cycle time for quality tracking")
            # For first cycle, use designed cycle time from shift settings
            current_shift = self.database.get_current_shift()
            if current_shift:
                designed_cycle_time = current_shift.get('designed_cycle_time', 30.0)
                if 'timing' not in results:
                    results['timing'] = {}
                results['timing']['gui_cycle_time'] = designed_cycle_time
                print(f"First cycle - using designed cycle time: {designed_cycle_time:.1f}s for database logging")
        
        # Also calculate trigger-to-completion time for reference
        if self.cycle_start_time is not None:
            processing_time = cycle_end_time - self.cycle_start_time
            results['timing']['processing_time'] = processing_time
            print(f"Processing time (trigger to completion): {processing_time:.3f}s")
        
        # Display annotated image
        if 'annotated_image' in results:
            annotated_image = results['annotated_image']
            # Convert numpy array to QPixmap
            pixmap = self.camera.numpy_to_qpixmap(annotated_image)
            if pixmap:
                scaled_pixmap = pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
        
        # Update detection results panel
        self.update_detection_results(results)
        
        # Log to database (both real cycles and first cycle with designed timing)
        if 'gui_cycle_time' in results.get('timing', {}):
            self.database.log_production_cycle(results)
            if is_real_cycle:
                print("Real cycle logged to database")
            else:
                print("First cycle logged to database with designed cycle time")
        else:
            print("Cycle not logged - no timing data available")
        
        # Statistics are automatically updated via database signal (data_updated)
    
    def update_detection_results(self, results):
        """Update the new detection results panel"""
        summary = results.get('detection_summary', {})
        detections = results.get('detections', [])
        
        # Update status
        quality_status = summary.get('quality_status', 'Unknown')
        if summary.get('has_defects', False):
            self.detection_status_label.setText("DEFECT DETECTED")
            self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
        else:
            self.detection_status_label.setText("GOOD")
            self.detection_status_label.setStyleSheet("font-weight: bold; color: #00ff00;")
        
        # Update individual detections list (only class and confidence)
        detections_text = ""
        if detections:
            for i, det in enumerate(detections, 1):
                class_name = det.get('class_name', 'unknown')
                confidence = det.get('confidence', 0.0)
                detections_text += f"{i}. {class_name} (conf: {confidence:.3f})\n"
        else:
            detections_text = "No detections found"
        
        self.detections_list.setPlainText(detections_text)
    
    
    def on_inference_error(self, error_message):
        self.detection_status_label.setText("Inference Error")
        self.detection_status_label.setStyleSheet("font-weight: bold; color: #ff0000;")
        self.detections_list.setPlainText("Inference failed")
        QMessageBox.critical(self, "Inference Error", error_message)
    
    def on_database_error(self, error_message):
        print(f"Database error: {error_message}")
    
    def update_camera_view_message(self, message):
        """Display a message in the camera view instead of an image"""
        self.image_label.clear()  # Clear any existing image
        self.image_label.setText(message)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def update_statistics(self):
        # Update OEE metrics
        oee_metrics = self.database.calculate_oee_metrics()
        
        # Update availability
        availability = oee_metrics.get('availability', 0) * 100
        self.availability_bar.setValue(int(availability))
        self.availability_label.setText(f"{availability:.1f}%")
        
        # Update performance
        performance = oee_metrics.get('performance', 0) * 100
        self.performance_bar.setValue(int(performance))
        self.performance_label.setText(f"{performance:.1f}%")
        
        # Update quality
        quality = oee_metrics.get('quality', 0) * 100
        self.quality_bar.setValue(int(quality))
        self.quality_label.setText(f"{quality:.1f}%")
        
        # Update OEE
        oee = oee_metrics.get('oee', 0) * 100
        self.oee_lcd.display(f"{oee:.1f}")
        
        # Update production statistics
        stats = self.database.get_production_stats()
        
        self.parts_made_label.setText(str(stats.get('total_cycles', 0)))
        self.defects_found_label.setText(str(stats.get('total_defects', 0)))
        
        defect_ratio = stats.get('defect_ratio', 0) * 100
        self.defect_ratio_label.setText(f"{defect_ratio:.1f}%")
        
        avg_cycle_time = stats.get('avg_cycle_time', 0)
        self.cycle_avg_label.setText(f"{avg_cycle_time:.1f}s")
        
        # Update average inference time
        avg_inference_time = stats.get('avg_inference_time', 0) * 1000  # Convert to ms
        self.avg_inference_label.setText(f"{avg_inference_time:.1f} ms")
        
        # Update shift elapsed time in H:M format
        current_shift = self.database.get_current_shift()
        if current_shift:
            elapsed_seconds = time.time() - current_shift['start_time']
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            self.shift_time_display.setText(f"{hours}:{minutes:02d}")
        else:
            self.shift_time_display.setText("0:00")
    
    def update_gui_display(self):
        """
        Update GUI elements with cached values - no OEE calculations or logging.
        This runs on a timer for real-time display updates.
        """
        # Get cached metrics from last calculation (no logging)
        current_shift = self.database.get_current_shift()
        if current_shift:
            # Update shift elapsed time display only
            elapsed_seconds = time.time() - current_shift['start_time']
            hours = int(elapsed_seconds // 3600)
            minutes = int((elapsed_seconds % 3600) // 60)
            self.shift_time_display.setText(f"{hours}:{minutes:02d}")
        else:
            self.shift_time_display.setText("0:00")
    
    def update_statistics_with_logging(self):
        """
        Calculate and log OEE metrics when production data changes.
        This is event-driven and includes logging.
        """
        # This triggers the full OEE calculation with logging
        self.update_statistics()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())