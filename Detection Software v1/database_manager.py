#!/usr/bin/env python3
"""
Database Manager - Production Data Management for Dana Heat Shield GUI

This module handles all database operations for the industrial defect detection system.
It provides a comprehensive data model for tracking production metrics, shift information,
detection results, and downtime events using SQLite for reliable local storage.

Database Schema:
- production_cycles: Individual detection cycles with timing and results
- shifts: Work shift information with operators and designed cycle times
- downtime_events: Machine downtime tracking with reasons and durations
- detection_details: Detailed bounding box and confidence data for each detection
- performance_metrics: Cached OEE calculations for performance optimization

Key Features:
- Real-time OEE (Overall Equipment Effectiveness) calculation
- Production statistics with defect ratios and cycle times
- Shift management with operator tracking
- Downtime event logging with duration calculation
- Comprehensive data export capabilities
- Signal-based notifications for GUI updates

OEE Calculation:
- Availability = (Total Time - Downtime) / Total Time
- Performance = Designed Cycle Time / Actual Average Cycle Time 
- Quality = Good Parts / Total Parts Made
- OEE = Availability × Performance × Quality

Date: 2025-07-29
Version: 1.0
"""

import sqlite3
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from logger_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

class DatabaseManager(QObject):
    """
    Manages all database operations for production data tracking and OEE calculation.
    
    This class provides a complete data management solution for industrial production
    monitoring. It handles everything from basic CRUD operations to complex OEE
    calculations and performance metrics.
    
    Signals:
        data_updated(): Emitted when data changes require GUI refresh
        error_occurred(str): Emitted when database errors occur
    
    The database uses SQLite for reliability and simplicity, with a normalized
    schema that supports both real-time operations and historical analysis.
    """
    data_updated = pyqtSignal()      # Signal when data changes
    error_occurred = pyqtSignal(str) # Signal when errors occur
    
    def __init__(self, db_path: str = "production_data.db"):
        super().__init__()
        self.db_path = db_path
        logger.info(f"DatabaseManager initializing with database: {db_path}")
        # Initialize database schema on startup
        self.init_database()
    
    def init_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Production cycles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS production_cycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        cycle_time REAL NOT NULL,
                        image_path TEXT,
                        model_type TEXT,
                        total_detections INTEGER DEFAULT 0,
                        defect_count INTEGER DEFAULT 0,
                        good_count INTEGER DEFAULT 0,
                        has_defects BOOLEAN DEFAULT 0,
                        quality_status TEXT,
                        inference_time REAL,
                        shift_id INTEGER,
                        FOREIGN KEY (shift_id) REFERENCES shifts (id)
                    )
                ''')
                
                # Shifts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS shifts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        shift_number INTEGER NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        operators_count INTEGER DEFAULT 1,
                        designed_cycle_time REAL DEFAULT 30.0,
                        total_operation_time REAL DEFAULT 0,
                        is_active BOOLEAN DEFAULT 1
                    )
                ''')
                
                # Downtime events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS downtime_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_time REAL NOT NULL,
                        end_time REAL,
                        duration REAL,
                        reason TEXT,
                        description TEXT,
                        shift_id INTEGER,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (shift_id) REFERENCES shifts (id)
                    )
                ''')
                
                # Detection details table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detection_details (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle_id INTEGER NOT NULL,
                        class_name TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        bbox_x1 REAL,
                        bbox_y1 REAL,
                        bbox_x2 REAL,
                        bbox_y2 REAL,
                        FOREIGN KEY (cycle_id) REFERENCES production_cycles (id)
                    )
                ''')
                
                # Performance metrics table (for caching OEE calculations)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        shift_id INTEGER,
                        availability REAL DEFAULT 0,
                        performance REAL DEFAULT 0,
                        quality REAL DEFAULT 0,
                        oee REAL DEFAULT 0,
                        parts_made INTEGER DEFAULT 0,
                        defects_found INTEGER DEFAULT 0,
                        total_downtime REAL DEFAULT 0,
                        avg_cycle_time REAL DEFAULT 0,
                        FOREIGN KEY (shift_id) REFERENCES shifts (id)
                    )
                ''')
                
                conn.commit()
                logger.info(f"Database schema initialized successfully: {self.db_path}")
                
        except Exception as e:
            error_msg = f"Database initialization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
    
    def start_new_shift(self, shift_number: int, operators_count: int = 1, 
                       designed_cycle_time: float = 30.0) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # End any active shifts
                cursor.execute('''
                    UPDATE shifts 
                    SET end_time = ?, is_active = 0 
                    WHERE is_active = 1
                ''', (time.time(),))
                
                # Start new shift
                cursor.execute('''
                    INSERT INTO shifts (shift_number, start_time, operators_count, designed_cycle_time)
                    VALUES (?, ?, ?, ?)
                ''', (shift_number, time.time(), operators_count, designed_cycle_time))
                
                shift_id = cursor.lastrowid
                conn.commit()
                
                self.data_updated.emit()
                print(f"Started new shift: {shift_number} (ID: {shift_id})")
                return shift_id
                
        except Exception as e:
            error_msg = f"Failed to start new shift: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return -1
    
    def end_current_shift(self) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE shifts 
                    SET end_time = ?, is_active = 0 
                    WHERE is_active = 1
                ''', (time.time(),))
                
                conn.commit()
                self.data_updated.emit()
                print("Current shift ended")
                return True
                
        except Exception as e:
            error_msg = f"Failed to end current shift: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def get_current_shift(self) -> Optional[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM shifts WHERE is_active = 1 LIMIT 1
                ''')
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            error_msg = f"Failed to get current shift: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return None
    
    def log_production_cycle(self, cycle_data: Dict[str, Any]) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current shift
                current_shift = self.get_current_shift()
                shift_id = current_shift['id'] if current_shift else None
                
                # Extract detection summary
                summary = cycle_data.get('detection_summary', {})
                timing = cycle_data.get('timing', {})
                
                # Insert production cycle - only if we have real cycle time
                # Don't log incomplete cycles that would corrupt OEE calculations
                if 'gui_cycle_time' not in timing:
                    logger.warning("No real cycle time available - skipping database logging to preserve OEE accuracy")
                    return -1  # Don't save meaningless data
                
                cycle_time = timing['gui_cycle_time']  # Only use real operator cycle time
                inference_time = timing.get('total_time', 0)
                
                cursor.execute('''
                    INSERT INTO production_cycles (
                        timestamp, cycle_time, image_path, model_type,
                        total_detections, defect_count, good_count, has_defects,
                        quality_status, inference_time, shift_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cycle_data.get('timestamp', time.time()),
                    cycle_time,
                    cycle_data.get('image_path', ''),
                    cycle_data.get('model_type', ''),
                    summary.get('total_detections', 0),
                    summary.get('defect_count', 0),
                    summary.get('good_count', 0),
                    summary.get('has_defects', False),
                    summary.get('quality_status', ''),
                    inference_time,
                    shift_id
                ))
                
                cycle_id = cursor.lastrowid
                
                # Insert detection details
                detections = cycle_data.get('detections', [])
                for det in detections:
                    bbox = det.get('bbox', [0, 0, 0, 0])
                    cursor.execute('''
                        INSERT INTO detection_details (
                            cycle_id, class_name, confidence,
                            bbox_x1, bbox_y1, bbox_x2, bbox_y2
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        cycle_id,
                        det.get('class_name', 'unknown'),
                        det.get('confidence', 0.0),
                        bbox[0] if len(bbox) > 0 else 0,
                        bbox[1] if len(bbox) > 1 else 0,
                        bbox[2] if len(bbox) > 2 else 0,
                        bbox[3] if len(bbox) > 3 else 0
                    ))
                
                conn.commit()
                self.data_updated.emit()
                
                print(f"Production cycle logged (ID: {cycle_id})")
                return cycle_id
                
        except Exception as e:
            error_msg = f"Failed to log production cycle: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return -1
    
    def start_downtime(self, reason: str, description: str = "") -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                current_shift = self.get_current_shift()
                shift_id = current_shift['id'] if current_shift else None
                
                cursor.execute('''
                    INSERT INTO downtime_events (start_time, reason, description, shift_id)
                    VALUES (?, ?, ?, ?)
                ''', (time.time(), reason, description, shift_id))
                
                downtime_id = cursor.lastrowid
                conn.commit()
                
                self.data_updated.emit()
                print(f"Downtime started: {reason} (ID: {downtime_id})")
                return downtime_id
                
        except Exception as e:
            error_msg = f"Failed to start downtime: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return -1
    
    def end_downtime(self, downtime_id: int) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                end_time = time.time()
                
                # Get start time to calculate duration
                cursor.execute('SELECT start_time FROM downtime_events WHERE id = ?', (downtime_id,))
                row = cursor.fetchone()
                if not row:
                    return False
                
                start_time = row[0]
                duration = end_time - start_time
                
                cursor.execute('''
                    UPDATE downtime_events 
                    SET end_time = ?, duration = ?, is_active = 0
                    WHERE id = ?
                ''', (end_time, duration, downtime_id))
                
                conn.commit()
                self.data_updated.emit()
                
                print(f"Downtime ended (ID: {downtime_id}, Duration: {duration:.2f}s)")
                return True
                
        except Exception as e:
            error_msg = f"Failed to end downtime: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def calculate_oee_metrics(self, shift_id: Optional[int] = None, 
                             time_window_hours: float = 8.0) -> Dict[str, float]:
        """
        Calculate Overall Equipment Effectiveness (OEE) metrics for production analysis.
        
        OEE is the gold standard for measuring manufacturing productivity. It combines
        three key factors:
        
        1. Availability: Percentage of scheduled time that equipment is available
           Formula: (Planned Production Time - Downtime) / Planned Production Time
           
        2. Performance: How efficiently equipment runs during available time
           Formula: (Designed Cycle Time × Parts Produced) / Operating Time
           Simplified: Average(Designed Cycle Time / Actual Cycle Time) per part
           
        3. Quality: Percentage of parts produced that meet quality standards
           Formula: Good Parts / Total Parts Produced
        
        Overall OEE = Availability × Performance × Quality
        
        Args:
            shift_id: Specific shift to calculate (None for current active shift)
            time_window_hours: Time window for calculation (currently unused)
            
        Returns:
            Dict containing availability, performance, quality, oee (0.0-1.0 scale)
            Plus additional metrics like parts counts and timing data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get shift info first
                if shift_id:
                    cursor.execute('SELECT * FROM shifts WHERE id = ?', (shift_id,))
                else:
                    cursor.execute('SELECT * FROM shifts WHERE is_active = 1 LIMIT 1')
                
                shift = cursor.fetchone()
                if not shift:
                    return {'availability': 0, 'performance': 0, 'quality': 0, 'oee': 0}
                
                # Use the shift's actual start time, not an arbitrary time window
                shift_start_time = shift[2]  # start_time column
                designed_cycle_time = shift[5]  # designed_cycle_time column (corrected index)
                
                print(f"Calculating OEE for shift from timestamp {shift_start_time} (shift start time)")
                
                # Calculate Availability
                cursor.execute('''
                    SELECT COALESCE(SUM(duration), 0) as total_downtime
                    FROM downtime_events 
                    WHERE shift_id = ? AND start_time >= ?
                ''', (shift[0], shift_start_time))
                
                total_downtime = cursor.fetchone()[0]
                current_time = time.time()
                total_shift_time = current_time - shift_start_time
                planned_production_time = max(total_shift_time, 1)  # Avoid division by zero
                availability = max(0, (planned_production_time - total_downtime) / planned_production_time)
                
                logger.info(f"Calculated availability: {availability:.1%} (downtime: {total_downtime:.1f}s of {total_shift_time:.1f}s)")
                
                # Calculate Performance using individual cycle method (more accurate)
                # Traditional OEE performance calculation uses: (Ideal Cycle Time × Total Count) / Operating Time
                # However, for more accurate results with variable cycle times, we calculate
                # performance for each individual cycle then average the results
                cursor.execute('''
                    SELECT cycle_time
                    FROM production_cycles 
                    WHERE shift_id = ? AND cycle_time > 0
                ''', (shift[0],))
                
                cycle_times = cursor.fetchall()
                cycles_count = len(cycle_times)
                
                if cycles_count > 0:
                    # Calculate performance for each individual cycle, then average
                    individual_performances = []
                    total_cycle_time = 0
                    
                    for (cycle_time,) in cycle_times:
                        total_cycle_time += cycle_time
                        # Performance for this cycle = designed / actual (capped at 100%)
                        cycle_performance = min(1.0, designed_cycle_time / cycle_time)
                        individual_performances.append(cycle_performance)
                    
                    # Average all individual cycle performances
                    performance = sum(individual_performances) / len(individual_performances)
                    avg_cycle_time = total_cycle_time / cycles_count
                    
                    logger.debug(f"Performance calculation: designed={designed_cycle_time:.2f}s, cycles={cycles_count}, avg_actual={avg_cycle_time:.2f}s")
                    logger.debug(f"Individual performances sample: {[f'{p:.3f}' for p in individual_performances[:5]]}{'...' if len(individual_performances) > 5 else ''}")
                    logger.info(f"Calculated performance: {performance:.1%} (average of {cycles_count} cycles)")
                else:
                    performance = 1.0  # Default to 100% if no cycles yet (assume perfect until proven otherwise)
                    avg_cycle_time = 0  # Default to 0 if no cycles yet
                    
                    logger.info(f"Calculated performance: {performance:.1%} (no production cycles yet - using default)")
                
                # Calculate Quality
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_parts,
                        SUM(CASE WHEN has_defects = 0 THEN 1 ELSE 0 END) as good_parts,
                        SUM(defect_count) as total_defects
                    FROM production_cycles 
                    WHERE shift_id = ? AND timestamp >= ?
                ''', (shift[0], shift_start_time))
                
                quality_data = cursor.fetchone()
                total_parts = quality_data[0] or 0
                good_parts = quality_data[1] or 0
                total_defects = quality_data[2] or 0
                
                quality = good_parts / total_parts if total_parts > 0 else 1.0
                
                logger.info(f"Calculated quality: {quality:.1%} ({good_parts} good parts of {total_parts} total)")
                
                # Calculate OEE
                oee = availability * performance * quality
                
                logger.info(f"Calculated OEE: {oee:.1%} (A:{availability:.1%} × P:{performance:.1%} × Q:{quality:.1%})")
                
                metrics = {
                    'availability': availability,
                    'performance': performance,
                    'quality': quality,
                    'oee': oee,
                    'total_parts': total_parts,
                    'good_parts': good_parts,
                    'total_defects': total_defects,
                    'total_downtime': total_downtime,
                    'avg_cycle_time': avg_cycle_time,
                    'designed_cycle_time': designed_cycle_time
                }
                
                # Cache the metrics
                cursor.execute('''
                    INSERT INTO performance_metrics (
                        timestamp, shift_id, availability, performance, quality, oee,
                        parts_made, defects_found, total_downtime, avg_cycle_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    time.time(), shift[0], availability, performance, quality, oee,
                    total_parts, total_defects, total_downtime, avg_cycle_time
                ))
                
                conn.commit()
                return metrics
                
        except Exception as e:
            error_msg = f"Failed to calculate OEE metrics: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return {'availability': 0, 'performance': 0, 'quality': 0, 'oee': 0}
    
    def get_production_stats(self, shift_id: Optional[int] = None) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get current or specified shift
                if shift_id:
                    cursor.execute('SELECT * FROM shifts WHERE id = ?', (shift_id,))
                else:
                    cursor.execute('SELECT * FROM shifts WHERE is_active = 1 LIMIT 1')
                
                shift = cursor.fetchone()
                if not shift:
                    return {}
                
                shift_id = shift[0]
                
                # Production statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_cycles,
                        SUM(CASE WHEN has_defects = 0 THEN 1 ELSE 0 END) as good_parts,
                        SUM(defect_count) as total_defects,
                        AVG(cycle_time) as avg_cycle_time,
                        MIN(cycle_time) as min_cycle_time,
                        MAX(cycle_time) as max_cycle_time,
                        AVG(inference_time) as avg_inference_time
                    FROM production_cycles 
                    WHERE shift_id = ?
                ''', (shift_id,))
                
                stats = cursor.fetchone()
                
                total_cycles = stats[0] or 0
                good_parts = stats[1] or 0
                total_defects = stats[2] or 0
                avg_cycle_time = stats[3] or 0
                min_cycle_time = stats[4] or 0
                max_cycle_time = stats[5] or 0
                avg_inference_time = stats[6] or 0
                
                defect_ratio = (total_defects / total_cycles) if total_cycles > 0 else 0
                
                return {
                    'shift_id': shift_id,
                    'shift_number': shift[1],
                    'total_cycles': total_cycles,
                    'good_parts': good_parts,
                    'total_defects': total_defects,
                    'defect_ratio': defect_ratio,
                    'avg_cycle_time': avg_cycle_time,
                    'min_cycle_time': min_cycle_time,
                    'max_cycle_time': max_cycle_time,
                    'avg_inference_time': avg_inference_time
                }
                
        except Exception as e:
            error_msg = f"Failed to get production stats: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return {}
    
    def get_active_downtime_events(self) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM downtime_events 
                    WHERE is_active = 1
                    ORDER BY start_time DESC
                ''')
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            error_msg = f"Failed to get active downtime events: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            return []

if __name__ == '__main__':
    # Test the database manager
    db = DatabaseManager("test_production.db")
    
    # Test shift management
    shift_id = db.start_new_shift(1, operators_count=2, designed_cycle_time=25.0)
    print(f"Started shift ID: {shift_id}")
    
    current_shift = db.get_current_shift()
    print(f"Current shift: {current_shift}")
    
    # Test production cycle logging
    test_cycle_data = {
        'timestamp': time.time(),
        'image_path': 'test_image.jpg',
        'model_type': 'yolo',
        'detection_summary': {
            'total_detections': 2,
            'defect_count': 1,
            'good_count': 1,
            'has_defects': True,
            'quality_status': 'DEFECT DETECTED'
        },
        'timing': {'total_time': 0.5},
        'detections': [
            {'class_name': 'defect', 'confidence': 0.85, 'bbox': [100, 100, 200, 200]},
            {'class_name': 'good', 'confidence': 0.95, 'bbox': [300, 300, 400, 400]}
        ]
    }
    
    cycle_id = db.log_production_cycle(test_cycle_data)
    print(f"Logged cycle ID: {cycle_id}")
    
    # Test OEE calculation
    oee_metrics = db.calculate_oee_metrics()
    print(f"OEE Metrics: {oee_metrics}")
    
    # Test production stats
    stats = db.get_production_stats()
    print(f"Production Stats: {stats}")
    
    # Clean up test database
    if os.path.exists("test_production.db"):
        os.remove("test_production.db")
        print("Test database cleaned up")