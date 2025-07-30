#!/usr/bin/env python3
"""
Export Manager - Production Data Export for Dana Heat Shield GUI

This module handles exporting production data and statistics in multiple formats
for reporting, analysis, and integration with other systems. It provides
comprehensive data export capabilities with professional formatting.

Supported Export Formats:
1. CSV: Multiple files for different data types (cycles, detections, downtime, summary)
2. JSON: Complete structured data export for system integration
3. HTML: Professional reports with charts and formatted tables

Export Data Includes:
- Production cycles with timing and quality results
- Individual detection details with bounding box coordinates
- Downtime events with reasons and durations
- OEE metrics and performance statistics
- Shift information and operator data
- Summary statistics and defect ratios

Use Cases:
- Shift reports for management review
- Quality analysis and trend identification
- Integration with ERP/MES systems
- Compliance documentation
- Performance optimization analysis

The export system maintains data integrity and provides flexible filtering
options for date ranges and specific shifts.

Date: 2025-07-29
Version: 1.0
"""

import sqlite3
import csv
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal

class ExportManager(QObject):
    """
    Manages export of production data to various formats for reporting and analysis.
    
    This class provides comprehensive data export capabilities, transforming
    raw production data into professional reports suitable for management
    review, quality analysis, and system integration.
    
    Signals:
        export_completed(str): Emitted when export finishes successfully (file_path)
        export_error(str): Emitted when export fails (error_message)
    
    The export process:
    1. Extracts data from database with optional filtering
    2. Calculates summary statistics and OEE metrics
    3. Formats data according to target format requirements
    4. Generates professional output files
    5. Signals completion or error status
    
    Export formats are designed for different audiences:
    - CSV: For spreadsheet analysis and data processing
    - JSON: For system integration and API consumption
    - HTML: For executive reports and presentations
    """
    export_completed = pyqtSignal(str)  # file_path when export succeeds
    export_error = pyqtSignal(str)      # error_message when export fails
    
    def __init__(self, database_manager):
        super().__init__()
        self.database = database_manager  # Reference to database for data access
    
    def export_production_report(self, output_path: str, format_type: str = 'csv', 
                                shift_id: Optional[int] = None, 
                                date_range: Optional[tuple] = None) -> bool:
        """Export production report in specified format"""
        try:
            # Get production data
            data = self._get_production_data(shift_id, date_range)
            
            if format_type.lower() == 'csv':
                success = self._export_csv(data, output_path)
            elif format_type.lower() == 'json':
                success = self._export_json(data, output_path)
            elif format_type.lower() == 'html':
                success = self._export_html(data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            if success:
                self.export_completed.emit(output_path)
                return True
            else:
                return False
                
        except Exception as e:
            error_msg = f"Export failed: {str(e)}"
            self.export_error.emit(error_msg)
            return False
    
    def _get_production_data(self, shift_id: Optional[int] = None, 
                           date_range: Optional[tuple] = None) -> Dict[str, Any]:
        """Get production data from database"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                
                # Build WHERE clause
                where_conditions = []
                params = []
                
                if shift_id:
                    where_conditions.append("pc.shift_id = ?")
                    params.append(shift_id)
                
                if date_range:
                    where_conditions.append("pc.timestamp BETWEEN ? AND ?")
                    params.extend(date_range)
                
                where_clause = " AND " + " AND ".join(where_conditions) if where_conditions else ""
                
                # Get production cycles with shift info
                query = f"""
                    SELECT 
                        pc.id, pc.timestamp, pc.cycle_time, pc.image_path, 
                        pc.model_type, pc.total_detections, pc.defect_count, 
                        pc.good_count, pc.has_defects, pc.quality_status, 
                        pc.inference_time, s.shift_number, s.operators_count,
                        s.designed_cycle_time
                    FROM production_cycles pc
                    LEFT JOIN shifts s ON pc.shift_id = s.id
                    WHERE 1=1 {where_clause}
                    ORDER BY pc.timestamp
                """
                
                cursor.execute(query, params)
                cycles = cursor.fetchall()
                
                # Get column names
                cycle_columns = [desc[0] for desc in cursor.description]
                
                # Get detection details
                detection_query = f"""
                    SELECT dd.cycle_id, dd.class_name, dd.confidence, 
                           dd.bbox_x1, dd.bbox_y1, dd.bbox_x2, dd.bbox_y2
                    FROM detection_details dd
                    JOIN production_cycles pc ON dd.cycle_id = pc.id
                    WHERE 1=1 {where_clause}
                    ORDER BY dd.cycle_id, dd.confidence DESC
                """
                
                cursor.execute(detection_query.replace("pc.", "pc."), params)
                detections = cursor.fetchall()
                
                # Get downtime events
                downtime_query = f"""
                    SELECT de.start_time, de.end_time, de.duration, de.reason, 
                           de.description, s.shift_number
                    FROM downtime_events de
                    LEFT JOIN shifts s ON de.shift_id = s.id
                    WHERE de.end_time IS NOT NULL {where_clause.replace("pc.", "de.")}
                    ORDER BY de.start_time
                """
                
                cursor.execute(downtime_query, params)
                downtime_events = cursor.fetchall()
                
                # Get OEE metrics
                oee_metrics = self.database.calculate_oee_metrics(shift_id)
                
                # Organize data
                cycles_data = []
                for cycle in cycles:
                    cycle_dict = dict(zip(cycle_columns, cycle))
                    cycle_dict['timestamp_formatted'] = datetime.fromtimestamp(cycle_dict['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    cycles_data.append(cycle_dict)
                
                detections_data = []
                for detection in detections:
                    detections_data.append({
                        'cycle_id': detection[0],
                        'class_name': detection[1],
                        'confidence': detection[2],
                        'bbox_x1': detection[3],
                        'bbox_y1': detection[4],
                        'bbox_x2': detection[5],
                        'bbox_y2': detection[6]
                    })
                
                downtime_data = []
                for downtime in downtime_events:
                    downtime_data.append({
                        'start_time': datetime.fromtimestamp(downtime[0]).strftime('%Y-%m-%d %H:%M:%S'),
                        'end_time': datetime.fromtimestamp(downtime[1]).strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_minutes': round(downtime[2] / 60, 2),
                        'reason': downtime[3],
                        'description': downtime[4],
                        'shift_number': downtime[5]
                    })
                
                return {
                    'production_cycles': cycles_data,
                    'detections': detections_data,
                    'downtime_events': downtime_data,
                    'oee_metrics': oee_metrics,
                    'summary': self._calculate_summary(cycles_data, downtime_data, oee_metrics),
                    'export_timestamp': datetime.now().isoformat(),
                    'shift_id': shift_id,
                    'date_range': date_range
                }
                
        except Exception as e:
            raise Exception(f"Failed to get production data: {str(e)}")
    
    def _calculate_summary(self, cycles_data: List[Dict], downtime_data: List[Dict], 
                          oee_metrics: Dict) -> Dict[str, Any]:
        """Calculate summary statistics"""
        if not cycles_data:
            return {}
        
        total_cycles = len(cycles_data)
        total_defects = sum(cycle.get('defect_count', 0) for cycle in cycles_data)
        total_good = sum(cycle.get('good_count', 0) for cycle in cycles_data)
        
        cycle_times = [cycle.get('cycle_time', 0) for cycle in cycles_data if cycle.get('cycle_time', 0) > 0]
        avg_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
        
        total_downtime = sum(dt.get('duration_minutes', 0) for dt in downtime_data)
        
        return {
            'total_cycles': total_cycles,
            'total_defects': total_defects,
            'total_good_parts': total_good,
            'defect_rate_percent': (total_defects / total_cycles * 100) if total_cycles > 0 else 0,
            'average_cycle_time': round(avg_cycle_time, 2),
            'total_downtime_minutes': round(total_downtime, 2),
            'availability_percent': round(oee_metrics.get('availability', 0) * 100, 1),
            'performance_percent': round(oee_metrics.get('performance', 0) * 100, 1),
            'quality_percent': round(oee_metrics.get('quality', 0) * 100, 1),
            'oee_percent': round(oee_metrics.get('oee', 0) * 100, 1)
        }
    
    def _export_csv(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data as CSV"""
        try:
            # Create summary CSV
            summary_path = output_path.replace('.csv', '_summary.csv')
            with open(summary_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                
                summary = data['summary']
                for key, value in summary.items():
                    writer.writerow([key.replace('_', ' ').title(), value])
            
            # Create production cycles CSV
            cycles_path = output_path.replace('.csv', '_cycles.csv')
            if data['production_cycles']:
                with open(cycles_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data['production_cycles'][0].keys())
                    writer.writeheader()
                    writer.writerows(data['production_cycles'])
            
            # Create detections CSV
            detections_path = output_path.replace('.csv', '_detections.csv')
            if data['detections']:
                with open(detections_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data['detections'][0].keys())
                    writer.writeheader()
                    writer.writerows(data['detections'])
            
            # Create downtime CSV
            downtime_path = output_path.replace('.csv', '_downtime.csv')
            if data['downtime_events']:
                with open(downtime_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data['downtime_events'][0].keys())
                    writer.writeheader()
                    writer.writerows(data['downtime_events'])
            
            print(f"CSV export completed: {output_path} (multiple files)")
            return True
            
        except Exception as e:
            raise Exception(f"CSV export failed: {str(e)}")
    
    def _export_json(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data as JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"JSON export completed: {output_path}")
            return True
            
        except Exception as e:
            raise Exception(f"JSON export failed: {str(e)}")
    
    def _export_html(self, data: Dict[str, Any], output_path: str) -> bool:
        """Export data as HTML report"""
        try:
            html_content = self._generate_html_report(data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML export completed: {output_path}")
            return True
            
        except Exception as e:
            raise Exception(f"HTML export failed: {str(e)}")
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        summary = data['summary']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Dana Heat Shield - Production Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #2b2b2b; color: white; padding: 20px; text-align: center; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2b2b2b; }}
        .metric-label {{ font-size: 0.9em; color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4a4a4a; color: white; }}
        .good {{ color: green; }}
        .defect {{ color: red; }}
        .timestamp {{ font-size: 0.8em; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Dana Heat Shield - Production Report</h1>
        <p>Generated: {data['export_timestamp']}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{summary.get('total_cycles', 0)}</div>
            <div class="metric-label">Total Parts</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('defect_rate_percent', 0):.1f}%</div>
            <div class="metric-label">Defect Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('average_cycle_time', 0):.1f}s</div>
            <div class="metric-label">Avg Cycle Time</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('oee_percent', 0):.1f}%</div>
            <div class="metric-label">Overall OEE</div>
        </div>
    </div>
    
    <h2>OEE Breakdown</h2>
    <div class="summary">
        <div class="metric">
            <div class="metric-value">{summary.get('availability_percent', 0):.1f}%</div>
            <div class="metric-label">Availability</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('performance_percent', 0):.1f}%</div>
            <div class="metric-label">Performance</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('quality_percent', 0):.1f}%</div>
            <div class="metric-label">Quality</div>
        </div>
    </div>
"""
        
        # Add production cycles table
        if data['production_cycles']:
            html += """
    <h2>Production Cycles</h2>
    <table>
        <tr>
            <th>Timestamp</th>
            <th>Cycle Time</th>
            <th>Quality Status</th>
            <th>Defects</th>
            <th>Good Parts</th>
            <th>Model</th>
        </tr>
"""
            for cycle in data['production_cycles'][-50:]:  # Show last 50 cycles
                status_class = "defect" if cycle.get('has_defects') else "good"
                html += f"""
        <tr>
            <td class="timestamp">{cycle.get('timestamp_formatted', '')}</td>
            <td>{cycle.get('cycle_time', 0):.2f}s</td>
            <td class="{status_class}">{cycle.get('quality_status', '')}</td>
            <td>{cycle.get('defect_count', 0)}</td>
            <td>{cycle.get('good_count', 0)}</td>
            <td>{cycle.get('model_type', '').upper()}</td>
        </tr>
"""
            html += "    </table>"
        
        # Add downtime events table
        if data['downtime_events']:
            html += """
    <h2>Downtime Events</h2>
    <table>
        <tr>
            <th>Start Time</th>
            <th>End Time</th>
            <th>Duration (min)</th>
            <th>Reason</th>
            <th>Description</th>
        </tr>
"""
            for downtime in data['downtime_events']:
                html += f"""
        <tr>
            <td class="timestamp">{downtime.get('start_time', '')}</td>
            <td class="timestamp">{downtime.get('end_time', '')}</td>
            <td>{downtime.get('duration_minutes', 0):.1f}</td>
            <td>{downtime.get('reason', '')}</td>
            <td>{downtime.get('description', '')}</td>
        </tr>
"""
            html += "    </table>"
        
        html += """
    <div class="timestamp" style="text-align: center; margin-top: 40px;">
        Report generated by Dana Heat Shield GUI
    </div>
</body>
</html>
"""
        return html

if __name__ == '__main__':
    # Test the export manager
    import sys
    sys.path.append('.')
    
    from database_manager import DatabaseManager
    
    # Create test database with some data
    db = DatabaseManager("test_export.db")
    shift_id = db.start_new_shift(1, 2, 25.0)
    
    # Add some test cycles
    import time
    for i in range(5):
        test_cycle_data = {
            'timestamp': time.time() + i,
            'image_path': f'test_image_{i}.jpg',
            'model_type': 'yolo',
            'detection_summary': {
                'total_detections': 1 + (i % 2),
                'defect_count': i % 2,
                'good_count': 1,
                'has_defects': bool(i % 2),
                'quality_status': 'DEFECT DETECTED' if i % 2 else 'GOOD'
            },
            'timing': {'total_time': 0.5 + (i * 0.1)},
            'detections': [{'class_name': 'defect' if i % 2 else 'good', 'confidence': 0.85, 'bbox': [100, 100, 200, 200]}]
        }
        db.log_production_cycle(test_cycle_data)
    
    # Test export
    export_manager = ExportManager(db)
    
    print("Testing CSV export...")
    export_manager.export_production_report("test_report.csv", "csv")
    
    print("Testing JSON export...")
    export_manager.export_production_report("test_report.json", "json")
    
    print("Testing HTML export...")
    export_manager.export_production_report("test_report.html", "html")
    
    # Clean up
    import os
    for file in ["test_export.db", "test_report_summary.csv", "test_report_cycles.csv", 
                 "test_report_detections.csv", "test_report_downtime.csv", 
                 "test_report.json", "test_report.html"]:
        if os.path.exists(file):
            os.remove(file)
    
    print("Export tests completed!")