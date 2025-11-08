"""
Report generation module for documenting restoration process and results.
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate detailed reports of restoration process."""
    
    def __init__(self):
        self.report_data = {}
        
    def create_report(self, 
                     restoration_results: Dict,
                     face_results: Optional[Dict] = None,
                     text_results: Optional[Dict] = None,
                     multi_frame_results: Optional[Dict] = None) -> Dict:
        """
        Create a comprehensive restoration report.
        
        Args:
            restoration_results: Results from restoration engine
            face_results: Results from face preservation
            text_results: Results from text reconstruction
            multi_frame_results: Results from multi-frame comparison
            
        Returns:
            Complete restoration report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'restoration_type': 'lost_detail_restoration',
            'summary': {
                'operations_performed': [],
                'improvements_made': {},
                'details_restored': []
            },
            'detailed_results': {}
        }
        
        # Add restoration engine results
        if restoration_results:
            report['detailed_results']['image_enhancement'] = restoration_results
            report['summary']['operations_performed'].extend(
                restoration_results.get('operations', [])
            )
            report['summary']['improvements_made'].update(
                restoration_results.get('improvements', {})
            )
            
            # Add metrics
            if 'metrics' in restoration_results:
                report['summary']['improvements_made']['quantitative_metrics'] = \
                    restoration_results['metrics']
        
        # Add face preservation results
        if face_results:
            report['detailed_results']['face_preservation'] = face_results
            if face_results.get('faces_detected', 0) > 0:
                report['summary']['details_restored'].append(
                    f"Preserved identity of {face_results['faces_detected']} face(s)"
                )
                report['summary']['operations_performed'].extend(
                    face_results.get('operations', [])
                )
        
        # Add text reconstruction results
        if text_results:
            report['detailed_results']['text_reconstruction'] = text_results
            if text_results.get('text_regions_found', 0) > 0:
                detected_texts = [t['text'] for t in text_results.get('text_detected', [])]
                report['summary']['details_restored'].append(
                    f"Reconstructed {len(detected_texts)} text region(s)"
                )
                report['summary']['details_restored'].append(
                    f"Detected text: {', '.join(detected_texts[:5])}"  # First 5 texts
                )
                report['summary']['operations_performed'].extend(
                    text_results.get('operations', [])
                )
        
        # Add multi-frame comparison results
        if multi_frame_results:
            report['detailed_results']['multi_frame_analysis'] = multi_frame_results
            if multi_frame_results.get('num_frames', 0) > 1:
                report['summary']['details_restored'].append(
                    f"Combined {multi_frame_results['num_frames']} frames for detail recovery"
                )
                report['summary']['operations_performed'].extend(
                    multi_frame_results.get('operations', [])
                )
                report['summary']['improvements_made'].update(
                    multi_frame_results.get('improvements', {})
                )
        
        # Generate human-readable summary
        report['human_readable_summary'] = self._generate_summary_text(report)
        
        return report
    
    def _generate_summary_text(self, report: Dict) -> str:
        """Generate human-readable summary text."""
        summary_lines = [
            "=== AI Lost Detail Restorer - Restoration Report ===\n",
            f"Timestamp: {report['timestamp']}\n",
            "\n--- Summary ---\n"
        ]
        
        # Operations
        if report['summary']['operations_performed']:
            summary_lines.append("Operations Performed:")
            for op in report['summary']['operations_performed']:
                summary_lines.append(f"  • {op.replace('_', ' ').title()}")
            summary_lines.append("")
        
        # Improvements
        if report['summary']['improvements_made']:
            summary_lines.append("Improvements Made:")
            for key, value in report['summary']['improvements_made'].items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        summary_lines.append(f"  • {sub_key.replace('_', ' ').title()}: {sub_value}")
                else:
                    summary_lines.append(f"  • {key.replace('_', ' ').title()}: {value}")
            summary_lines.append("")
        
        # Details restored
        if report['summary']['details_restored']:
            summary_lines.append("Details Restored:")
            for detail in report['summary']['details_restored']:
                summary_lines.append(f"  • {detail}")
            summary_lines.append("")
        
        # Detailed results
        summary_lines.append("\n--- Detailed Results ---\n")
        for section, data in report['detailed_results'].items():
            summary_lines.append(f"{section.replace('_', ' ').title()}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if key not in ['operations', 'improvements', 'metrics']:
                        if isinstance(value, (list, dict)):
                            summary_lines.append(f"  {key}: {len(value) if isinstance(value, list) else 'Available'}")
                        else:
                            summary_lines.append(f"  {key}: {value}")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def save_report(self, report: Dict, filepath: str, format: str = 'json'):
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            filepath: Output file path
            format: Output format ('json' or 'txt')
        """
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        elif format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report.get('human_readable_summary', str(report)))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Report saved to {filepath}")


