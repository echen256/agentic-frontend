import cv2
import numpy as np
import pytesseract
import json
import argparse
from PIL import Image, ImageFont, ImageDraw
import easyocr
from pathlib import Path
import re

class TextAnalyzer:
    def __init__(self, use_easyocr=True):
        self.use_easyocr = use_easyocr
        if use_easyocr:
            self.reader = easyocr.Reader(['en'])
    
    def extract_text_with_boxes(self, image_path):
        """Extract text with bounding boxes and estimated properties"""
        image = cv2.imread(image_path)
        
        if self.use_easyocr:
            return self._extract_with_easyocr(image_path)
        else:
            return self._extract_with_tesseract(image)
    
    def _extract_with_easyocr(self, image_path):
        """Extract text using EasyOCR"""
        results = self.reader.readtext(image_path)
        text_data = []
        
        for i, (bbox, text, confidence) in enumerate(results):
            if confidence > 0.5:  # Filter low confidence detections
                # Convert bbox to standard format
                bbox_points = np.array(bbox, dtype=np.int32)
                x = int(np.min(bbox_points[:, 0]))
                y = int(np.min(bbox_points[:, 1]))
                w = int(np.max(bbox_points[:, 0]) - x)
                h = int(np.max(bbox_points[:, 1]) - y)
                
                # Estimate font properties
                font_size = self._estimate_font_size(h)
                font_weight = self._estimate_font_weight(text, bbox_points)
                
                text_data.append({
                    'id': i,
                    'text': text.strip(),
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'center': {'x': x + w//2, 'y': y + h//2},
                    'font_size': font_size,
                    'font_weight': font_weight,
                    'confidence': confidence
                })
        
        return text_data
    
    def _extract_with_tesseract(self, image):
        """Extract text using Tesseract OCR"""
        # Convert to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(image_pil, output_type=pytesseract.Output.DICT)
        
        text_data = []
        text_id = 0
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if text and conf > 30:  # Filter low confidence and empty text
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                # Estimate font properties
                font_size = self._estimate_font_size(h)
                font_weight = self._estimate_font_weight(text, None)
                
                text_data.append({
                    'id': text_id,
                    'text': text,
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'center': {'x': x + w//2, 'y': y + h//2},
                    'font_size': font_size,
                    'font_weight': font_weight,
                    'confidence': conf / 100.0
                })
                text_id += 1
        
        return text_data
    
    def _estimate_font_size(self, height):
        """Estimate font size based on text height"""
        # Rough approximation: font size ≈ height * 0.75
        return max(8, int(height * 0.75))
    
    def _estimate_font_weight(self, text, bbox_points):
        """Estimate font weight based on text characteristics"""
        # Simple heuristic: uppercase text and certain keywords suggest bold
        if text.isupper() or any(keyword in text.lower() for keyword in ['button', 'title', 'heading']):
            return 'bold'
        elif any(char in text for char in ['!', '?']) or text.endswith('.'):
            return 'normal'
        else:
            return 'normal'
    
    def match_text_elements(self, target_texts, current_texts):
        """Match text elements between target and current images"""
        matches = []
        unmatched_target = []
        unmatched_current = list(range(len(current_texts)))
        
        for i, target_text in enumerate(target_texts):
            best_match = None
            best_score = 0
            best_index = -1
            
            for j, current_text in enumerate(current_texts):
                if j not in unmatched_current:
                    continue
                    
                # Calculate similarity score
                score = self._calculate_similarity(target_text, current_text)
                
                if score > best_score and score > 0.5:  # Minimum threshold
                    best_score = score
                    best_match = current_text
                    best_index = j
            
            if best_match:
                matches.append({
                    'target_id': i,
                    'current_id': best_index,
                    'similarity': best_score,
                    'target': target_text,
                    'current': best_match
                })
                unmatched_current.remove(best_index)
            else:
                unmatched_target.append(i)
        
        return matches, unmatched_target, unmatched_current
    
    def _calculate_similarity(self, target_text, current_text):
        """Calculate similarity between two text elements"""
        # Text similarity
        text_sim = self._text_similarity(target_text['text'], current_text['text'])
        
        # Position similarity (normalized by image size)
        pos_sim = self._position_similarity(
            target_text['center'], 
            current_text['center'],
            max_distance=200  # pixels
        )
        
        # Size similarity
        size_sim = self._size_similarity(
            target_text['bbox'], 
            current_text['bbox']
        )
        
        # Weighted combination
        return (text_sim * 0.6 + pos_sim * 0.3 + size_sim * 0.1)
    
    def _text_similarity(self, text1, text2):
        """Calculate text similarity using Levenshtein distance"""
        if text1.lower() == text2.lower():
            return 1.0
        
        # Simple Levenshtein distance implementation
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if text1[i-1].lower() == text2[j-1].lower() else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,      # deletion
                    matrix[i][j-1] + 1,      # insertion
                    matrix[i-1][j-1] + cost  # substitution
                )
        
        distance = matrix[len1][len2]
        similarity = 1 - (distance / max(len1, len2))
        return similarity
    
    def _position_similarity(self, pos1, pos2, max_distance):
        """Calculate position similarity"""
        distance = np.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)
        return max(0, 1 - (distance / max_distance))
    
    def _size_similarity(self, bbox1, bbox2):
        """Calculate size similarity"""
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def calculate_differences(self, matches):
        """Calculate differences between matched text elements"""
        differences = []
        
        for match in matches:
            target = match['target']
            current = match['current']
            
            diff = {
                'target_id': match['target_id'],
                'current_id': match['current_id'],
                'similarity': match['similarity'],
                'text_diff': {
                    'target': target['text'],
                    'current': current['text'],
                    'changed': target['text'] != current['text']
                },
                'position_diff': {
                    'target': target['center'],
                    'current': current['center'],
                    'distance': np.sqrt(
                        (target['center']['x'] - current['center']['x'])**2 + 
                        (target['center']['y'] - current['center']['y'])**2
                    ),
                    'dx': current['center']['x'] - target['center']['x'],
                    'dy': current['center']['y'] - target['center']['y']
                },
                'font_size_diff': {
                    'target': target['font_size'],
                    'current': current['font_size'],
                    'difference': current['font_size'] - target['font_size']
                },
                'font_weight_diff': {
                    'target': target['font_weight'],
                    'current': current['font_weight'],
                    'changed': target['font_weight'] != current['font_weight']
                },
                'bbox_diff': {
                    'target': target['bbox'],
                    'current': current['bbox'],
                    'width_diff': current['bbox']['width'] - target['bbox']['width'],
                    'height_diff': current['bbox']['height'] - target['bbox']['height']
                }
            }
            
            differences.append(diff)
        
        return differences
    
    def visualize_text_boxes(self, image_path, text_data, output_path):
        """Create visualization with bounding boxes around detected text"""
        image = cv2.imread(image_path)
        
        for text_info in text_data:
            bbox = text_info['bbox']
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add text label
            label = f"{text_info['text'][:20]}..."
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Text Analysis and Comparison')
    parser.add_argument('--target', type=str, required=True, help='Path to target image')
    parser.add_argument('--current', type=str, required=True, help='Path to current image')
    parser.add_argument('--output', type=str, default='text_analysis.json', help='Output JSON file')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--use-tesseract', action='store_true', help='Use Tesseract instead of EasyOCR')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TextAnalyzer(use_easyocr=not args.use_tesseract)
    
    print("Extracting text from target image...")
    target_texts = analyzer.extract_text_with_boxes(args.target)
    
    print("Extracting text from current image...")
    current_texts = analyzer.extract_text_with_boxes(args.current)
    
    print(f"Found {len(target_texts)} text elements in target image")
    print(f"Found {len(current_texts)} text elements in current image")
    
    # Match text elements
    print("Matching text elements...")
    matches, unmatched_target, unmatched_current = analyzer.match_text_elements(target_texts, current_texts)
    
    print(f"Matched {len(matches)} text elements")
    print(f"Unmatched target elements: {len(unmatched_target)}")
    print(f"Unmatched current elements: {len(unmatched_current)}")
    
    # Calculate differences
    differences = analyzer.calculate_differences(matches)
    
    # Prepare output data
    output_data = {
        'summary': {
            'target_texts_count': len(target_texts),
            'current_texts_count': len(current_texts),
            'matched_count': len(matches),
            'unmatched_target_count': len(unmatched_target),
            'unmatched_current_count': len(unmatched_current)
        },
        'target_texts': target_texts,
        'current_texts': current_texts,
        'matches': matches,
        'differences': differences,
        'unmatched_target_ids': unmatched_target,
        'unmatched_current_ids': unmatched_current
    }
    
    # Save to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis saved to: {args.output}")
    
    # Create visualizations if requested
    if args.visualize:
        analyzer.visualize_text_boxes(args.target, target_texts, 'target_text_boxes.png')
        analyzer.visualize_text_boxes(args.current, current_texts, 'current_text_boxes.png')

if __name__ == "__main__":
    main() 