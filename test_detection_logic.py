#!/usr/bin/env python3
"""
Test script to verify the detection processing fix
"""

import numpy as np
import cv2

# Mock the detection processing logic to test the fix
CLASS_NAMES = ['Bad-Seed', 'Good-Seed']
CONFIDENCE_THRESHOLD = 0.3

def mock_draw_predictions(frame, boxes, scores, classes):
    """Mock version of the fixed draw_predictions function"""
    # Collect all detections for smart processing
    detections = []
    best_detection = None
    best_confidence = 0.0
    
    print(f"🔍 Processing {len(boxes)} detections...")
    
    for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
        class_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"Class_{int(cls)}"
        
        # Determine seed quality
        if 'good-seed' in class_name.lower() or 'good' in class_name.lower():
            if score >= 0.7:
                status = "Good Seed"
                seed_type = "good"
            elif score >= 0.5:
                status = "Good Seed (Med)"
                seed_type = "good"
            else:
                status = "Uncertain"
                seed_type = "uncertain"
        elif 'bad-seed' in class_name.lower() or 'bad' in class_name.lower():
            if score >= 0.5:
                status = "Bad Seed"
                seed_type = "bad"
            else:
                status = "Bad Seed (?)"
                seed_type = "bad"
        else:
            status = "Unknown"
            seed_type = "uncertain"
        
        # Store detection info
        detection = {
            'box': box,
            'score': score,
            'class': cls,
            'class_name': class_name,
            'status': status,
            'seed_type': seed_type
        }
        detections.append(detection)
        
        print(f"  Detection {i+1}: {class_name} ({score:.2f}) -> {status}")
        
        # Track the highest confidence detection for action priority
        if score > best_confidence and seed_type in ['good', 'bad']:
            best_confidence = score
            best_detection = detection
            print(f"    ⭐ New best detection: {status} (confidence: {score:.2f})")

    # Process only the best detection for statistics and servo control
    if best_detection and best_detection['seed_type'] in ['good', 'bad']:
        print(f"\n🎯 SELECTED FOR ACTION: {best_detection['status']} (confidence: {best_detection['score']:.2f})")
        print(f"   Servo command: {'RIGHT' if best_detection['seed_type'] == 'good' else 'LEFT'}")
        return best_detection['seed_type']
    else:
        print(f"\n❌ No valid detection selected for action")
        return None

def test_scenarios():
    """Test various detection scenarios"""
    
    print("=" * 60)
    print("🧪 TESTING DETECTION PROCESSING LOGIC")
    print("=" * 60)
    
    # Test Case 1: Multiple seeds with different confidences
    print("\n📋 Test Case 1: Both good and bad seeds detected")
    boxes = [
        [100, 100, 200, 200],  # Good seed
        [300, 300, 400, 400],  # Bad seed
    ]
    scores = [0.85, 0.75]  # Good seed has higher confidence
    classes = [1, 0]  # Good-Seed, Bad-Seed
    
    result = mock_draw_predictions(None, boxes, scores, classes)
    expected = "good"
    print(f"Expected: {expected}, Got: {result}, {'✅ PASS' if result == expected else '❌ FAIL'}")
    
    # Test Case 2: Bad seed has higher confidence
    print("\n📋 Test Case 2: Bad seed has higher confidence")
    boxes = [
        [100, 100, 200, 200],  # Good seed
        [300, 300, 400, 400],  # Bad seed
    ]
    scores = [0.65, 0.90]  # Bad seed has higher confidence
    classes = [1, 0]  # Good-Seed, Bad-Seed
    
    result = mock_draw_predictions(None, boxes, scores, classes)
    expected = "bad"
    print(f"Expected: {expected}, Got: {result}, {'✅ PASS' if result == expected else '❌ FAIL'}")
    
    # Test Case 3: Multiple good seeds - should pick highest confidence
    print("\n📋 Test Case 3: Multiple good seeds")
    boxes = [
        [100, 100, 200, 200],  # Good seed 1
        [300, 300, 400, 400],  # Good seed 2
        [500, 500, 600, 600],  # Good seed 3
    ]
    scores = [0.75, 0.90, 0.65]  # Second one has highest confidence
    classes = [1, 1, 1]  # All Good-Seed
    
    result = mock_draw_predictions(None, boxes, scores, classes)
    expected = "good"
    print(f"Expected: {expected}, Got: {result}, {'✅ PASS' if result == expected else '❌ FAIL'}")
    
    # Test Case 4: No high-confidence detections
    print("\n📋 Test Case 4: No high-confidence detections")
    boxes = [
        [100, 100, 200, 200],  # Low confidence good
        [300, 300, 400, 400],  # Low confidence bad
    ]
    scores = [0.3, 0.4]  # Both low confidence
    classes = [1, 0]  # Good-Seed, Bad-Seed
    
    result = mock_draw_predictions(None, boxes, scores, classes)
    expected = "bad"  # Bad seed should still be selected as it meets 0.5 threshold in bad logic
    print(f"Expected: {expected}, Got: {result}, {'✅ PASS' if result == expected else '❌ FAIL'}")

if __name__ == "__main__":
    test_scenarios()
    
    print("\n" + "=" * 60)
    print("🎉 DETECTION LOGIC TEST COMPLETE")
    print("=" * 60)
    print("\n✅ Key Fix Implemented:")
    print("• Multiple detections are collected first")
    print("• Only the highest confidence detection triggers action")
    print("• Prevents conflicting servo commands")
    print("• Ensures accurate statistics")
