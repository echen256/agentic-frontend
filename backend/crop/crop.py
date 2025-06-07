import cv2
import numpy as np


def detect_colored_features_bbox(image, saturation_threshold=30, min_area=100):
    """
    Detect colored regions and return their bounding box
    """
    # Convert to HSV to better detect colored regions
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for colored regions (high saturation)
    mask = cv2.inRange(hsv[:,:,1], saturation_threshold, 255)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None, mask
    
    # Filter contours by area
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if len(large_contours) == 0:
        return None, mask
    
    # Get bounding box that encompasses all large colored regions
    all_points = np.vstack(large_contours)
    x, y, w, h = cv2.boundingRect(all_points)
    
    return (x, y, w, h), mask

def preprocess_images(img1, img2):
    """
    Apply preprocessing including grayscale conversion
    """
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Optional: Apply histogram equalization for better feature detection
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)
    
    return gray1, gray2

# Load images
button_template = cv2.imread('target.png')
image2 = cv2.imread('current.png')

print("Original image shapes:")
print(f"Template: {button_template.shape}")
print(f"Current: {image2.shape}")

# Detect colored features first
print("\nDetecting colored features...")
template_bbox, template_color_mask = detect_colored_features_bbox(button_template)
current_bbox, current_color_mask = detect_colored_features_bbox(image2)

if template_bbox:
    x, y, w, h = template_bbox
    print(f"Template colored region bbox: x={x}, y={y}, w={w}, h={h}")
    
    # Draw bounding box on template
    template_with_bbox = button_template.copy()
    cv2.rectangle(template_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('template_colored_bbox.png', template_with_bbox)
else:
    print("No significant colored regions found in template")

if current_bbox:
    x, y, w, h = current_bbox
    print(f"Current colored region bbox: x={x}, y={y}, w={w}, h={h}")
    
    # Draw bounding box on current image
    current_with_bbox = image2.copy()
    cv2.rectangle(current_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('current_colored_bbox.png', current_with_bbox)
else:
    print("No significant colored regions found in current image")

# Save color masks
cv2.imwrite('template_color_mask.png', template_color_mask)
cv2.imwrite('current_color_mask.png', current_color_mask)

# Apply grayscale preprocessing
print("\nApplying grayscale preprocessing...")
gray_template, gray_current = preprocess_images(button_template, image2)

# Save preprocessed images
cv2.imwrite('template_gray_processed.png', gray_template)
cv2.imwrite('current_gray_processed.png', gray_current)

# Feature detection on grayscale images
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray_template, None)
kp2, des2 = orb.detectAndCompute(gray_current, None)

print(f"\nFeature detection results:")
print(f"Template keypoints: {len(kp1) if kp1 else 0}")
print(f"Current keypoints: {len(kp2) if kp2 else 0}")

# Match descriptors
if des1 is not None and des2 is not None:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    print(f"Found {len(matches)} matches")
    
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Method 1: Generate mask using homography (for finding the object region)
    def generate_homography_mask(matches, kp1, kp2, template_shape, target_shape):
        if len(matches) < 4:
            print("Not enough matches for homography")
            return None
        
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask_homo = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print("Could not find homography")
            return None
        
        # Get template corners
        h, w = template_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to target image
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        # Create mask from transformed corners
        mask = np.zeros(target_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(transformed_corners)], 255)
        
        return mask, transformed_corners, mask_homo
    
    # Method 2: Generate mask based on keypoint locations (for highlighting match points)
    def generate_keypoint_mask(matches, kp2, target_shape, radius=20):
        mask = np.zeros(target_shape[:2], dtype=np.uint8)
        
        for match in matches:
            # Get keypoint location in target image
            pt = kp2[match.trainIdx].pt
            center = (int(pt[0]), int(pt[1]))
            
            # Draw circle at keypoint location
            cv2.circle(mask, center, radius, 255, -1)
        
        return mask
    
    # Method 3: Generate confidence-based mask (weighted by match quality)
    def generate_confidence_mask(matches, kp2, target_shape, max_distance=50):
        mask = np.zeros(target_shape[:2], dtype=np.float32)
        
        # Find max and min distances for normalization
        distances = [m.distance for m in matches]
        min_dist = min(distances)
        max_dist = max(distances)
        
        for match in matches:
            pt = kp2[match.trainIdx].pt
            center = (int(pt[0]), int(pt[1]))
            
            # Calculate confidence (inverse of normalized distance)
            if max_dist > min_dist:
                confidence = 1.0 - (match.distance - min_dist) / (max_dist - min_dist)
            else:
                confidence = 1.0
            
            # Add gaussian blob weighted by confidence
            y, x = np.ogrid[:target_shape[0], :target_shape[1]]
            center_dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            gaussian = np.exp(-(center_dist**2) / (2 * (max_distance/3)**2))
            mask += gaussian * confidence
        
        # Normalize to 0-255
        if mask.max() > 0:
            mask = (mask / mask.max() * 255).astype(np.uint8)
        
        return mask
    
    # Generate different types of masks
    if len(matches) > 0:
        print(f"\nProcessing {len(matches)} matches...")
        
        # Method 1: Homography-based mask (finds the object region)
        homo_result = generate_homography_mask(
            matches, kp1, kp2, gray_template.shape, gray_current.shape
        )
        
        if homo_result is not None:
            homo_mask, corners, homo_inliers = homo_result
            cv2.imwrite('homography_mask.png', homo_mask)
            print("Saved homography mask")
            
            # Draw the found region on the original color image
            result_img = image2.copy()
            cv2.polylines(result_img, [np.int32(corners)], True, (0, 255, 0), 3)
            cv2.imwrite('detected_region.png', result_img)
        
        # Method 2: Keypoint-based mask
        keypoint_mask = generate_keypoint_mask(matches[:20], kp2, gray_current.shape)  # Use top 20 matches
        cv2.imwrite('keypoint_mask.png', keypoint_mask)
        print("Saved keypoint mask")
        
        # Method 3: Confidence-based mask
        confidence_mask = generate_confidence_mask(matches, kp2, gray_current.shape)
        cv2.imwrite('confidence_mask.png', confidence_mask)
        print("Saved confidence mask")
        
        # Optional: Create combined visualization using grayscale images
        visualization = cv2.drawMatches(
            gray_template, kp1, gray_current, kp2, matches[:20], None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite('matches_visualization.png', visualization)
        print("Saved matches visualization")
    
    else:
        print("No matches found!")

else:
    print("Could not detect features in one or both images!")
