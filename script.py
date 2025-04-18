import cv2
import numpy as np

def points_to_mask(shapes, image_shape, class_colors=None):
    """
    Convert point annotations to a segmentation mask.
    
    Args:
        shapes (list): List of shape dictionaries, each containing:
            - label (str): The class label
            - points (list): List of [x, y] coordinates
        image_shape (tuple): (height, width) of the target mask
        class_colors (dict, optional): Dictionary mapping class labels to BGR colors
            
    Returns:
        numpy.ndarray: The segmentation mask as a BGR image
    """
    if not shapes:
        return None
    
    if class_colors is None:
        class_colors = {
            "rectangle": (255, 0, 0),      # Red in BGR
            "triangle": (0, 255, 0),       # Green in BGR
            "circle": (0, 0, 255),         # Blue in BGR
            "polygon": (255, 255, 0),      # Yellow in BGR
            "ellipse": (255, 0, 255),      # Magenta in BGR
            "line": (0, 255, 255),         # Cyan in BGR
            "default": (128, 128, 128)     # Gray in BGR
        }
    
    h, w = image_shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for shape in shapes:
        label = shape.get("label", "default")
        points = shape.get("points", [])
        color = class_colors.get(label, class_colors["default"])
        
        if not points:
            continue
        
        points = np.array(points, dtype=np.int32)
        points = points.reshape((-1, 1, 2))
        
        # Fill the shape on the mask
        cv2.fillPoly(mask, [points], color)
    
    return mask

def mask_to_points(mask, point_density=0.01, min_points=10, max_points=100, class_colors=None):
    """
    Convert a segmentation mask back to point annotations.
    
    Args:
        mask (numpy.ndarray): BGR segmentation mask
        point_density (float): Density factor for point generation
        min_points (int): Minimum number of points per shape
        max_points (int): Maximum number of points per shape
        class_colors (dict, optional): Dictionary mapping class labels to BGR colors
            
    Returns:
        list: List of shape dictionaries, each containing:
            - label (str): The class label
            - group_id (int): Shape identifier
            - points (list): List of [x, y] coordinates
    """
    if mask is None:
        return []
    
    # Default class colors if not provided
    if class_colors is None:
        class_colors = {
            "rectangle": (255, 0, 0),      # Red in BGR
            "triangle": (0, 255, 0),       # Green in BGR
            "circle": (0, 0, 255),         # Blue in BGR
            "polygon": (255, 255, 0),      # Yellow in BGR
            "ellipse": (255, 0, 255),      # Magenta in BGR
            "line": (0, 255, 255),         # Cyan in BGR
            "default": (128, 128, 128)     # Gray in BGR
        }
    
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    all_shapes = []
    for i, contour in enumerate(contours):
        # Skip contours that are too small
        if len(contour) < 3:
            continue
            
        sample_point = tuple(contour[0][0])
        if 0 <= sample_point[1] < mask.shape[0] and 0 <= sample_point[0] < mask.shape[1]:
            color = tuple(mask[sample_point[1], sample_point[0]])
        else:
            color = class_colors["default"]
        
        label = "default"
        for class_name, class_color in class_colors.items():
            if np.array_equal(color, class_color):
                label = class_name
                break
        
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        num_points = int(np.sqrt(area) * point_density)
        num_points = max(min(num_points, max_points), min_points)
        
        epsilon = 0.01 * perimeter
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        while len(approx_contour) > num_points and epsilon < 1.0:
            epsilon *= 1.2
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx_contour) < num_points and len(contour) > num_points:
            step = len(contour) // num_points
            indices = [i * step for i in range(num_points)]
            approx_contour = contour[indices]
        
        points = [list(map(int, point[0])) for point in approx_contour]
        
        all_shapes.append({
            "label": label,
            "group_id": i + 1,
            "points": points
        })
    
    return all_shapes

# Example usage
if __name__ == "__main__":
    # Example image dimensions
    img_height, img_width = 480, 640
    
    # Example shapes (polygon points)
    example_shapes = [
        {
            "label": "rectangle",
            "group_id": 1,
            "points": [[100, 100], [300, 100], [300, 200], [100, 200]]
        },
        {
            "label": "triangle",
            "group_id": 2,
            "points": [[400, 300], [500, 200], [600, 300]]
        }
    ]
    
    # Convert points to mask
    mask = points_to_mask(example_shapes, (img_height, img_width))
    
    # Convert mask back to points
    shapes = mask_to_points(mask)
    
    print(f"Original shapes: {len(example_shapes)}")
    print(f"Recovered shapes: {len(shapes)}")
    
    # Visualize (optional)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()