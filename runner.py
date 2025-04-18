from script import points_to_mask, mask_to_points


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
    
    print(mask)
    print(shapes)
    print(f"Original shapes: {len(example_shapes)}")
    print(f"Recovered shapes: {len(shapes)}")
    
    # Visualize (optional)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()