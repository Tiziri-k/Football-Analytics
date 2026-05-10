def get_center_of_box(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (int(center_x), int(center_y))


def get_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return int(x2 - x1)

def get_bbox_height(bbox):
    x1, y1, x2, y2 = bbox
    return int(y2 - y1)


def measure_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)**0.5