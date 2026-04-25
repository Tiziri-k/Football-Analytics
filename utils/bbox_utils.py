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