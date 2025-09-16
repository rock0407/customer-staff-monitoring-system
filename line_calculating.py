# to get the two coordinates in the frame....

import cv2

def get_line_from_user(frame):
    """
    Shows the frame, lets the user draw a line, and returns the two points as ((x1, y1), (x2, y2)).
    """
    points = []
    frame_copy = frame.copy()
    drawing = [False]

    def draw_line(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not drawing[0] and len(points) < 1:
                points.append((x, y))
                drawing[0] = True
            elif drawing[0] and len(points) < 2:
                points.append((x, y))
                drawing[0] = False
                cv2.line(frame_copy, points[0], points[1], (0, 255, 0), 2)

    cv2.namedWindow("Draw ROI Line")
    cv2.setMouseCallback("Draw ROI Line", draw_line)

    while True:
        temp = frame_copy.copy()
        if len(points) == 1:
            cv2.circle(temp, points[0], 4, (0, 255, 0), -1)
        if len(points) == 2:
            cv2.line(temp, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Draw ROI Line", temp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(points) == 2:  # ESC or line drawn
            break

    cv2.destroyAllWindows()
    if len(points) == 2:
        print(f"Line coordinates: {points[0]} to {points[1]}")
        return (points[0], points[1])
    else:
        return None


