# to get the two coordinates in the frame....

import cv2

def get_line_from_user(frame):
    """
    Shows the frame scaled to fit typical screens, lets the user draw a line,
    and returns the two points in ORIGINAL frame coordinates as ((x1, y1), (x2, y2)).
    """
    h, w = frame.shape[:2]
    # Target max display size
    max_w, max_h = 1280, 720
    scale = min(max_w / w, max_h / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)

    # Prepare display frame
    disp_frame = cv2.resize(frame, (disp_w, disp_h)) if scale < 1.0 else frame.copy()
    temp_frame = disp_frame.copy()
    points_disp = []  # Points in display coords

    def to_orig(pt):
        # Map display coords back to original image coords
        return (int(round(pt[0] / scale)), int(round(pt[1] / scale)))

    def on_mouse(event, x, y, flags, param):
        nonlocal temp_frame
        if event == cv2.EVENT_LBUTTONDOWN and len(points_disp) < 2:
            points_disp.append((x, y))
        # Redraw overlay
        temp_frame = disp_frame.copy()
        if len(points_disp) >= 1:
            cv2.circle(temp_frame, points_disp[0], 5, (0, 255, 0), -1)
        if len(points_disp) == 2:
            cv2.line(temp_frame, points_disp[0], points_disp[1], (0, 255, 0), 2)

    win = "Draw ROI Line"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if len(points_disp) == 2:
            break

    cv2.destroyAllWindows()
    if len(points_disp) == 2:
        p1, p2 = to_orig(points_disp[0]), to_orig(points_disp[1])
        print(f"Line coordinates: {p1} to {p2}")
        return (p1, p2)
    return None


