import cv2
import numpy as np

class LineDrawer:
    def __init__(self):
        self.line_pts = []
        self.drawing = False

    def draw_line(self, frame):
        clone = frame.copy()
        cv2.namedWindow('Draw Line')
        cv2.setMouseCallback('Draw Line', self.mouse_callback, param=clone)
        while True:
            temp = clone.copy()
            if len(self.line_pts) == 2:
                cv2.line(temp, self.line_pts[0], self.line_pts[1], (0,255,0), 2)
            cv2.imshow('Draw Line', temp)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == 27:  # Enter or Esc
                break
        cv2.destroyWindow('Draw Line')
        return self.line_pts

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.line_pts) < 2:
            self.line_pts.append((x, y))

    def point_side(self, pt):
        # Returns >0 if on one side, <0 on the other, 0 on the line
        if len(self.line_pts) < 2:
            return 0
        (x1, y1), (x2, y2) = self.line_pts
        return (x2 - x1)*(pt[1] - y1) - (y2 - y1)*(pt[0] - x1) 