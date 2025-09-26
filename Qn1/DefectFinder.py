import cv2
import numpy as np

def classify_defect(image_path, flash_threshold=0.02, cut_threshold=0.017):
    try:

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return "NO OBJECT FOUND"

  
        outer_contour = max(contours, key=cv2.contourArea)
        inner_contour = None
        for i, c in enumerate(contours):
            parent_idx = hierarchy[0][i][3]
            if parent_idx != -1 and np.array_equal(contours[parent_idx], outer_contour):
                if inner_contour is None or cv2.contourArea(c) > cv2.contourArea(inner_contour):
                    inner_contour = c

        if inner_contour is None: return "DEFECT: Single-contour object"

        M = cv2.moments(outer_contour)
        if M['m00'] == 0: cx, cy = image.shape[1] // 2, image.shape[0] // 2
        else: cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

        dists_out = [np.linalg.norm(np.array([cx, cy]) - np.array(p[0])) for p in outer_contour]
        R_out = max(dists_out)
        R_out_min = min(dists_out)
        R_out_avg = np.mean(dists_out)

        dists_in = [np.linalg.norm(np.array([cx, cy]) - np.array(p[0])) for p in inner_contour]
        R_in = min(dists_in)  
        R_in_max = max(dists_in)
        R_in_avg = np.mean(dists_in)
        
  
        FLASH_TOLERANCE = R_out_avg * flash_threshold
        CUT_TOLERANCE = R_out_avg * cut_threshold

  
        flash_out_dev = R_out - R_out_avg
 
        flash_in_dev = R_in_avg - R_in 

        if flash_out_dev > FLASH_TOLERANCE or \
           flash_in_dev > FLASH_TOLERANCE:
            return "FLASH (Material Protrusion)"

        cut_out_dev = R_out_avg - R_out_min

        cut_in_dev = R_in_max - R_in_avg 

        if cut_out_dev > CUT_TOLERANCE or \
           cut_in_dev > CUT_TOLERANCE:
            return "CUT (Material Recession)"

        return "OK"

    except Exception as e:
        return f"ERROR: An error occurred: {e}"


print(f"Defect 1: {classify_defect('./Images/DefectiveImage/defect1.png')}") #is a flash
print(f"Defect 4: {classify_defect('./Images/DefectiveImage/defect4.png')}") #is a flash
print(f"Defect 2: {classify_defect('./Images/DefectiveImage/defect2.png')}") #is a cut
print(f"Defect 3: {classify_defect('./Images/DefectiveImage/defect3.png')}") #is a cut

print(f"good: {classify_defect('./Images/GoodImage/good.png')}") #is a good
