import numpy as np
import cv2
from Line import Line

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def canny(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(image_gray,(5,5),0)
    canny = cv2.Canny(blur,50,250)
    return canny

def canny(img, low_threshold = 50, high_threshold = 250):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
 
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, mask


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, alpha=0.8, beta=1., lamda=0.):
    """
    Returns resulting blend image computed as follows:

    initial_img * alpha + img * beta + lamda
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane


def get_lane_lines(color_image, solid_lines=True):
    """
    This function take as input a color road frame and tries to infer the lane lines in the image.
    :param color_image: input frame
    :param solid_lines: if True, only selected lane lines are returned. If False, all candidate lines are returned.
    :return: list of (candidate) lane lines.
    """
    # resize to 960 x 540
    color_image = np.array(color_image)
    # convert to grayscale
    if (len(color_image.shape) == 3):
        img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = color_image
    # perform gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

    # perform edge detection
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

    # perform hough transform
    detected_lines = hough_lines_detection(img=img_edge,
                                           rho=2,
                                           theta=np.pi / 180,
                                           threshold=1,
                                           min_line_len=15,
                                           max_line_gap=5)

    # convert (x1, y1, x2, y2) tuples into Lines
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

    # if 'solid_lines' infer the two lane lines
    if solid_lines:
        candidate_lines = []
        for line in detected_lines:
                # consider only lines with slope between 30 and 60 degrees
                if 0.5 <= np.abs(line.slope) <= 2:
                    candidate_lines.append(line)
        # interpolate lines candidates to find both lanes
        lane_lines = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    else:
        # if not solid_lines, just return the hough transform output
        lane_lines = detected_lines

    return lane_lines


def smoothen_over_time(lane_lines):
    """
    Smooth the lane line inference over a window of frames and returns the average lines.
    """

    avg_line_lt = np.zeros((len(lane_lines), 4))
    avg_line_rt = np.zeros((len(lane_lines), 4))

    for t in range(0, len(lane_lines)):
        avg_line_lt[t] += lane_lines[t][0].get_coords()
        avg_line_rt[t] += lane_lines[t][1].get_coords()

    return Line(*np.mean(avg_line_lt, axis=0)), Line(*np.mean(avg_line_rt, axis=0))


def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    """
    Entry point for lane_detection pipeline. Takes as input a list of frames (RGB) and returns an image (RGB)
    with overlaid the inferred road lanes. Eventually, len(frames)==1 in the case of a single image.
    """
    
    is_videoclip = len(frames) > 0

    img_h, img_w = frames[0].shape[0], frames[0].shape[1]

    lane_lines = []
    for t in range(0, len(frames)):
        inferred_lanes = get_lane_lines(color_image=frames[t], solid_lines=solid_lines)
        lane_lines.append(inferred_lanes)

    if temporal_smoothing and solid_lines:
        lane_lines = smoothen_over_time(lane_lines)
    else:
        lane_lines = lane_lines[0]

    # prepare empty mask on which lines are drawn
    line_img = np.zeros(shape=(img_h, img_w))

    # draw lanes found
    for lane in lane_lines:
        try:
            lane.draw(line_img)
        except:
            print('Catn see lane_lines')

    # keep only region of interest by masking
    imshape = frames[t].shape
    vertices = np.array([[(int(0*imshape[1]),imshape[0]),(int(0*imshape[1]), 
                    int(0.25*imshape[0])), (int(0.80*imshape[1]), int(0.3*imshape[0])), (
                    imshape[1],imshape[0])]], dtype=np.int32)
    img_masked, _ = region_of_interest(line_img, vertices)
    
    #lane_lines = get_lane_lines(img_masked)
  
    # make blend on color image
    img_color = frames[-1] if is_videoclip else frames[0]
    img_blend = weighted_img(img_masked, img_color, alpha=0.8, beta=1., lamda=0.)
    return img_blend, lane_lines[0], lane_lines[1]

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # Function has been written to work with Challenge video as well
    # b -0, g-1, r-2 
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # At the bottom of the image, imshape[0] and top has been defined as 330
    imshape = img.shape 
    
    slope_left=0
    slope_right=0
    leftx=0
    lefty=0
    rightx=0
    righty=0
    i=0
    j=0
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope >0.1: #Left lane and not a straight line
                # Add all values of slope and average position of a line
                slope_left += slope 
                leftx += (x1+x2)/2
                lefty += (y1+y2)/2
                i+= 1
            elif slope < -0.2: # Right lane and not a straight line
                # Add all values of slope and average position of a line
                slope_right += slope
                rightx += (x1+x2)/2
                righty += (y1+y2)/2
                j+= 1
    # Left lane - Average across all slope and intercepts
    if i>0: # If left lane is detected
        avg_slope_left = slope_left/i
        avg_leftx = leftx/i
        avg_lefty = lefty/i
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_l = int(((int(0.97*imshape[0])-avg_lefty)/avg_slope_left) + avg_leftx)
        xt_l = int(((int(0.61*imshape[0])-avg_lefty)/avg_slope_left)+ avg_leftx)

    else: # If Left lane is not detected - best guess positions of bottom x and top x
        xb_l = int(0.21*imshape[1])
        xt_l = int(0.43*imshape[1])
    
    # Draw a line
    cv2.line(img, (xt_l, int(0.61*imshape[0])), (xb_l, int(0.97*imshape[0])), color, thickness)
    
    #Right lane - Average across all slope and intercepts
    if j>0: # If right lane is detected
        avg_slope_right = slope_right/j
        avg_rightx = rightx/j
        avg_righty = righty/j
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_r = int(((int(0.97*imshape[0])-avg_righty)/avg_slope_right) + avg_rightx)
        xt_r = int(((int(0.61*imshape[0])-avg_righty)/avg_slope_right)+ avg_rightx)
    
    else: # If right lane is not detected - best guess positions of bottom x and top x
        xb_r = int(0.89*imshape[1])
        xt_r = int(0.53*imshape[1])
    
    # Draw a line    
    cv2.line(img, (xt_r, int(0.61*imshape[0])), (xb_r, int(0.97*imshape[0])), color, thickness)


def split_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # At the bottom of the image, imshape[0] and top has been defined as 330
    imshape = img.shape 
    
    slope_left=0
    slope_right=0
    leftx=0
    lefty=0
    rightx=0
    righty=0
    i=0
    j=0
    
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope >0.1: #Left lane and not a straight line
                # Add all values of slope and average position of a line
                slope_left += slope 
                leftx += (x1+x2)/2
                lefty += (y1+y2)/2
                i+= 1
            elif slope < -0.2: # Right lane and not a straight line
                # Add all values of slope and average position of a line
                slope_right += slope
                rightx += (x1+x2)/2
                righty += (y1+y2)/2
                j+= 1
    # Left lane - Average across all slope and intercepts
    if i>0: # If left lane is detected
        avg_slope_left = slope_left/i
        avg_leftx = leftx/i
        avg_lefty = lefty/i
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_l = int(((int(0.97*imshape[0])-avg_lefty)/avg_slope_left) + avg_leftx)
        xt_l = int(((int(0.61*imshape[0])-avg_lefty)/avg_slope_left)+ avg_leftx)

    else: # If Left lane is not detected - best guess positions of bottom x and top x
        xb_l = int(0.21*imshape[1])
        xt_l = int(0.43*imshape[1])
    
    # Draw a line
    #cv2.line(img, (xt_l, int(0.61*imshape[0])), (xb_l, int(0.97*imshape[0])), color, thickness)
    yb = imshape[0]
    yt = yb/2
    left_line = Line(xb_l,yb,xt_l,yt)
    #Right lane - Average across all slope and intercepts
    if j>0: # If right lane is detected
        avg_slope_right = slope_right/j
        avg_rightx = rightx/j
        avg_righty = righty/j
        # Calculate bottom x and top x assuming fixed positions for corresponding y
        xb_r = int(((int(0.97*imshape[0])-avg_righty)/avg_slope_right) + avg_rightx)
        xt_r = int(((int(0.61*imshape[0])-avg_righty)/avg_slope_right)+ avg_rightx)
    
    else: # If right lane is not detected - best guess positions of bottom x and top x
        xb_r = int(0.89*imshape[1])
        xt_r = int(0.53*imshape[1])
    
    # Draw a line    
    #cv2.line(img, (xt_r, int(0.61*imshape[0])), (xb_r, int(0.97*imshape[0])), color, thickness)
    right_line = Line(xb_r,yb,xt_r,yt)
    return left_line,right_line


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# TODO: Build your pipeline that will draw lane lines
def lane_detector(image):
    gray = grayscale(image)
    #print(image.shape)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 10
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Create masked edges image
    imshape = image.shape
    vertices = np.array([[(int(0*imshape[1]),imshape[0]),(int(0*imshape[1]), 
                    int(0.25*imshape[0])), (int(0.80*imshape[1]), int(0.3*imshape[0])), (
                    imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)


    # Define the Hough transform parameters and detect lines using it
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = (np.pi/180) # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 60 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    final_img = weighted_img(line_img, image, alpha=0.6, beta=1., lamda=0.)
    return edges, masked_edges, final_img


#########################################################################3
#Not stable
############################################################################
def split_hough_point(lines_update):
        try:
            lines_update = np.array(lines_update)
            #print(lines_update.shape,'update lines')
            xx = lines_update[:,:,0:2]
            xx = xx.reshape(len(lines_update),2)
            yy = lines_update[:,:,2:4]
            yy = yy.reshape(len(lines_update),2)

            left_line = Line(xx)
            right_line = Line(yy)
            return left_line, right_line
        except:
            return
def find_slopes_lines(image,polygons):
    canny_image = canny(image)
    
    crop_image, _ = region_of_interest(canny_image,polygons)
    lines = cv2.HoughLinesP(crop_image,2,np.pi/180,50,np.array([]),minLineLength=20,maxLineGap=100)
    ## y=mx+b
    line_image=np.zeros_like(image)
    #print(lines.shape)
    x = lines[:,:,0:2]
    x = x.reshape(len(lines),2)
    y = lines[:,:,2:4]
    y = y.reshape(len(lines),2)
    
    slopes = []
    lines_update = []
    threshold = 0.35
    for l in range(len(lines)):
        if (y[l][0]-x[l][0] < 0.1):
            slope = 1000 # //oy
        else:
            slope = (y[l][1]-x[l][1])/(y[l][0]-x[l][0])

        if (abs(slope) > threshold):
            slopes.append(slope)
            lines_update.append(lines[l])
    return slopes,lines_update
def detect_left_right(image,polygons):
        

    slopes,lines_update = find_slopes_lines(image,polygons)
    #print(slopes)
    #print(len(lines_update),'lines_update')
    try:         
        first,second = split_hough_point(lines_update)
    except:
        return
    center = image.shape[1]/2
    right_lines = []
    left_lines = []
    tag = []
    #global tagleft,tagright
    for l in range(len(lines_update)):
        #print(l,': Index')
        if ((slopes[l] > 0) and (first[l][0] > center) and (second[l][0] > center)) :
            right_lines.append(lines_update[l])
            tagright = 1;
            tagleft = 0;
        elif ((slopes[l] < 0)  and (first[l][0] < center) and (second[l][0] < center-10)) :
            left_lines.append(lines_update[l])
            tagleft = 1;
            tagright = 0;
    right_lines = np.array(right_lines)
#     right_lines = right_lines.reshape(right_lines.shape[0],4)
    left_lines  = np.array(left_lines)
#     left_lines  = left_lines.reshape(left_lines.shape[0],4)
    return right_lines,left_lines
