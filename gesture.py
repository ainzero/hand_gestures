from collections import deque
import numpy as np
import cv2

DEBUG = True

def get_skin_mask(frame):
    # change colorspaces
    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    
    cv2.blur(ycrcb, (3,3))

    cr_l = cv2.getTrackbarPos('Cr_lower','gesture')
    cr_u = cv2.getTrackbarPos('Cr_upper','gesture')
    cb_l = cv2.getTrackbarPos('Cb_lower','gesture')
    cb_u = cv2.getTrackbarPos('Cb_upper','gesture')

    dilation_amount = cv2.getTrackbarPos('Dil_Number','gesture')
    errosion_amount = cv2.getTrackbarPos('Err_Number','gesture')
    
    # Set threshold values
    lower_bound= np.array([0,cr_l,cb_l],dtype="uint8") # 133 108 
    upper_bound = np.array([255,cr_u,cb_u],dtype="uint8") # 165,161

    # mask according to bounds
    skin_mask = cv2.inRange(ycrcb, lower_bound, upper_bound)

    # this seems to work well for lab lighting conditions!
    # dilation works well with Ellipse and Rect!
    kernel_for_errosion = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel_for_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # great results with 5 errosions and 6 dilations, using 5x5 kernels
    skin_mask = cv2.erode(skin_mask, kernel_for_errosion, iterations = errosion_amount)
    skin_mask = cv2.dilate(skin_mask, kernel_for_dilation, iterations = dilation_amount)

    return skin_mask


def find_index_of_largest_contour(contours):
    max_contour_index = 0
    max_contour_area = -1
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i])
        if a > max_contour_area:
            max_contour_index = i
            max_contour_area = a

    return max_contour_index


def hand_comparison(frame,box_x,box_y,box_width,box_height):
    hand = get_roi(frame,box_x,box_y,box_width,box_height)

    binary_hand_image = get_skin_mask(hand)

    _ ,hand_contours,_ = cv2.findContours(binary_hand_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if (len(hand_contours) > 1):

        max_contour_index = find_index_of_largest_contour(hand_contours)

        recognition_threshold = cv2.getTrackbarPos('Rec_Thresh','gesture') * .01

        recognition_probability_five = cv2.matchShapes(FIVE_FINGER_CONTOUR,hand_contours[max_contour_index],1,0)
        recognition_probability_peace = cv2.matchShapes(PEACE_CONTOUR,hand_contours[max_contour_index],1,0)

        prob_dict = {recognition_probability_peace:"Hand shape is HAND_PEACE_SIGN",recognition_probability_five:"Hand shape is HAND_5_FINGERS"}

        recognition_probability_key = min(prob_dict)

        if DEBUG:
            print(prob_dict)
            print(recognition_probability_key)
            print(recognition_threshold)

        if recognition_probability_key <= recognition_threshold:
            result = prob_dict[recognition_probability_key]
        else:
            result = "Hand shape is UNKNOWN"

        return result

    return "Hand shape NOT FOUND"

def get_roi(frame,box_x,box_y,box_width,box_height):
    # create a mask
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask[box_y+1:box_height,box_x+1:box_width] = 255
    roi = cv2.bitwise_and(frame, frame, mask = mask)

    return roi

def nothing(x):
    pass

def find_palm_and_draw(frame,box_x,box_y,box_width,box_height):
    
    palm_threshold = .04
    center_point = (0,0)

    hand = get_roi(frame,box_x,box_y,box_width,box_height)
    binary_hand_image = get_skin_mask(hand)
    _ ,hand_contours,_ = cv2.findContours(binary_hand_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if (len(hand_contours) > 0):
        max_contour_index = find_index_of_largest_contour(hand_contours)
        hand_contour = hand_contours[max_contour_index]

        x,y,width,height = cv2.boundingRect(hand_contour)

        max_dimension = 0
        for y in range(int(y+0.7*height),int(y+0.85*height)):
            for x in range(int(x+0.3*width),int(x+0.6*width)):
                distance = cv2.pointPolygonTest(hand_contour,(x,y),True)
                if(distance>max_dimension):
                    max_dimension = distance
                    center_point=(x,y)
        
        if(max_dimension > palm_threshold * frame.shape[1]):
            cv2.circle(frame,center_point,int(max_dimension),(0,255,0),2)
            cv2.circle(frame,center_point,1,(0,255,0),2)
            points_to_draw.appendleft(center_point)
        
        return frame

    return frame

def draw_contrail(frame,width,color):
    for i in range(1, len(points_to_draw)):
        if points_to_draw[i-1] is None or points_to_draw[i] is None:
            continue
        cv2.line(frame, points_to_draw[i-1], points_to_draw[i], color, width)


# create window
cv2.namedWindow("gesture")

# trackbars for skin detection parameters
cv2.createTrackbar('Cr_lower','gesture',133,255,nothing)
cv2.createTrackbar('Cr_upper','gesture',165,255,nothing)
cv2.createTrackbar('Cb_lower','gesture',108,255,nothing)
cv2.createTrackbar('Cb_upper','gesture',161,255,nothing)

cv2.createTrackbar('Dil_Number','gesture',6,10,nothing)
cv2.createTrackbar('Err_Number','gesture',5,10,nothing)

cv2.createTrackbar('Rec_Thresh','gesture',20,100,nothing)


# prepare templates for comparisons
five_fingers = cv2.imread('hand_template.png',2)
_ ,five_finger_contours,_ = cv2.findContours(five_fingers,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
max_contour_index = find_index_of_largest_contour(five_finger_contours)
FIVE_FINGER_CONTOUR = five_finger_contours[max_contour_index]

peace = cv2.imread('hand_peace.png',2)
_ ,peace_contours,_ = cv2.findContours(peace,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
max_contour_index = find_index_of_largest_contour(peace_contours)
PEACE_CONTOUR = peace_contours[max_contour_index]        

points_to_draw = deque(maxlen=16)

static_gesture_capture = False
dynamic_gesture_capture = False
showing_skin_mask = False

# Drawing Variables
red = 0
green = 0
blue = 255
thickness = 4

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Get the dimensions of the handbox
    hand_area_x = int(frame.shape[1]/5)
    hand_area_y = int(frame.shape[0]/8)
    hand_area_width = int(frame.shape[1]/5)+350
    hand_area_height = int(frame.shape[0]/5)+350

    # Draw hand box
    cv2.rectangle(frame,(hand_area_x,hand_area_y), 
        (hand_area_width,hand_area_height), (0,255,0))
    

    if not (static_gesture_capture or dynamic_gesture_capture):
        cv2.putText(frame,"Press 'T' to recognize hand shape. Press 'M' to enable palm tracking.",(5,25), font, .55,(0,255,0),2,cv2.LINE_AA)
    elif static_gesture_capture:
        cv2.putText(frame,result,(35,25), font, 1,(0,255,0),2,cv2.LINE_AA)
    else:
        cv2.putText(frame,"Palm drawing enabled.",(160,25), font, .75,(0,255,0),2,cv2.LINE_AA)

    if showing_skin_mask:
        frame = get_skin_mask(frame)

    if dynamic_gesture_capture:
        frame = find_palm_and_draw(frame,hand_area_x,hand_area_y,hand_area_width,hand_area_height)
        draw_contrail(frame,thickness,(blue,green,red))


    key_press=cv2.waitKey(10)
    
    if key_press & 0xFF == ord('q'):
        break
    elif key_press & 0xFF == ord('m'):
        if dynamic_gesture_capture:
            dynamic_gesture_capture = False
        else:
            dynamic_gesture_capture = True
    elif key_press & 0xFF == ord('s'):
        static_gesture_capture = False
        dynamic_gesture_capture = False
        if showing_skin_mask:
            showing_skin_mask = False
        else:
            showing_skin_mask = True
    elif key_press & 0xFF == ord('t'):
        dynamic_gesture_capture = False
        if not showing_skin_mask:
            result = hand_comparison(frame, hand_area_x,hand_area_y,hand_area_width,hand_area_height)
            static_gesture_capture = True
    # Display the resulting frame
    cv2.imshow('image',frame)



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()