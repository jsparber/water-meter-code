import sys, getopt
import cv2
import numpy as np
import time
import os
from datetime import datetime

def get_angle(hand,centre):
    x_h=hand[0]
    y_h=hand[1]
    x_c=centre[0]
    y_c=centre[1]

    x_diff=x_h-x_c
    y_diff=y_h-y_c
    x_diff=float(x_diff)
    y_diff=float(y_diff)

    if(x_diff*y_diff>0):
        if(x_diff>=0 and y_diff>0):
            angle=np.pi-np.arctan(x_diff/y_diff)
        elif(x_diff<=0 and y_diff<0):
            angle=2*np.pi-np.arctan(x_diff/y_diff)
    elif(x_diff*y_diff<0):
        if(y_diff>=0 and x_diff<0):
            angle=(3*np.pi)/4+np.arctan(x_diff/y_diff)
        elif(y_diff<=0 and x_diff>0):
            angle=-np.arctan(x_diff/y_diff)

    return angle


def dist_2_pts(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calibrate_gauge(inputfile):
    img = cv2.imread(inputfile)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 30, 30, 35, 50)

    a, b, c = circles.shape
    x = 0
    i = 0
    for i in range(b):
        if (x < int(circles[0][i][0])):
            x = int(circles[0][i][0])
            y = int(circles[0][i][1])
            r = int(circles[0][i][2])
            
        center = (int(circles[0][i][0]), int(circles[0][i][1]))
        radius = int(circles[0][i][2])
        #cv2.circle(gray, center, radius, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle

    x2 = 0
    for i in range(b):
        if (x2 < int(circles[0][i][0]) and int(circles[0][i][0]) < x):
            x2 = int(circles[0][i][0])
            y2 = int(circles[0][i][1])
            r2 = int(circles[0][i][2])
     
    #cv2.imwrite('circles.jpg', gray)

    #draw center and circle
    #cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    #cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    # Draw secound circle
    #cv2.circle(img, (x2, y2), r2, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    #cv2.circle(img, (x2, y2), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    #for testing, output circles on image
    #cv2.imwrite('color-circles.jpg', img)

    return x, y, r, x2, y2, r2

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, inputfile):
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 175
    maxValue = 255

    # apply thresholding which helps for finding lines
    th, dst2 = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV);
    dst2 = cv2.GaussianBlur(dst2, (5, 5), 0)

    #print "radius: %s" %r

    # find pointer arrow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Range for lower red
    lower_red = np.array([0, 30, 30])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170, 30, 30])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final mask to detect red color
    mask = mask1+mask2
    cv2.imwrite("mask.jpg", mask)

    res = cv2.bitwise_and(img, img, mask = mask)
    edges = cv2.Canny(res, 100, 400, apertureSize=5)

    #Getting and Displaying Contours
    im2,contours,hierarchy=cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(img, contours,-1,(255, 255, 0), 1)

    #Getting Contours around the centre point
    shortlist=[]
    for i in contours:
        bx, by, bw, bh = cv2.boundingRect(i)
        if (x < (bx+bw) and x > bx) and (y > by and y < by+bh):
            shortlist.append(i)

    #cv2.drawContours(img, shortlist[0],-1,(255, 255, 0), 1)

    #cv2.imwrite("output-%s" %inputfile, img)

    #Getting and Clustering Hull Points
    hull = cv2.convexHull(shortlist[0])

    mindist = 0
    x1 = 0
    y1 = 0
    hand_points=[]
    #for i in hull_points_clustered:
    for i in hull:
        xb=i[0][0]
        yb=i[0][1]
        #cv2.circle(img,(xb,yb), 2, (0, 0, 255), 1)
        dist = dist_2_pts (xb, yb, x, y)
        if (mindist < dist):
            mindist = dist
            x1 = xb
            y1 = yb


    #cv2.line(img, (x1,y1), (x, y), (0, 255, 0), 1)

    #draw center and circle
    #cv2.circle(img, (x, y), r, (0, 0, 255), 1, cv2.LINE_AA)  # draw circle
    #cv2.circle(img, (x, y), 2, (0, 255, 0), 1, cv2.LINE_AA)  # draw center of circle

    #cv2.imwrite("output-%s" %inputfile, img)

    x_angle = x1 - x
    y_angle = y - y1
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))

    #these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  #in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  #in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  #in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  #in quadrant IV
        final_angle = 270 - res

    # 180 degress is 0
    old_value = (final_angle + 180) % 360

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o",["ifile="])
    except getopt.GetoptError:
        print 'python2.7 analog_gauge_reader.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'python2.7 analog_gauge_reader.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    if (inputfile == ''):
        print 'python2.7 analog_gauge_reader.py -i <inputfile>'
        sys.exit(2)

#    print 'Input file is ', inputfile

    # Find the correct circle
    x, y, r, x2, y2, r2 = calibrate_gauge(inputfile)

    img = cv2.imread(inputfile)
    val = get_current_value(img, 0, 360, 0, 10, x, y, r, inputfile)

    # This is the secound gauge showing 10l
    val2 = get_current_value(img, 0, 360, 0, 10, x2, y2, r2, inputfile)

    base = os.path.basename(inputfile)
    filename = os.path.splitext(base)[0]
    date = datetime.strptime(filename, '%Y%m%d%H%M')

    '''
    Output format has to look like this
    { "date": "2014-01-01",
      "value1": 2 // * 100
      "value2": 2 // * 10
    },
    '''
    final_value = (int(val) * 100) + (round(val2, 1) * 10)
    print ("{\"date\": \"%s\", \"value1\": %s, \"value2\": %s}," %(date, round (val, 3), round (val2, 3)))

if __name__=='__main__':
    main(sys.argv[1:])
   	
