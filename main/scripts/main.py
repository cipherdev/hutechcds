#!/usr/bin/env python3

import sys
import numpy as np
import cv2
import rospy
from sensor_msgs.msg import CompressedImage

from detect_all import Detect


###########################################################################

class image_feature:
    def __init__(self):
        self.subscriber = None

    def init(self):
        """Initialize ros publisher, ros subscriber"""
        self.subscriber = rospy.Subscriber("Team1_image/compressed",
                                           CompressedImage, self.callback)

    @staticmethod
    def callback(ros_data):
        """Callback function of subscribed topic.  Here images get converted and features detected"""
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # cv2.CV_LOAD_IMAGE_COLOR)
        # cv2.imshow('view', image_np)        
        # k = cv2.waitKey(1)
        # traffic_image, direction = TD.detect_sign(image_np)        

        # final_image, _ = TD.detect_sign(image_np)
        detect1 = Detect(image_np)
        final_image = detect1.drive()
        cv2.imshow("Final image", final_image)
        k = cv2.waitKey(1)
        if k == ord('q'):  # wait for ESC key to exit
            cv2.destroyAllWindows()
            rospy.signal_shutdown('Exit')


def main(args):
    """Initializes and cleanup ros node"""
    rospy.init_node('image_feature')
    rospy.myargv(argv=args)

    ifeature = image_feature()
    ifeature.init()

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
