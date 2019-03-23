import rospy
from std_msgs.msg import Image
import PIL
from cvbridge import bridge

import cv2

def main():
    rospy.init_node("sign_detector", anonymous=True)
    sub = rospy.Subscriber("team1_image", Image, callback)
    pub = rospy.Publisher("team1_detected_image", Image, queue_size=10)

    def callback(data):
        img = data.data

        cv_img = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")

        cv_arr = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv_arr)
        pil_im.show()



        #pub.publish(detected_image)


if __name__ == '__main__':
    main()
