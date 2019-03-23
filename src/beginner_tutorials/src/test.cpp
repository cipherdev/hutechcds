
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include "std_msgs/Float32.h"

#include <vector>
#include <math.h>

#include <iostream>

class controller
{
  public:
    ros::NodeHandle node_obj1;
    ros::NodeHandle node_obj2;

    ros::Publisher steer_publisher;
    ros::Publisher speed_publisher;
    controller()
    {

        steer_publisher = node_obj1.advertise<std_msgs::Float32>("team1_steerAngle", 10);
        speed_publisher = node_obj2.advertise<std_msgs::Float32>("team1_speed", 10);
    }
};
int main(int argc, char **argv)
{

    ros::init(argc, argv, "controller_node");
    controller con;
    std_msgs::Float32 angle;
    std_msgs::Float32 speed;
    float error, velocity;
    char a;
    while (true)
    {
        std::cin >> a;
        if (a == 'a')
        {
            angle.data = -50;
            speed.data = 30;
        }

        if (a == 'd')
        {
            angle.data = 50;
            speed.data = 30;
        }

        if(a == 's'){
            speed.data = 10;
        }
        if (a == 'x')
            return 0;
        ROS_ERROR_STREAM(a);
        for (int i = 0; i < 300; ++i)
        {
            con.steer_publisher.publish(angle);
            con.speed_publisher.publish(speed);
        }
    }
}
