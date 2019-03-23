#include "carcontrol.h"
#include <ros/console.h>
#include <fstream>

// #include <iostream>
CarControl::CarControl()
{
    carPos.x = 120;
    carPos.y = 300;
    steer_publisher = node_obj1.advertise<std_msgs::Float32>("team1_steerAngle", 10);
    speed_publisher = node_obj2.advertise<std_msgs::Float32>("team1_speed", 10);
}

CarControl::~CarControl() {}

float CarControl::errorAngle(const Point &dst)
{
    if (dst.x == carPos.x)
        return 0;
    if (dst.y == carPos.y)
        return (dst.x < carPos.x ? -90 : 90);
    double pi = acos(-1.0);
    double dx = dst.x - carPos.x;
    double dy = carPos.y - dst.y;
    if (dx < 0)
        return -atan(-dx / dy) * 180 / pi;
    return atan(dx / dy) * 180 / pi;
}

void CarControl::driverCar(const vector<Point> &left, const vector<Point> &right, float velocity)
{
    int checkpoints = 0;
    int i = left.size() - 11;
    float error = preError;

    while (left[i] == DetectLane::null && right[i] == DetectLane::null)
    {
        i--;
        if (i < 0)
        {
            return;
        }
    }

    if(left[0] != DetectLane::null && errorAngle(left[0] > 0){
        velocity = 10;
        error = 
    }

    if (left[i] != DetectLane::null && right[i] != DetectLane::null)
    {
        //ROS_ERROR_STREAM(left[i].x << " " << left[i].y << " "<< right[i].x << " "<<  right[i].y << " ");
        error = errorAngle((left[i] + right[i]) / 2);
    }
    else if (left[i] != DetectLane::null)
    {
        // ROS_ERROR_STREAM(left[i].x << " " << left[i].y << " "<< right[i].x << " "<<  right[i].y << " ");
        error = errorAngle(left[i] + Point(laneWidth / 2, 0));
        // error = errorAngle(left[i]);
    }
    else if (right[i] != DetectLane::null)
    {
        //ROS_ERROR_STREAM(left[i].x << " " << left[i].y << " "<< right[i].x << " "<<  right[i].y << " ");

        error = errorAngle(right[i] - Point(laneWidth / 2, 0));
        // error = errorAngle(right[i]);
    }

    std_msgs::Float32 angle;
    std_msgs::Float32 speed;

    // if (abs(error - preError) > 20)
    // {
    //     velocity = 10;
    // }

    angle.data = error;
    speed.data = velocity;
    // ROS_ERROR_STREAM(error << " " << velocity);
    steer_publisher.publish(angle);
    speed_publisher.publish(speed);

    preError = error;
}

// void CarControl::driverCar(const vector<Point> &left, const vector<Point> &right, float velocity, bool &inIntersection, int &direction)
// {
//     int checkpoints = 0;
//     int i = left.size() - 11;
//     float error = preError;

//     // if (left.size() <= int(0.4 * right.size()))
//     // {
//     //     velocity = 10;
//     //     error = -50;
//     // }
//     // else if (right.size() <= int(0.4 * left.size()))
//     // {
//     //     velocity = 10;
//     //     error = 50;
//     // }

//     if (!inIntersection)
//     {
//         if (left.size() < 5)
//         {
//             inIntersection = true;
//             direction = 1;
//             return;
//         }
//         else if (right.size() < 5)
//         {
//             inIntersection = true;
//             direction = 2;
//             return;
//         }
//     }
//     else
//     {
//         if (left.size() > 5 && right.size() > 5)
//         {
//             inIntersection = false;
//         }
//         else if (direction == 1)
//         {
//             error = 50;
//             velocity = 10;
//         }
//         else
//         {
//             error = -50;
//             velocity = 10;
//         }
//     }
//     if (!inIntersection)
//     {
//         while (left[i] == DetectLane::null && right[i] == DetectLane::null)
//         {
//             i--;
//             if (i < 0)
//             {
//                 return;
//             }

//             //      std::ifstream file;
//             //      file.open("/home/nghia/CuocDuaSo-Test/logfile_team1.txt");
//             //      string a;
//             //      while(file >> a) checkpoints++;
//             //      file.close();
//             //     // string s;
//             //     // fin >> s;
//             //     // if (s[0] == '1')
//             //     // {
//             //     //     error = 50;
//             //     // }
//             //     // else
//             //     // {
//             //     //     error = -50;
//             //     // }

//             //     // fin.close();

//             //     // velocity = 20;

//             //     // std_msgs::Float32 angle;
//             //     // std_msgs::Float32 speed;

//             //     // angle.data = error;
//             //     // speed.data = velocity;
//             //     // ROS_ERROR_STREAM(error << " " << velocity);
//             //     // steer_publisher.publish(angle);
//             //     // speed_publisher.publish(speed);
//             //     ///return;
//             // }
//             // if(checkpoints == 0){
//             //     velocity = 40;
//             //     error = 50;
//             // }
//             // else
//             // if(checkpoints == 1){
//             //     velocity = 40;
//             //     error = -50;
//             // }
//         }
//         if (left[i] != DetectLane::null && right[i] != DetectLane::null)
//         {
//             //ROS_ERROR_STREAM(left[i].x << " " << left[i].y << " "<< right[i].x << " "<<  right[i].y << " ");
//             error = errorAngle((left[i] + right[i]) / 2);
//         }
//         else if (left[i] != DetectLane::null)
//         {
//             // ROS_ERROR_STREAM(left[i].x << " " << left[i].y << " "<< right[i].x << " "<<  right[i].y << " ");
//             error = errorAngle(left[i] + Point(laneWidth / 2, 0));
//         }
//         else if (right[i] != DetectLane::null)
//         {
//             //ROS_ERROR_STREAM(left[i].x << " " << left[i].y << " "<< right[i].x << " "<<  right[i].y << " ");

//             error = errorAngle(right[i] - Point(laneWidth / 2, 0));
//         }
//     }

//     std_msgs::Float32 angle;
//     std_msgs::Float32 speed;

//     if (abs(error - preError) > 20)
//     {
//         velocity = 10;
//     }

//     angle.data = error;
//     speed.data = velocity;
//     // ROS_ERROR_STREAM(error << " " << velocity);
//     steer_publisher.publish(angle);
//     speed_publisher.publish(speed);

//     preError = error;
// }
