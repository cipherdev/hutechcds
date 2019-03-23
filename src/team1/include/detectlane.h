#ifndef DETECTLANE_H
#define DETECTLANE_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <vector>
#include <math.h>
#include <algorithm>

using namespace std;
using namespace cv;

class DetectLane
{
public:
    DetectLane();
    ~DetectLane();

    void update(const Mat &src);
    
    vector<Point> getLeftLane();
    vector<Point> getRightLane();

    static int slideThickness;

    static int BIRDVIEW_WIDTH;
    static int BIRDVIEW_HEIGHT;

    static int VERTICAL;
    static int HORIZONTAL;

    static Point null; // 

private:
    Mat preProcess(const Mat &src);

    Mat morphological(const Mat &imgHSV);
    Mat birdViewTranform(const Mat &source);
    void fillLane(Mat &src);
    vector<Mat> splitLayer(const Mat &src, int dir = VERTICAL);
    vector<vector<Point> > centerRoadSide(const vector<Mat> &src, int dir = VERTICAL);
    void detectLeftRight(const vector<vector<Point> > &points);
    Mat laneInShadow(const Mat &src);
    Mat combinedShadowAndNoShadow(const Mat& src, Mat& dest);


    int minThreshold[3] = {0, 0, 180};
    int maxThreshold[3] = {179, 30, 255};
    int minShadowTh[3] = {92, 47, 96};
    int maxShadowTh[3] = {111, 91, 130};
    int minLaneInShadow[3] = {32, 8, 97};
    int maxLaneInShadow[3] = {130, 149, 171};
    int binaryThreshold = 180;

    int skyLine = 120;
    int shadowParam = 40;

    
    // int minThreshold[3] = {0, 0, 180};
    // int maxThreshold[3] = {179, 0, 255};
    // int minShadowTh[3] = {114, 49, 82};
    // int maxShadowTh[3] = {124, 139, 136};
    // int minLaneInShadow[3] = {13, 33, 71};
    // int maxLaneInShadow[3] = {125, 118, 155};
    // int binaryThreshold = 180;

    // int skyLine = 120;
    // int shadowParam = 40;


    vector<Point> leftLane, rightLane;
};

#endif
