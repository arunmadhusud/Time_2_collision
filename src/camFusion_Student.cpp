
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // 3D homogeneous coordinate
    cv::Mat X = cv::Mat(4,1,CV_64F);
    // homogeneous pixel coordinates
    cv::Mat Y = cv::Mat(3,1,CV_64F);

    // loop over all the LiDar points and associate them to a 2D bounding box
    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        X.at<double>(0,0) = it1->x;
        X.at<double>(1,0) = it1->y;
        X.at<double>(2,0) = it1->z;
        X.at<double>(3,0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        // pixel co-ordinates
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

        std::vector<int> boxIDs;

        for (auto it = boundingBoxes.begin(); it != boundingBoxes.end(); it++)
        {
            cv::Rect s_roi; // shrinked bounding box
            s_roi.x = (it->roi.x) + shrinkFactor * (it->roi.width) / 2.0;
            s_roi.y = (it->roi.y) + shrinkFactor * (it->roi.height) / 2.0;
            s_roi.width = it->roi.width * (1 - shrinkFactor);
            s_roi.height = it->roi.height * (1 - shrinkFactor);

            if (s_roi.contains(pt)){
                boxIDs.push_back(it->boxID);
            }
        }
        if (boxIDs.size()==1)
        {
            boundingBoxes[boxIDs[0]].lidarPoints.push_back(*it1);
        }
 
    }

}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
   // create topview image
   cv::Mat topviewImg(imageSize,CV_8UC3,cv::Scalar(255,255,255));

   for(auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
   {
        // create random color for the current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0,150), rng.uniform(0,150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0; // top, left, bottom, right of the bounding box in the top view
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8; // x, y min and max of the bounding box in the world

        // loop over all Lidar points within the bounding box and plot them into the top view
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = it2->x; // world position in m with x facing forward from the sensor
            float yw = it2->y; // world position in m with y facing left from the sensor

            //update the min and max of the bounding box
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top view coordinates
            // scale the x, y world coordinates to match the image size and cordinates are inverted to match the top down perspective
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), currColor, 2);

        // augment object with some key data
        // char str1[200], str2[200];
        std::string str1 = cv::format("id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        std::string str2 = cv::format("x=%2.2f, y=%2.2f", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; i++)
    {
        int y = (-lineSpacing * i * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255,0,0));
    }

    // display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}



// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> matchesInBox; // temp vector to store the matches within the bounding box
    std::vector<double> matchesDistance; // temp vector to store the euclidean distances of the matches within the bounding box

    // loop the keypoint matches and check if it's within the bounding box
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        cv::KeyPoint kptPrev = kptsPrev[it->queryIdx];
        cv::KeyPoint kptCurr = kptsCurr[it->trainIdx];

        if (boundingBox.roi.contains(kptCurr.pt))
        {
            matchesInBox.push_back(*it);
            matchesDistance.push_back(cv::norm(kptCurr.pt - kptPrev.pt)); // calculate the euclidean distance
        }
    }

    // filter the matches whose euclidean distances are too far away from the mean value
    double meanDistance = std::accumulate(matchesDistance.begin(), matchesDistance.end(), 0.0) / matchesDistance.size();
    for (int idx = 0; idx < matchesDistance.size(); ++idx)
    {
        if (matchesDistance[idx] < meanDistance) // if the distance is less than the mean value, then keep it
        {
            boundingBox.keypoints.push_back(kptsCurr[matchesInBox[idx].trainIdx]); // store the keypoint
            boundingBox.kptMatches.push_back(matchesInBox[idx]); // store the match
        }
    }

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    } // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // // compute camera-based TTC from distance ratios
    // double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    // double dT = 1 / frameRate;
    // TTC = -dT / (1 - meanDistRatio);

    // using medianDistRatio is more robust than meanDistRatio
    std::sort(distRatios.begin(), distRatios.end());
    long medianIndex = floor(distRatios.size() / 2.0);
    double medianDistRatio = (distRatios.size() % 2 == 0) ? (distRatios[medianIndex-1] + distRatios[medianIndex]) / 2.0 : distRatios[medianIndex];

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // auxiliary parameters
    double dT = 1.0 / frameRate;    // delta time between two frames in seconds
    double laneWidth = 4.0;         // assumed width of the ego lane

    // to eliminate the outliers, create a vector of cloest Lidar points, the number of the canadicates is a percentage of total Lidar points
    double percentage = 0.2;        // 20% of the Lidar points will be used to calculate the cloest distance
    int numMinsPrev = lidarPointsPrev.size() * percentage;
    int numMinsCurr = lidarPointsCurr.size() * percentage;

    // If numer of Lidar points are small, then use all Lidar points for estimation
    if (lidarPointsPrev.size() < 10)
    {
        std::cout << "WARNING: Too few Lidar points from previous frame for TTC estimation!" << std::endl;
        numMinsPrev = lidarPointsPrev.size();
    }
    if (lidarPointsCurr.size() < 10)
    {
        std::cout << "WARNING: Too few Lidar points from current frame for TTC estimation!" << std::endl;
        numMinsCurr = lidarPointsCurr.size();
    }

    // calculate the average Lidar points distance to the preceding vehicle within ego lane
    double avgXPrev = 1e9, avgXCurr = 1e9;
    vector<double> minXPrev, minXCurr;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        // 3D pionts within the ego-lane
        if (abs(it->y) <= (laneWidth/2.0))
        {
            if (minXPrev.size() < numMinsPrev)
            {
                minXPrev.push_back(it->x);

            }
            else
            {
                auto item = max_element(std::begin(minXPrev), std::end(minXPrev));
                if (it->x < *item)
                {
                    minXPrev.erase(item);
                    minXPrev.push_back(it->x);
                }
            }
        }
    }
    avgXPrev = std::accumulate(minXPrev.begin(), minXPrev.end(), 0.0) / numMinsPrev;

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        // 3D pionts within the ego-lane
        if (abs(it->y) <= (laneWidth/2.0))
        {
            if (minXCurr.size() < numMinsCurr)
            {
                minXCurr.push_back(it->x);

            }
            else
            {
                auto item = max_element(std::begin(minXCurr), std::end(minXCurr));
                if (it->x < *item)
                {
                    minXCurr.erase(item);
                    minXCurr.push_back(it->x);
                }
            }
        }
    }
    avgXCurr = std::accumulate(minXCurr.begin(), minXCurr.end(), 0.0) / numMinsCurr;

    // compute TTC from two frames
    TTC = avgXCurr * dT / (avgXPrev - avgXCurr);

    bool bDebug = false;
    if (bDebug)
    {
        std::cout << "avgXPrev= " << avgXPrev << std::endl;
        std::cout << "avgXCurr= " << avgXCurr << std::endl;
        std::cout << "TTC = " << TTC << std::endl;
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Create a 2D vector to store match counts between bounding boxes
    std::vector<std::vector<int>> matchMatrix(prevFrame.boundingBoxes.size(), std::vector<int>(currFrame.boundingBoxes.size(),0));
    
    // Iterate through all matches
    for (auto match : matches){
        cv::KeyPoint prevkpt = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currkpt = currFrame.keypoints[match.trainIdx];

        for (size_t i = 0; i < prevFrame.boundingBoxes.size(); i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prevkpt.pt))
            {
                for (size_t j = 0; j < currFrame.boundingBoxes.size(); j++)
                {
                    if (currFrame.boundingBoxes[j].roi.contains(currkpt.pt))
                    {
                         matchMatrix[i][j]++;
                    }
                }
            }
        }
    }

    // find best matches
    for (size_t i = 0; i < prevFrame.boundingBoxes.size(); i++)
    {
        int maxCount = 0;
        int bestMatch = -1;
        for (size_t j = 0; j < currFrame.boundingBoxes.size(); j++)
        {
            if (matchMatrix[i][j] > maxCount)
            {
                maxCount = matchMatrix[i][j];
                bestMatch = j;
            }
        }
        bbBestMatches[prevFrame.boundingBoxes[i].boxID] = 
            (bestMatch!=-1) ? currFrame.boundingBoxes[bestMatch].boxID : -1;

    } 
    
}

