#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <numeric>
#include "Supp.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

Mat shapeSegmentation(const Mat&);
Mat redSegmentation(const Mat&);
Mat yellowSegmentation(const Mat&);
Mat blueSegmentation(const Mat&);
string classifyShape(const vector<Point>& contour);
int countColorPixels(const Mat& image, const Scalar& lowerBound, const Scalar& upperBound);
Mat selectSegmentation(const Mat&);
void showWindow(Mat, Mat,String);
void PicSegmentation() {
    Mat srcI, writefile;
    vector<Mat> segmentedImages;
    String path("Inputs/Traffic signs/*.png");
    vector<string> imageNames;
    String outputFolder = "Segmented/";
    glob(path, imageNames, true);
    for (size_t i = 0; i < imageNames.size(); ++i)
    {
        srcI = imread(imageNames[i]);
        if (srcI.empty()) {
            cout << "Cannot open image for reading" << endl;
            continue;
        }
        resize(srcI, srcI, Size(200, 200));

        //Segmentation Process
        writefile = selectSegmentation(srcI);
        segmentedImages.push_back(writefile);
        if (writefile.empty()) {
            cerr << "Error: Segmented image is empty for index " << i << endl;
            continue;
        }
        // Extract the file name without the path
        String fileName = imageNames[i].substr(imageNames[i].find_last_of("/\\") + 1);

        // Generate the output file path
        String outputFilePath = outputFolder + fileName;

        // Save the segmented image to the output folder
        if (!imwrite(outputFilePath, writefile)) {
            cerr << "Error: Failed to save image " << outputFilePath << endl;
        }
        else {
            cout << "Saved segmented image to " << outputFilePath << endl;
        }

    }
}
Mat selectSegmentation(const Mat& image) {
    Mat returnimage;
    // Define the color ranges for red, yellow, and blue
    Scalar lowerRed(0, 0, 100);   // Lower bound for red color
    Scalar upperRed(50, 50, 255); // Upper bound for red color
    Scalar lowerYellow(0, 100, 100); // Lower bound for yellow color
    Scalar upperYellow(50, 255, 255); // Upper bound for yellow color
    Scalar lowerBlue(100, 0, 0);   // Lower bound for blue color
    Scalar upperBlue(255, 50, 50); // Upper bound for blue color

    // Count the number of pixels in each color range
    int redCount = countColorPixels(image, lowerRed, upperRed);
    int yellowCount = countColorPixels(image, lowerYellow, upperYellow);
    int blueCount = countColorPixels(image, lowerBlue, upperBlue);

    // Print the pixel counts for debugging
    cout << "Red Pixel Count: " << redCount << endl;
    cout << "Yellow Pixel Count: " << yellowCount << endl;
    cout << "Blue Pixel Count: " << blueCount << endl;

    // Choose the color segmentation function based on the pixel counts
    if (redCount > yellowCount || redCount > blueCount) {
        returnimage = redSegmentation(image);
    }
    else if (yellowCount > redCount || yellowCount > blueCount) {
        returnimage = yellowSegmentation(image);
    }
    else if (blueCount > yellowCount || blueCount > redCount) {
        returnimage = blueSegmentation(image);
    }
    else {
        returnimage = shapeSegmentation(image);
    }
    return returnimage;

}
int countColorPixels(const Mat& image, const Scalar& lowerBound, const Scalar& upperBound) {
    Mat mask;
    // Create a mask where the color is within the specified range
    inRange(image, lowerBound, upperBound, mask);

    // Count the number of non-zero pixels in the mask
    int count = countNonZero(mask);
    return count;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////                        SHAPE DETECTION                     /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
string classifyShape(const vector<Point>& contour) {
    double peri = arcLength(contour, true);
    vector<Point> approx; //to store the vertices of the shape
    approxPolyDP(contour, approx, 0.04 * peri, true); // to approximate the shape of a contour

    if (approx.size() == 3) {
        return "Triangle";
    }
    else if (approx.size() == 4) {
        Rect rect = boundingRect(approx);
        float aspectRatio = (float)rect.width / (float)rect.height;
        return (aspectRatio >= 0.95 && aspectRatio <= 1.05) ? "Square" : "Rectangle";
    }
    else if (approx.size() > 6) {
        return "Circle";
    }
    else {
        return "Unknown";
    }
}

Mat shapeSegmentation(const Mat& src) {

    Mat  clone, grayImage, blurredImage, cannyImage, segmented;
    String shape;

    src.copyTo(clone);

    // 2. Preprocessing: Convert to grayscale and apply Gaussian blur
    cvtColor(src, grayImage, COLOR_BGR2GRAY);
    GaussianBlur(grayImage, blurredImage, Size(5, 5), 0);
    //medianBlur(blurredImage, blurredImage, 5);
    //cvtColor(grayImage, win[1], COLOR_GRAY2BGR);
    //cvtColor(blurredImage, win[2], COLOR_GRAY2BGR);

    // 3. Edge detection using Canny
    Canny(blurredImage, cannyImage, 60, 150);
    //dilate(cannyImage, cannyImage, Mat());
    //erode(cannyImage, cannyImage, Mat());
    //cvtColor(cannyImage, win[3], COLOR_GRAY2BGR);


    // Find contours
    vector<vector<Point>> contours;
    findContours(cannyImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //Find largest contour by Area
    double maxArea = 0.0;
    int index = -1;// in case if no contour detected -> fail
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            index = i;
        }
    }

    // Proceed with shape classification if a valid contour was found
    if (index != -1) {
        const vector<Point>& largestContour = contours[index];
        shape = classifyShape(largestContour);
        // Draw the longest contour and label the shape
        drawContours(src, contours, index, Scalar(0, 255, 0), 2);
        //Rect rect = boundingRect(longestContour); // compute the rectangle that completely encloses a given contour or a set of points.
        //putText(src, shape, Point(rect.x, rect.y +10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);

        // Create a mask for the longest contour
        Mat mask = Mat::zeros(clone.size(), CV_8U);
        drawContours(mask, contours, index, Scalar(255), FILLED);
        cvtColor(mask, mask, COLOR_GRAY2BGR);

        // Perform segmentation by applying the mask to the original image
        segmented = mask & clone;

      //  showWindow(src, segmented, "Segmented");
    }
    else {
        cout << "No contours found!" << endl;
    }

    return segmented;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////                        RED SEGMENTATION                     /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat redSegmentation(const Mat& srcI) {
    string windowName;
    Mat  redMask, canvasColor, canvasGray;
    Mat blueChannel, greenChannel, redChannel;
    Point2i center;
    vector<Scalar> colors;
    int const MAXfPt = 200;
    int t1, t2, t3, t4;
    RNG rng(0);

    // Get MAXfPt random but brighter colors for drawing later
    for (int i = 0; i < MAXfPt; i++) {
        for (;;) {
            t1 = rng.uniform(0, 255); // blue
            t2 = rng.uniform(0, 255); // green
            t3 = rng.uniform(0, 255); // red
            t4 = t1 + t2 + t3;
            if (t4 > 255) break;
        }
        colors.push_back(Scalar(t1, t2, t3));
    }

    //// Open 2 large windows to display the results
    //int const noOfImagePerCol = 2, noOfImagePerRow = 3;
    //Mat detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    //createWindowPartition(srcI, detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

    //putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //putText(legend[1], "redMask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //putText(legend[2], "Contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //putText(legend[3], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //putText(legend[4], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //putText(legend[5], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    //int const noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
    //Mat resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
    //createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

    //putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    //putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    //srcI.copyTo(win[0]);
    //srcI.copyTo(win2[0]);


    // Convert to HSV and use a range of red colors
    Mat hsvImage;
    cvtColor(srcI, hsvImage, COLOR_BGR2HSV);

    // Define red color range in HSV
    Scalar lowerRed1(0, 120, 0);
    Scalar upperRed1(10, 255, 255);
    Scalar lowerRed2(170, 120, 0);
    Scalar upperRed2(220, 255, 255);

    // Create masks for the red color range
    Mat redMask1, redMask2;
    inRange(hsvImage, lowerRed1, upperRed1, redMask1);
    inRange(hsvImage, lowerRed2, upperRed2, redMask2);

    // Combine the masks
    redMask = redMask1 | redMask2;

    // Morphological operations to remove small noise and fill gaps in detected regions
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernel);
    morphologyEx(redMask, redMask, MORPH_OPEN, kernel);

    //// Show result of red color
    //cvtColor(redMask, win[1], COLOR_GRAY2BGR);

    // Create canvases for drawing
    canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
    canvasGray.create(srcI.rows, srcI.cols, CV_8U);
    canvasColor = Scalar(0, 0, 0);

    // Get contours of the red regions
    vector<vector<Point>> contours;
    findContours(redMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int index = -1, max = 0;

    for (int j = 0; j < contours.size(); j++) {
        canvasGray = 0;
        if (max < contours[j].size()) {
            max = contours[j].size();
            index = j;
        }
        drawContours(canvasColor, contours, j, colors[j % MAXfPt]);
        drawContours(canvasGray, contours, j, 255);

        Moments M = moments(canvasGray);
        center.x = M.m10 / M.m00;
        center.y = M.m01 / M.m00;

        floodFill(canvasGray, center, 255);
        //if (countNonZero(canvasGray) > 100) {
        //    sprintf_s(str, "Mask %d (area > 100)", j);
        //    imshow(str, canvasGray);
        //}
    }
    //canvasColor.copyTo(win[2]);
    if (index < 0) {
        waitKey(0);
    }

    canvasGray = 0;
    drawContours(canvasGray, contours, index, 255);
    //cvtColor(canvasGray, win[3], COLOR_GRAY2BGR);

    Moments M = moments(canvasGray);
    center.x = M.m10 / M.m00;
    center.y = M.m01 / M.m00;

    if (center.x >= 0 && center.x < srcI.cols && center.y >= 0 && center.y < srcI.rows) {
        floodFill(canvasGray, center, 255);
    }
    else {
        cerr << "Error: Calculated center is out of bounds!" << endl;
    }
    cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);
    //canvasGray.copyTo(win[4]);

    canvasColor = canvasGray & srcI;
    //canvasColor.copyTo(win[5]);
    //canvasColor.copyTo(win2[1]);

    windowName = "Segmentation  (detail)";
    //imshow(windowName, detailResultWin);
    //imshow("Traffic sign segmentation", resultWin);

   // showWindow(srcI, canvasColor, "Segmented");
    return canvasColor;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////                        YELLOW SEGMENTATION                     /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat yellowSegmentation(const Mat& srcI) {
    string windowName;
    Mat  yellowMask, canvasColor, canvasGray;
    Mat blueChannel, greenChannel, redChannel;
    Point2i center;
    vector<Scalar> colors;
    int const MAXfPt = 200;
    int t1, t2, t3, t4;
    RNG rng(0);

    // Get MAXfPt random but brighter colors for drawing later
    for (int i = 0; i < MAXfPt; i++) {
        for (;;) {
            t1 = rng.uniform(0, 255); // blue
            t2 = rng.uniform(0, 255); // green
            t3 = rng.uniform(0, 255); // red
            t4 = t1 + t2 + t3;
            if (t4 > 255) break;
        }
        colors.push_back(Scalar(t1, t2, t3));
    }

    // Open 2 large windows to display the results
    /*int const noOfImagePerCol = 2, noOfImagePerRow = 3;
    Mat detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    createWindowPartition(srcI, detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

    putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[1], "yellowMask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[2], "Contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[3], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[4], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[5], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    int const noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
    Mat resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
    createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

    putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    srcI.copyTo(win[0]);
    srcI.copyTo(win2[0]);*/

    // Convert to HSV and use a range of yellow colors
    Mat hsvImage;
    cvtColor(srcI, hsvImage, COLOR_BGR2HSV);

    // Define yellow color range in HSV
    Scalar lowerYellow(10, 120, 0);
    Scalar upperYellow(40, 255, 255);

    // Create mask for the yellow color range
    inRange(hsvImage, lowerYellow, upperYellow, yellowMask);

    // Morphological operations to remove small noise and fill gaps in detected regions
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(yellowMask, yellowMask, MORPH_CLOSE, kernel);
    morphologyEx(yellowMask, yellowMask, MORPH_OPEN, kernel);

    // Show result of yellow color
    //cvtColor(yellowMask, win[1], COLOR_GRAY2BGR);

    // Create canvases for drawing
    canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
    canvasGray.create(srcI.rows, srcI.cols, CV_8U);
    canvasColor = Scalar(0, 0, 0);

    // Get contours of the yellow regions
    vector<vector<Point>> contours;
    findContours(yellowMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int index = -1, max = 0;

    for (int j = 0; j < contours.size(); j++) {
        canvasGray = 0;
        if (max < contours[j].size()) {
            max = contours[j].size();
            index = j;
        }
        drawContours(canvasColor, contours, j, colors[j % MAXfPt]);
        drawContours(canvasGray, contours, j, 255);

        Moments M = moments(canvasGray);
        center.x = M.m10 / M.m00;
        center.y = M.m01 / M.m00;

        floodFill(canvasGray, center, 255);
    }
    //canvasColor.copyTo(win[2]);

    if (index < 0) {
        waitKey(0);
    }

    canvasGray = 0;
    drawContours(canvasGray, contours, index, 255);
    //cvtColor(canvasGray, win[3], COLOR_GRAY2BGR);

    Moments M = moments(canvasGray);
    center.x = M.m10 / M.m00;
    center.y = M.m01 / M.m00;

    if (center.x >= 0 && center.x < srcI.cols && center.y >= 0 && center.y < srcI.rows) {
        floodFill(canvasGray, center, 255);
    }
    else {
        cerr << "Error: Calculated center is out of bounds!" << endl;
    }
    cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);
    //canvasGray.copyTo(win[4]);

    canvasColor = canvasGray & srcI;
    //canvasColor.copyTo(win[5]);
    //canvasColor.copyTo(win2[1]);

    windowName = "Yellow Segmentation (detail)";
    //imshow(windowName, detailResultWin);


   // showWindow(srcI, canvasColor,"Segmented");
    return canvasColor;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////                        BLUE SEGMENTATION                     /////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Mat blueSegmentation(const Mat& srcI) {
    string windowName;
    Mat blueMask, canvasColor, canvasGray;
    Mat blueChannel, greenChannel, redChannel;
    char str[256];
    Point2i center;
    vector<Scalar> colors;
    int const MAXfPt = 200;
    int t1, t2, t3, t4;
    RNG rng(0);
    vector<string> imageNames;

    // Get MAXfPt random but brighter colors for drawing later
    for (int i = 0; i < MAXfPt; i++) {
        for (;;) {
            t1 = rng.uniform(0, 255); // blue
            t2 = rng.uniform(0, 255); // green
            t3 = rng.uniform(0, 255); // red
            t4 = t1 + t2 + t3;
            if (t4 > 255) break;
        }
        colors.push_back(Scalar(t1, t2, t3));
    }


    // Open 2 large windows to display the results
    /*int const noOfImagePerCol = 2, noOfImagePerRow = 3;
    Mat detailResultWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    createWindowPartition(srcI, detailResultWin, win, legend, noOfImagePerCol, noOfImagePerRow);

    putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[1], "blueMask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[2], "Contours", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[3], "Longest contour", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[4], "Mask", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[5], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    int const noOfImagePerCol2 = 1, noOfImagePerRow2 = 2;
    Mat resultWin, win2[noOfImagePerRow2 * noOfImagePerCol2], legend2[noOfImagePerRow2 * noOfImagePerCol2];
    createWindowPartition(srcI, resultWin, win2, legend2, noOfImagePerCol2, noOfImagePerRow2);

    putText(legend2[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend2[1], "Sign segmented", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);

    srcI.copyTo(win[0]);
    srcI.copyTo(win2[0]);*/


    // Convert to HSV and use a range of blue colors
    Mat hsvImage;
    cvtColor(srcI, hsvImage, COLOR_BGR2HSV);

    // Define blue color range in HSV
    Scalar lowerBlue(96, 70, 70); // You might need to adjust these values
    Scalar upperBlue(130, 255, 255);

    // Create a mask for the blue color range
    inRange(hsvImage, lowerBlue, upperBlue, blueMask);

    //reduce noise
    //medianBlur(blueMask, blueMask, 5);

    //Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    //morphologyEx(blueMask, blueMask, MORPH_CLOSE, kernel); // Close small holes in the blue region
    //morphologyEx(blueMask, blueMask, MORPH_OPEN, kernel);  // Remove small noise


    //// Show result of blue color
    //cvtColor(blueMask, win[1], COLOR_GRAY2BGR);

    // Create canvases for drawing
    canvasColor.create(srcI.rows, srcI.cols, CV_8UC3);
    canvasGray.create(srcI.rows, srcI.cols, CV_8U);
    canvasColor = Scalar(0, 0, 0);

    // Get contours of the blue regions
    vector<vector<Point>> contours;
    findContours(blueMask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    int index = -1, max = 0;

    for (int j = 0; j < contours.size(); j++) {
        canvasGray = 0;
        if (max < contours[j].size()) {
            max = contours[j].size();
            index = j;
        }
        drawContours(canvasColor, contours, j, colors[j % MAXfPt]);
        drawContours(canvasGray, contours, j, 255);

        Moments M = moments(canvasGray);
        center.x = M.m10 / M.m00;
        center.y = M.m01 / M.m00;

        if (center.x >= 0 && center.x < srcI.cols && center.y >= 0 && center.y < srcI.rows) {
            floodFill(canvasGray, center, 255);
        }
        else {
            cerr << "Error: Calculated center is out of bounds!" << endl;
        }
        if (countNonZero(canvasGray) > 20) {
            sprintf_s(str, "Mask %d (area > 20)", j);
            //imshow(str, canvasGray);
        }
    }
    /*    canvasColor.copyTo(win[2]);*/
    if (index < 0) {
        waitKey(0);
    }

    canvasGray = 0;
    drawContours(canvasGray, contours, index, 255);
    //cvtColor(canvasGray, win[3], COLOR_GRAY2BGR);

    Moments M = moments(canvasGray);
    center.x = M.m10 / M.m00;
    center.y = M.m01 / M.m00;

    if (center.x >= 0 && center.x < srcI.cols && center.y >= 0 && center.y < srcI.rows) {
        floodFill(canvasGray, center, 255);
    }
    else {
        cerr << "Error: Calculated center is out of bounds!" << endl;
    }
    cvtColor(canvasGray, canvasGray, COLOR_GRAY2BGR);
    /*       canvasGray.copyTo(win[4]);*/

    canvasColor = canvasGray & srcI;
    //canvasColor.copyTo(win[5]);
    //canvasColor.copyTo(win2[1]);

    windowName = "Segmentation (detail)";
    //imshow(windowName, detailResultWin);
    //imshow("Traffic sign segmentation", resultWin);


   // showWindow(srcI, canvasColor, "Segmented");
    return canvasColor;
}

void showWindow(Mat src, Mat seg,String class_name) {
    int const	noOfImagePerCol = 1, noOfImagePerRow = 2;
    Mat			largeWin, win[noOfImagePerRow * noOfImagePerCol], legend[noOfImagePerRow * noOfImagePerCol];
    createWindowPartition(src, largeWin, win, legend, noOfImagePerCol, noOfImagePerRow);
    src.copyTo(win[0]);
    putText(legend[0], "Original", Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    putText(legend[1], class_name, Point(5, 11), 1, 1, Scalar(250, 250, 250), 1);
    seg.copyTo(win[1]);
    imshow("Window", largeWin);
    waitKey(0);
    destroyAllWindows();
}
// Function to extract HOG features from an image
void extractHOGFeatures(const Mat& image, vector<float>& features) {
    Mat resizedImage;
    Size resizeSize(64, 64); // or whatever size you decide
    resize(image, resizedImage, resizeSize);

    Mat grayImage;
    cvtColor(resizedImage, grayImage, COLOR_BGR2GRAY);

    HOGDescriptor hog(
        Size(64, 64), // Window size should match the resized image size
        Size(16, 16),   // Block size
        Size(8, 8),     // Block stride
        Size(8, 8),     // Cell size
        9               // Number of bins
    );

    hog.compute(grayImage, features);
}

// Function to extract color histogram features from an image
void extractColorHistogram(const Mat& image, vector<float>& features) {
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    int histSize[] = { 5, 5, 5 }; // Number of bins for each channel (Hue, Saturation, Value)
    float hRange[] = { 0, 256 }; // Range of values for each channel
    const float* hRangeList[] = { hRange, hRange, hRange }; // Range for each channel

    Mat hist;
    int channels[] = { 0, 1, 2 }; // Channels for Hue, Saturation, Value
    calcHist(&hsvImage, 1, channels, Mat(), hist, 3, histSize, hRangeList, true, false);

    hist.convertTo(hist, CV_32F);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    features.clear();
    int binCount = histSize[0] * histSize[1] * histSize[2];
    features.resize(binCount);

    for (int i = 0; i < histSize[0]; ++i) {
        for (int j = 0; j < histSize[1]; ++j) {
            for (int k = 0; k < histSize[2]; ++k) {
                int idx = i + histSize[0] * (j + k * histSize[1]);
                features[idx] = hist.at<float>(i, j, k);
            }
        }
    }
}

// Function to load features and labels from CSV
void loadCSV(const string& filename, Mat& features, Mat& labels) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }

    string line;
    getline(file, line); // Skip header line

    vector<float> rowFeatures;
    vector<int> rowLabels;

    while (getline(file, line)) {
        stringstream ss(line);
        string item;

        // Read filename (skip it)
        getline(ss, item, ',');

        // Read label
        int label;
        if (!(getline(ss, item, ',') && istringstream(item) >> label)) {
            cerr << "Error: Could not parse label from line: " << line << endl;
            continue; // Skip this line if label parsing fails
        }
        rowLabels.push_back(label);

        // Read features
        rowFeatures.clear();
        while (getline(ss, item, ',')) {
            try {
                float feature = stof(item);
                rowFeatures.push_back(feature);
            }
            catch (const invalid_argument&) {
                cerr << "Error: Could not parse feature from line: " << line << endl;
                rowFeatures.clear();
                break; // Skip this line if feature parsing fails
            }
        }

        // Convert rowFeatures to a Mat if it is not empty
        if (!rowFeatures.empty()) {
            Mat row(rowFeatures);
            features.push_back(row.t());
        }
    }

    // Convert labels vector to Mat
    labels = Mat(rowLabels, true).reshape(1, rowLabels.size());
    labels.convertTo(labels, CV_32S);
    features.convertTo(features, CV_32F);

    // Debug: Print the size of the features and labels
    cout << "Loaded features: " << features.rows << "x" << features.cols << endl;
    cout << "Loaded labels: " << labels.rows << "x" << labels.cols << endl;
}

// Function to shuffle and split data
void shuffleAndSplit(const Mat& features, const Mat& labels, Mat& trainFeatures, Mat& trainLabels, Mat& testFeatures, Mat& testLabels, float trainRatio) {
    vector<int> indices(features.rows);
    iota(indices.begin(), indices.end(), 0);

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    int trainSize = static_cast<int>(trainRatio * features.rows);

    trainFeatures.create(trainSize, features.cols, features.type());
    trainLabels.create(trainSize, labels.cols, labels.type());
    testFeatures.create(features.rows - trainSize, features.cols, features.type());
    testLabels.create(features.rows - trainSize, labels.cols, labels.type());

    for (int i = 0; i < features.rows; ++i) {
        if (i < trainSize) {
            features.row(indices[i]).copyTo(trainFeatures.row(i));
            labels.row(indices[i]).copyTo(trainLabels.row(i));
        }
        else {
            features.row(indices[i]).copyTo(testFeatures.row(i - trainSize));
            labels.row(indices[i]).copyTo(testLabels.row(i - trainSize));
        }
    }
}

// Function to find the class meaning
string getClassMeaning(int classNumber) {
    map<int, string> classMeanings = {
        {0, "Speed limit (5km/h)"},
        {1, "Speed limit (15km/h)"},
        {2, "Speed limit (30km/h)"},
        {3, "Speed limit (40km/h)"},
        {4, "Speed limit (50km/h)"},
        {5, "Speed limit (60km/h)"},
        {6, "Speed limit (70km/h)"},
        {7, "Speed limit (80km/h)"},
        {8, "No go straight or turn left"},
        {9, "No go straight or turn right"},
        {10, "No go straight"},
        {11, "No turn left"},
        {12, "No turn left and right"},
        {13, "No turn right"},
        {14, "No overtaking"},
        {15, "No U-turn"},
        {16, "No vehicle allowed"},
        {17, "No horn allowed"},
        {18, "Speed limit (40km/h)"},
        {19, "Speed limit (50km/h)"},
        {20, "Go straight or turn right"},
        {21, "Go straight"},
        {22, "Turn left ahead"},
        {23, "Turn left and right ahead"},
        {24, "Turn right ahead"},
        {25, "Keep left"},
        {26, "Keep right"},
        {27, "Roundabout"},
        {28, "Only car allowed"},
        {29, "Sound horn sign"},
        {30, "Bicycle lane"},
        {31, "U-turn sign"},
        {32, "Road divides"},
        {33, "Traffic light sign"},
        {34, "Warning sign"},
        {35, "Pedestrian crossing symbol"},
        {36, "Bicycle traffic warning"},
        {37, "School crossing sign"},
        {38, "Sharp bend"},
        {39, "Sharp bend"},
        {40, "Danger steep hill ahead warning"},
        {41, "Danger steep hill ahead warning"},
        {42, "Slowing sign"},
        {43, "T-junction ahead"},
        {44, "T-junction ahead"},
        {45, "Village warning sign"},
        {46, "Snake road"},
        {47, "Railroad level crossing sign"},
        {48, "Under construction"},
        {49, "Snake road"},
        {50, "Railroad level crossing sign"},
        {51, "Accident frequent happened sign"},
        {52, "Stop"},
        {53, "No entry"},
        {54, "No Stopping"},
        {55, "No entry"},
        {56, "Give way"},
        {57, "Stop for checking purpose"}
    };

    if (classMeanings.find(classNumber) != classMeanings.end()) {
        return classMeanings[classNumber];
    }
    else {
        return "Unknown traffic sign";
    }
}

void HOG() {
    // Folder path for images
    string folderPath = "Data/"; // Replace with your folder path
    vector<string> imageNames;
    glob(folderPath + "*.png", imageNames, true);

    if (imageNames.empty()) {
        cerr << "Error: No images found in the directory: " << folderPath << endl;
    }

    // Open CSV file for writing
    ofstream csvFile("HOGFeatures.csv");
    if (!csvFile.is_open()) {
        cerr << "Error: Could not open CSV file for writing." << endl;
    }

    // Extract features for the first image to determine the number of features
    Mat firstImage = imread(imageNames[0]);
    if (firstImage.empty()) {
        cerr << "Error: Could not open or find the first image " << imageNames[0] << endl;
    }

    vector<float> sampleFeatures;
    extractHOGFeatures(firstImage, sampleFeatures);

    // Write CSV header dynamically based on the number of features
    csvFile << "filename,label";
    for (int i = 0; i < sampleFeatures.size(); ++i) {
        csvFile << ",feature_" << i;
    }
    csvFile << endl;

    // Process all images in the directory
    for (const string& imagePath : imageNames) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Could not open or find the image " << imagePath << endl;
            continue;
        }

        vector<float> features;
        extractHOGFeatures(image, features);
        cout << "Extracting features......" << endl;
        // Extract filename from path
        string filename = imagePath.substr(imagePath.find_last_of("/\\") + 1);

        // Extract label from the first 3 digits of the filename
        string labelStr = filename.substr(0, 3); // Extract first 3 characters
        int label = stoi(labelStr); // Convert to integer

        // Write features to CSV
        csvFile << filename << "," << label;
        for (float feature : features) {
            csvFile << "," << feature;
        }
        csvFile << endl;
    }

    csvFile.close();
    cout << "Feature extraction complete and saved to HOGFeatures.csv" << endl;
}

void trainSVM() {
    //Load all data
    Mat allFeatures, allLabels;
    loadCSV("ColorHistogramFeatures.csv", allFeatures, allLabels);

    //Shuffle and split data
    Mat trainFeatures, trainLabels, testFeatures, testLabels;
    shuffleAndSplit(allFeatures, allLabels, trainFeatures, trainLabels, testFeatures, testLabels, 0.80f);

    //Define SVM parameters
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(SVM::LINEAR); // or SVM::RBF for non-linear
    svm->setC(1); // Regularization parameter
    svm->setGamma(0.5); // Used if kernel is RBF
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

    //Train SVM
    svm->train(trainFeatures, ROW_SAMPLE, trainLabels);
    svm->save("svm_modelColor.yml");

    cout << "SVM model trained and saved to svm_modelColor.yml" << endl;

    //Predict and evaluate
    Mat predictions;
    Ptr<SVM> loadedSVM = SVM::load("svm_modelColor.yml");
    if (loadedSVM.empty()) {
        cerr << "Error: Could not load the SVM model." << endl;
    }

    loadedSVM->predict(testFeatures, predictions);

    //Calculate accuracy
    int correct = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        cout << "Test Label: " << testLabels.at<int>(i, 0) << "   Prediction Label: " << static_cast<int>(predictions.at<float>(i, 0))<<endl;
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictions.at<float>(i, 0))) {
            ++correct;
        }
    }

    float accuracy = static_cast<float>(correct) / testLabels.rows;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;
}

void trainRandomForest() {
    // Load all data
    Mat allFeatures, allLabels;
    loadCSV("HOGFeatures.csv", allFeatures, allLabels);

    // Shuffle and split data
    Mat trainFeatures, trainLabels, testFeatures, testLabels;
    shuffleAndSplit(allFeatures, allLabels, trainFeatures, trainLabels, testFeatures, testLabels, 0.99f);

    // Define Random Forest parameters
    Ptr<RTrees> randomForest = RTrees::create();
    randomForest->setMaxDepth(10); // Maximum depth of the tree
    randomForest->setMinSampleCount(2); // Minimum number of samples required at a leaf node
    randomForest->setRegressionAccuracy(0); // Regression accuracy: N/A here
    randomForest->setUseSurrogates(false); // Use surrogate splits
    randomForest->setMaxCategories(10); // Maximum number of categories (useful for categorical data)
    randomForest->setPriors(Mat()); // The prior probabilities of the classes
    randomForest->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6)); // Termination criteria

    // Train Random Forest
    randomForest->train(trainFeatures, ROW_SAMPLE, trainLabels);
    randomForest->save("randomForest_modelHOG.yml");

    cout << "Random Forest model trained and saved to randomForest_modelHOG.yml" << endl;

    // Predict and evaluate
    Mat predictions;
    Ptr<RTrees> loadedRandomForest = RTrees::load("randomForest_modelHOG.yml");
    if (loadedRandomForest.empty()) {
        cerr << "Error: Could not load the Random Forest model." << endl;
    }

    loadedRandomForest->predict(testFeatures, predictions);

    // Calculate accuracy
    int correct = 0;
    for (int i = 0; i < testLabels.rows; ++i) {
        cout << "Test Label: " << testLabels.at<int>(i, 0) << "   Prediction Label: " << static_cast<int>(predictions.at<float>(i, 0))<<endl;
        if (testLabels.at<int>(i, 0) == static_cast<int>(predictions.at<float>(i, 0))) {
            ++correct;
        }
    }

    float accuracy = static_cast<float>(correct) / testLabels.rows;
    cout << "Accuracy: " << accuracy * 100 << "%" << endl;
}

void histogramExtraction() {
    // Folder path for images
    string folderPath = "Data/"; // Replace with your folder path
    vector<string> imageNames;
    glob(folderPath + "*.png", imageNames, true);

    // Open CSV file for writing
    ofstream csvFile("ColorHistogramFeatures.csv");
    if (!csvFile.is_open()) {
        cerr << "Error: Could not open CSV file for writing." << endl;
    }

    // Write CSV header
    csvFile << "filename,label,feature_0,feature_1,...,feature_N" << endl;

    for (const string& imagePath : imageNames) {
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Could not open or find the image " << imagePath << endl;
            continue;
        }

        vector<float> features;
        extractColorHistogram(image, features);
        cout << "Extracting features......" << endl;

        // Extract filename from path
        string filename = imagePath.substr(imagePath.find_last_of("/\\") + 1);

        // Extract label from the first 3 digits of the filename
        string labelStr = filename.substr(0, 3); // Extract first 3 characters
        int label = stoi(labelStr); // Convert to integer

        // Write features to CSV
        csvFile << filename << "," << label;
        for (float feature : features) {
            csvFile << "," << feature;
        }
        csvFile << endl;
    }

    csvFile.close();
    cout << "Feature extraction complete and saved to ColorHistogramFeatures.csv" << endl;
}

void predictAndShowMeaning(const string& modelPath, const string& modelType) {
    string folderPath = "Inputs/Traffic signs/"; // Folder containing the images
    Ptr<ml::StatModel> model;

    // Load the appropriate model based on modelType
    if (modelType == "SVM") {
        model = StatModel::load<SVM>(modelPath);
    }
    else if (modelType == "RTrees") {
        model = StatModel::load<RTrees>(modelPath);
    }
    else {
        cerr << "Error: Unsupported model type." << endl;
        return;
    }

    // Check if model is loaded correctly
    if (model.empty()) {
        cerr << "Error: Failed to load the model from: " << modelPath << endl;
        return;
    }

    // Get list of images
    vector<cv::String> imageFiles;
    glob(folderPath + "*.png", imageFiles);

    for (const auto& imageFile : imageFiles) {
        Mat image = imread(imageFile);
        if (image.empty()) {
            cerr << "Error: Failed to load image: " << imageFile << endl;
            continue;
        }

        resize(image, image, Size(200, 200));
        // Extract features based on the chosen model type
        vector<float> features;
        if (modelType == "SVM") {
            extractColorHistogram(image, features);
        }
        else if (modelType == "RTrees") {
            extractHOGFeatures(image, features);
        }

        // Convert features to Mat
        Mat featureMat(1, (int)features.size(), CV_32F, features.data());

        // Predict using the loaded model
        Mat predictions;
        model->predict(featureMat, predictions);

        int classNumber = static_cast<int>(predictions.at<float>(0));

        // Show the image and print the predicted class meaning
        string classMeaning = getClassMeaning(classNumber);
        cout << "Image: " << imageFile << " -> Class: " << classNumber << " (" << classMeaning << ")" << endl;
        Mat segmented = selectSegmentation(image);
        showWindow(image, segmented, classMeaning);
        //imshow("Image", image);
        waitKey(0); // Wait for keypress before closing image
    }
}



int main() {
    int choice = -1;  // Initialize choice to an invalid value

    while (choice != 0) {
        cout << "Please choose what methods to do Feature extraction and classifier." << endl;
        cout << "1. Segmentation on 84 Original Data" << endl;
        cout << "2. Histogram Extraction and Train SVM" << endl;
        cout << "3. HOG Extraction and Train Random Forest" << endl;
        cout << "0. Exit" << endl;
        cout << "Enter your choice: ";

        // Check if input is an integer
        if (!(cin >> choice)) {
            cout << "Invalid input. Please input integer only." << endl;
            cin.clear();  // Clear the error state
            cin.ignore(numeric_limits<streamsize>::max(), '\n');  // Ignore the rest of the invalid input
            continue;  // Skip to the next iteration of the loop
        }

        if (choice == 1) {
            PicSegmentation();
        }
        else if (choice == 2) {
            // Histogram extraction and SVM training
            // You need to implement these functions
            histogramExtraction();
            trainSVM();
            predictAndShowMeaning("svm_modelColor.yml", "SVM");
        }
        else if (choice == 3) {
            // HOG extraction and Random Forest training
            // You need to implement these functions
            HOG();
            trainRandomForest();
            predictAndShowMeaning("randomForest_modelHOG.yml", "RTrees");
        }
        else if (choice == 0) {
            cout << "Exiting the program..." << endl;
        }
        else {
            cout << "Invalid input. Please choose a valid option." << endl;
        }
    }

    return 0;
}
