/* This program is the final year project belong to GuoZiheng
 * ID number: 1101732
 *
 *
 *
 *
 * This file is part of MultiTLD
 *
 */

/*
 This is a live demo load a local video.
 It makes use of OpenCV to capture the frames and highgui to display the output.
 
 There are some keys to customize which components are displayed:
 D - dis/enable drawing of detections (green boxes)
 P - dis/enable drawing of learned patches
 T - dis/enable drawing of tracked points
 L - dis/enable learning
 S - save current classifier to CLASSIFIERFILENAME
 O - load classifier from CLASSIFIERFILENAME (not implemented)
 R - reset classifier (not implemented)
 ESC - exit
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
//#include <SDKDDKVer.h>
// #include "cv.h"
// #include "highgui.h"
#include <opencv2/opencv.hpp>
#include "MultiObjectTLD.h"

#define LOADCLASSIFIERATSTART 0
#define CLASSIFIERFILENAME "test.moctld"

//resize the high resolution camera and speed up tracking
#define FORCE_RESIZING
#define RESOLUTION_X 640
#define RESOLUTION_Y 360

#define MOUSE_MODE_MARKER 0
#define MOUSE_MODE_ADD_BOX 1
#define MOUSE_MODE_IDLE 2
IplImage* curImage = NULL;
IplImage* capframe = NULL;
bool ivQuit = false;
int ivWidth, ivHeight;
CvCapture* capture;
ObjectBox mouseBox = {0,0,0,0,0};
int mouseMode = MOUSE_MODE_IDLE;
int drawMode = 255;
int x[2] = { 0, 0 };
int y[2] = { 0, 0 };
int width[2] = { 0, 0 };
int height[2] = { 0, 0 };
int indexOfFrame = 0;
int pointX[2], pointY[2], pointWidth[2], pointHeight[2];
bool oneDone=false, twoDone=true;
bool learningEnabled = true, save = false, load = false, reset = false;

void Init(int argc, char *argv[]);
void* Run(void*);
void HandleInput(int interval = 1);
void MouseHandler(int event, int x, int y, int flags, void* param);
void FromRGB(Matrix& maRed, Matrix& maGreen, Matrix& maBlue);

using namespace std;
using namespace cv;


//行人检测
void peopleDetection()
{
    Mat img;
    FILE* f = 0;
    char _filename[1024] = "1.jpg";
    cvSaveImage("1.jpg", curImage);
    img = imread("1.jpg");
    
    
    // resize(img,img,Size(640,360)) ;
    
    imshow("1", img);
    waitKey(1);
    
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    namedWindow("people detector", 1);
    
    
    for (;;)
    {
        char* filename = _filename;
        if (f)
        {
            if (!fgets(filename, (int)sizeof(_filename)-2, f))
                break;
            //while(*filename && isspace(*filename))
            //  ++filename;
            if (filename[0] == '#')
                continue;
            int l = (int)strlen(filename);
            while (l > 0 && isspace(filename[l - 1]))
                --l;
            filename[l] = '\0';
            img = imread(filename);
        }
        printf("%s:\n", filename);
        if (!img.data)
            continue;
        
        fflush(stdout);
        vector<Rect> found, found_filtered;
        double t = (double)getTickCount();
        // run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
        waitKey(10);
        t = (double)getTickCount() - t;
        printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
        size_t i, j;
        /*if (oneDone)
         {
         mouseBox.x = pointX[1];
         mouseBox.y = pointY[1];
         mouseBox.width = pointWidth[1];
         cout << "oneDong : " << pointWidth[1] << endl;
         mouseBox.height = pointHeight[1];
         oneDone = false;
         mouseMode = MOUSE_MODE_ADD_BOX;
         }
         if (!twoDone)
         {
         mouseBox.x = pointX[0];
         mouseBox.y = pointY[0];
         mouseBox.width = pointWidth[0];
         cout << "twoDong : " << pointWidth[0] << endl;
         mouseBox.height = pointHeight[0];
         oneDone = true;
         twoDone = true;
         mouseMode = MOUSE_MODE_ADD_BOX;
         }*/
        for (i = 0; i < found.size(); i++)
        {
            Rect r = found[i];
            for (j = 0; j < found.size(); j++)
                if (j != i && (r & found[j]) == r)
                    break;
            if (j == found.size())
                found_filtered.push_back(r);
        }
        cout << found_filtered.size() << endl;
        for (i = 0; i < found_filtered.size(); i++)
        {
            
            Rect r = found_filtered[i];
            // the HOG detector returns slightly larger rectangles than the real objects.
            // so we slightly shrink the rectangles to get a nicer output.
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);
            pointX[i] = r.x;
            pointY[i] = r.y;
            pointWidth[i] = r.width;
            pointHeight[i] = r.height;
            cout << r.width << endl;
            cout << r.height << endl;
            rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
            cout << r.tl() << endl;
            cout << r.br() << endl;
            twoDone = false;
            cout << "循环" << endl;
            
        }
        imshow("people detector", img);
        // 		int c = waitKey(0);
        int c = waitKey(0) & 255;
        if (c == 'q' || c == 'Q' || !f)
            break;
    }
    if (f)
        fclose(f);
    //	return 0;
}

//void faceDetection()
//{
//
//
//	static CvMemStorage* storage = 0;
//	static CvHaarClassifierCascade* cascade = 0;
//
//	cascade = (CvHaarClassifierCascade*)cvLoad("haarcascade_frontalface_alt.xml", 0, 0, 0);
//
//	storage = cvCreateMemStorage(0);
//
////	cvNamedWindow("face", 1);
//
//		IplImage* img = NULL;
//		CvSeq* faces;
//
//		mouseMode = MOUSE_MODE_MARKER;
//
//		img = cvCloneImage(curImage);
//		img->origin = 0;
//
//		if (curImage->origin)
//			cvFlip(img, img);
//
//		cvClearMemStorage(storage);
//		//目标检测
//		faces = cvHaarDetectObjects(img, cascade, storage, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(20, 20));
//		std::cout << "detection the total faces number : " << faces->total << std::endl;
//		for (int i = 0; i < (faces ? faces->total : 0); i++)
//		{
//			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);
//			cvRectangle(img, cvPoint(r->x, r->y),
//				cvPoint(r->x + r->width, r->y + r->height), CV_RGB(255, 0, 0), 1);
//			x[i] = r->x;
//			mouseBox.x = x[0];
//			y[i] = r->y;
//			mouseBox.y = y[0];
//			width[i] = r->width;
//			mouseBox.width = width[0];
//			std::cout << mouseBox.width << std::endl;
//			height[i] = r->height;
//			mouseBox.height = height[0];
//			std::cout << mouseBox.height << std::endl;
//			mouseMode = MOUSE_MODE_ADD_BOX;
//
//		}
//
//		cvShowImage("Video", img);
//		//cvResizeWindow("Video", ivWidth, ivHeight);
//}

void saveVariable(int i)
{
    mouseMode = MOUSE_MODE_MARKER;
    mouseBox.x = pointX[i];
    mouseBox.y = pointY[i];
    mouseBox.width = pointWidth[i];
    cout << "twoDong : " << pointWidth[0] << endl;
    mouseBox.height = pointHeight[i];
    //oneDone = true;
    //twoDone = true;
    mouseMode = MOUSE_MODE_ADD_BOX;
}

int main(int argc, char *argv[])
{
    Init(argc, argv);
    Run(0);
    cvDestroyAllWindows();
    return 0;
}


void Init(int argc, char *argv[])
{
    //    //get the video from the camero
    //    capture = cvCaptureFromCAM(CV_CAP_ANY);
    
    //load the local video
    capture = cvCreateFileCapture("//Users//joseph//Desktop//0.mov");
    
    if(!capture)
    {
        std::cout << "error starting video capture" << std::endl;
        exit(0);
    }
    //propose a resolution
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 360);
    //get the actual (supported) resolution
    ivWidth = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
    ivHeight = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
    std::cout << "camera/video resolution: " << ivWidth << "x" << ivHeight << std::endl;
#ifdef FORCE_RESIZING
    ivWidth = RESOLUTION_X;
    ivHeight = RESOLUTION_Y;
#endif
    
    cvNamedWindow("MOCTLD", 0); //CV_WINDOW_AUTOSIZE );
    
    CvSize wsize = {ivWidth, ivHeight};
    curImage = cvCreateImage(wsize, IPL_DEPTH_8U, 3);
    
    cvResizeWindow("MOCTLD", ivWidth, ivHeight);
    
    //    cvSetMouseCallback("MOCTLD", MouseHandler);
}

void* Run(void*)
{
    int size = ivWidth*ivHeight;
    
    std::cout<<ivWidth<<"\n"<<ivHeight<<"\n" ;
    
    // Initialize MultiObjectTLD
#if LOADCLASSIFIERATSTART
    MultiObjectTLD p = MultiObjectTLD::loadClassifier((char*)CLASSIFIERFILENAME);
#else
    MOTLDSettings settings(COLOR_MODE_RGB);
    settings.useColor = true;
    MultiObjectTLD p(ivWidth, ivHeight, settings);
#endif
    
    Matrix maRed;
    Matrix maGreen;
    Matrix maBlue;
    //unsigned char img[size*3];
    unsigned char img[640*360*3];
#ifdef FORCE_RESIZING
    CvSize wsize = {ivWidth, ivHeight};
    IplImage* frame = cvCreateImage(wsize, IPL_DEPTH_8U, 3);
#endif
    int n = 0;
    while (!ivQuit)
    {
        
        //if(reset){
        //  p = *(new MultiObjectTLD(ivWidth, ivHeight, COLOR_MODE_RGB));
        //  reset = false;
        //}
        //if(load){
        //  p = MultiObjectTLD::loadClassifier(CLASSIFIERFILENAME);
        //  load = false;
        //}
        
        
        // Grab an image
        if(!cvGrabFrame(capture))
        {
            std::cout << "error grabbing frame" << std::endl;
            break;
        }
#ifdef FORCE_RESIZING
        capframe = cvRetrieveFrame(capture);
        cvResize(capframe, frame);
        //n++;
        //cvShowImage("Video", curImage);
        //if(n=1)
        //{
        //	faceDetection();
        //}
#else
        IplImage* frame = cvRetrieveFrame(capture);
#endif
        for(int j = 0; j<size; j++)
        {
            img[j] = frame->imageData[j*3+2];
            img[j+size] = frame->imageData[j*3+1];
            img[j+2*size] = frame->imageData[j*3];
        }
        
        // Process it with motld
        p.processFrame(img);
        
        // Add new box
        if(mouseMode == MOUSE_MODE_ADD_BOX)
        {
            p.addObject(mouseBox);
            mouseMode = MOUSE_MODE_IDLE;
        }
        
        // Display result
        HandleInput();
        p.getDebugImage(img, maRed, maGreen, maBlue, drawMode);
        
        FromRGB(maRed, maGreen, maBlue);
        cvShowImage("MOCTLD", curImage);
        n++;
        cout << n << endl;
        if (n == 10)
        {
            peopleDetection();
        }
        
        p.enableLearning(learningEnabled);
        if(save)
        {
            p.saveClassifier((char*)CLASSIFIERFILENAME);
            save = false;
        }
        
        if (oneDone)
        {
            saveVariable(1);
            oneDone = false;
        }
        if (!twoDone)
        {
            saveVariable(0);
            oneDone = true;
            twoDone = true;
        }
    }
    //delete[] img;
    cvReleaseCapture(&capture);
    return 0;
}


void HandleInput(int interval)
{
    int key = cvWaitKey(interval);
    if(key >= 0)
    {
        switch (key)
        {
                //这个drawMode是什么？有没有传去给头文件？为什么初始值是225？
            case 'd': drawMode ^= DEBUG_DRAW_DETECTIONS;  break;
            case 't': drawMode ^= DEBUG_DRAW_CROSSES;  break;
            case 'p': drawMode ^= DEBUG_DRAW_PATCHES;  break;
            case 'l':
                learningEnabled = !learningEnabled;
                std::cout << "learning " << (learningEnabled? "en" : "dis") << "abled" << std::endl;
                break;
            case 'r': reset = true; break;
            case 's': save = true;  break;
            case 'o': load = true;  break;
            case 27:  ivQuit = true; break; //ESC
            default:
                //std::cout << "unhandled key-code: " << key << std::endl;
                break;
        }
    }
}

//void MouseHandler(int event, int x, int y, int flags, void* param)
//{
//    switch(event)
//    {
//        case CV_EVENT_LBUTTONDOWN:
//            mouseBox.x = x;
//            mouseBox.y = y;
//            mouseBox.width = mouseBox.height = 0;
//            mouseMode = MOUSE_MODE_MARKER;
//            cv::waitKey(0);
//            break;
//        case CV_EVENT_MOUSEMOVE:
//            if(mouseMode == MOUSE_MODE_MARKER)
//            {
//                mouseBox.width = x - mouseBox.x;
//                mouseBox.height = y - mouseBox.y;
//            }
//            break;
//        case CV_EVENT_LBUTTONUP:
//            if(mouseMode != MOUSE_MODE_MARKER)
//                break;
//            if(mouseBox.width < 0)
//            {
//                mouseBox.x += mouseBox.width;
//                mouseBox.width *= -1;
//            }
//            if(mouseBox.height < 0)
//            {
//                mouseBox.y += mouseBox.height;
//                mouseBox.height *= -1;
//            }
//            if(mouseBox.width < 4 || mouseBox.height < 4)
//            {
//                std::cout << "bounding box too small!" << std::endl;
//                mouseMode = MOUSE_MODE_IDLE;
//            }else
//                mouseMode = MOUSE_MODE_ADD_BOX;
//            break;
//        case CV_EVENT_RBUTTONDOWN:
//            mouseMode = MOUSE_MODE_IDLE;
//            break;
//    }
//}


//这个方法是在扫描整个图像的RGB值
void FromRGB(Matrix& maRed, Matrix& maGreen, Matrix& maBlue)
{
    for(int i = 0; i < ivWidth*ivHeight; ++i)
    {
        curImage->imageData[3*i+2] = maRed.data()[i];
        curImage->imageData[3*i+1] = maGreen.data()[i];
        curImage->imageData[3*i+0] = maBlue.data()[i];
    }
    //CvPoint pt1; pt1.x = mouseBox.x; pt1.y = mouseBox.y;
    //CvPoint pt2; pt2.x = mouseBox.x + mouseBox.width; pt2.y = mouseBox.y + mouseBox.height;
    //cvRectangle(curImage, pt1, pt2, CV_RGB(0, 0, 255));
    indexOfFrame++;
    //at this place you could save the images using
    string rootPath = "C:\\Users\\st\\Desktop\\1\\";
    stringstream indexOfFrameStringstream;
    string indexOfFrameString;
    indexOfFrameStringstream << indexOfFrame;
    indexOfFrameStringstream >> indexOfFrameString;
    string picturePathName = rootPath + indexOfFrameString + ".jpg";
    const char* picturePathNameChar = picturePathName.c_str();
    cvSaveImage(picturePathNameChar, curImage);
    
    if (mouseMode == MOUSE_MODE_MARKER)
    {
        
        system("pause");
        std::cout << mouseBox.width << std::endl;
        std::cout << mouseBox.height << std::endl;
    }
}
