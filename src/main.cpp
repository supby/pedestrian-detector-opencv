#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"


#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplayPDHOG( ocl::oclMat oclFrame,  Mat frame, Mat img_aux );

/** Global variables */

ocl::HOGDescriptor ocl_hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9,
                           ocl::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, true,
                           ocl::HOGDescriptor::DEFAULT_NLEVELS);

/** @function main */
int main( int argc, const char** argv )
{
//    CvCapture* capture;
    VideoCapture capture;
    Mat frame, img_aux;
    ocl::oclMat oclFrame;

    const char* keys =
        "{ i |  input   |                | specify input image}"
        "{ c | camera   | -1             | enable camera capturing }"
        "{ v | video    |                | use video as input }";
    CommandLineParser cmd(argc, argv, keys);

    string vdo_source = cmd.get<string>("v");
    string img_source = cmd.get<string>("i");
    int camera_id = cmd.get<int>("c");

    ocl_hog.setSVMDetector(ocl::HOGDescriptor::getPeopleDetector64x128());

    if(vdo_source!="" || camera_id != -1)
    {
        if (vdo_source!="")
        {
            capture.open(vdo_source.c_str());
            if (!capture.isOpened())
                throw std::runtime_error(string("can't open video file: " + vdo_source));
        }
        else if (camera_id != -1)
        {
            capture.open(camera_id);
            if (!capture.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw runtime_error(msg.str());
            }
        }

        while( true )
        {
            capture >> frame;

            //-- 3. Apply the classifier to the frame
            if( !frame.empty() )
            {
                detectAndDisplayPDHOG(oclFrame, frame, img_aux);
            }
            else
            {
                printf(" --(!) No captured frame -- Break!");
                break;
            }

            int c = waitKey(10);
            if( (char)c == 'c' )
            {
                break;
            }
        }
    }
    else
    {
        frame = imread(img_source);
        if (frame.empty())
            throw runtime_error(string("can't open image file: " + img_source));

        detectAndDisplayPDHOG(oclFrame, frame, img_aux);

        while( true )
        {
            int c = waitKey(10);
            if( (char)c == 'c' )
            {
                break;
            }
        }
    }

    return 0;
}

void detectAndDisplayPDHOG( ocl::oclMat oclFrame, Mat frame, Mat img_aux )
{
    cvtColor(frame, img_aux, CV_BGR2BGRA);
    oclFrame.upload(img_aux);

    vector<Rect> found, found_filtered;
    ocl_hog.detectMultiScale(oclFrame, found, 0, Size(8,8), Size(0, 0), 1.05, 8);

    size_t i, j;
    for( i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];
        for( j = 0; j < found.size(); j++ )
            if( j != i && (r & found[j]) == r)
                break;
        if( j == found.size() )
            found_filtered.push_back(r);
    }
    for( i = 0; i < found_filtered.size(); i++ )
    {
        Rect r = found_filtered[i];
        // the HOG detector returns slightly larger rectangles than the real objects.
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
    }
    imshow("people detector", frame);
}
