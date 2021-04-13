// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include<random>
# define M_PI        3.141592653589793238462643383279502884L /* pi */

void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = MAX_PATH - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar* lpSrc = src.data;
		uchar* lpDst = dst.data;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i * w + j];
				lpDst[i * w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i * width + j;

				dstDataPtrH[gi] = hsvDataPtr[hi] * 510 / 360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}

void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



// laborator 01

void changeGrayAdditive()
{
	char fname[MAX_PATH];
	int additive_factor;
	printf("additive factor = ");
	scanf("%d", &additive_factor);

	while (openFileDlg(fname))
	{
		Mat_<uchar> src;
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat_<uchar> dst = Mat_<uchar>(src.rows, src.cols);

		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				dst(i, j) = src(i, j) + additive_factor;
			}

		imshow("src image", src);
		imshow("dst image", dst);
		waitKey();
	}
}

void changeGrayMultiplicative()
{
	char fname[MAX_PATH];
	int multiplicative_factor = 2;
	printf("multiplicative factor = ");
	scanf("%d", &multiplicative_factor);

	while (openFileDlg(fname))
	{
		Mat_<uchar> src;
		src = imread(fname, IMREAD_GRAYSCALE);

		Mat_<uchar> dst = Mat_<uchar>(src.rows, src.cols);

		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++) {
				dst(i, j) = src(i, j) * multiplicative_factor;
			}

		imshow("src image", src);
		imshow("dst image", dst);
		imwrite("D:\\UTC_an3_sem2\\PI\\Laborator\\lab01\\imaginiTest\\c1.bmp", dst);
		waitKey();
	}
}

void fourSquares()
{
	Mat_<Vec3b> img = Mat_<Vec3b>(256, 256);

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {

			if (i < 128 && j < 128) {
				img(i, j)[0] = 255;
				img(i, j)[1] = 255;
				img(i, j)[2] = 255;
			}
			else if (i < 128 && j >= 128) {
				img(i, j)[0] = 12;
				img(i, j)[1] = 12;
				img(i, j)[2] = 194;
			}
			else if (i >= 128 && j < 128) {
				img(i, j)[0] = 50;
				img(i, j)[1] = 153;
				img(i, j)[2] = 31;
			}
			else if (i >= 128 && j >= 128) {
				img(i, j)[0] = 50;
				img(i, j)[1] = 217;
				img(i, j)[2] = 206;
			}
		}

	imshow("imagine", img);
	waitKey();
}


// laborator 02

void splitRGB()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
		Mat_<uchar> dest_r = Mat_<uchar>(img.rows, img.cols);
		Mat_<uchar> dest_g = Mat_<uchar>(img.rows, img.cols);
		Mat_<uchar> dest_b = Mat_<uchar>(img.rows, img.cols);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				dest_b(i, j) = img(i, j)[0];
				dest_g(i, j) = img(i, j)[1];
				dest_r(i, j) = img(i, j)[2];

			}

		imshow("imagine originala", img);
		imshow("plan rosu", dest_r);
		imshow("plan verde", dest_g);
		imshow("plan albastru", dest_b);
		waitKey();
	}
}

void colorToGrayscale()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
		Mat_<uchar> dest = Mat_<uchar>(img.rows, img.cols);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				unsigned char average = (img(i, j)[0] + img(i, j)[1] + img(i, j)[2]) / 3;
				dest(i, j) = average;
			}

		imshow("color", img);
		imshow("grayscale", dest);
		waitKey();
	}
}

void grayscaleToBinary()
{
	char fname[MAX_PATH];
	int treshold;
	printf("treshold = ");
	scanf("%d", &treshold);

	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dest = Mat_<uchar>(img.rows, img.cols);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				if (img(i, j) < treshold)
					dest(i, j) = 0;
				else
					dest(i, j) = 255;
			}

		imshow("grayscale", img);
		imshow("binary", dest);
		waitKey();
	}
}

void RGBtoHSV()
{
	char fname[MAX_PATH];

	while (openFileDlg(fname)) {
		Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
		Mat_<uchar> hImg = Mat_<uchar>(img.rows, img.cols);
		Mat_<uchar> sImg = Mat_<uchar>(img.rows, img.cols);
		Mat_<uchar> vImg = Mat_<uchar>(img.rows, img.cols);

		for (int i = 0; i < img.rows; i++)
			for (int j = 0; j < img.cols; j++) {
				float r = (float)img(i, j)[2] / 255;
				float g = (float)img(i, j)[1] / 255;
				float b = (float)img(i, j)[0] / 255;

				float M = max(max(r, g), b);
				float m = min(min(r, g), b);
				float C = M - m;

				//value
				float V = M;
				float S, H;

				//saturation
				if (V != 0)
					S = C / V;
				else
					S = 0;

				//hue
				if (C != 0) {
					if (M == r) H = 60 * (g - b) / C;
					if (M == g) H = 120 + 60 * (b - r) / C;
					if (M == b) H = 240 + 60 * (r - g) / C;
				}
				else
					H = 0;
				if (H < 0)
					H = H + 360;

				float H_norm = H * 255 / 360;
				float	S_norm = S * 255;
				float	V_norm = V * 255;

				hImg(i, j) = H_norm;
				sImg(i, j) = S_norm;
				vImg(i, j) = V_norm;
			}

		imshow("H", hImg);
		imshow("S", sImg);
		imshow("V", vImg);
		waitKey();
	}
}

bool isInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
		return true;
	else
		return false;
}


// laborator 03

void showHistogram(const string& name, int* hist, const int hist_cols, const int hist_height) {
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image
	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;
	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins // colored in magenta
	}
	imshow(name, imgHist);
}

vector<int> computeHistogram(Mat_<uchar> img) {
	vector<int> histogram(256);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histogram.at(img(i, j)) = histogram.at(img(i, j)) + 1;
		}
	}
	return histogram;
}

void testHistogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		vector<int> hst = computeHistogram(img);
		showHistogram("Histograma", hst.data(), 256, 256);
		waitKey(0);
	}
}

vector<float> computeFDP(Mat_<uchar> img) {
	vector<float> fdp(256);
	vector<int> histogram = computeHistogram(img);
	int M = img.rows * img.cols;

	for (int i = 0; i < 256; i++) {
		fdp.at(i) = ((float)histogram.at(i) / M);
	}
	return fdp;
}

void testFDP() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		vector<float> fdp = computeFDP(img);

		for (int i = 0; i < 256; i++) {
			printf("%lf\n", fdp.at(i));
		}

		waitKey(0);
	}
}

Mat_<uchar> grayReduction(Mat_<uchar> img) {
	vector<float> fdp = computeFDP(img);
	Mat_<uchar> dst = Mat_<uchar>(img.rows, img.cols);

	int WH = 5;
	int f = 2 * WH + 1;
	float TH = 0.0003;

	std::vector<float> mx;
	mx.push_back(0);

	for (int k = 0 + WH; k <= 255 - WH; k++) {
		float sum = 0.0f;
		bool highest = true;

		for (int i = k - WH; i <= k + WH; i++) {
			if (fdp[k] < fdp[i]) {
				highest = false;
				break;
			}
			sum += fdp.at(i);
		}

		float v = sum / f;

		// memoram maximul local
		if (highest == true && (fdp.at(k) > (v + TH))) {
			mx.push_back(k);
		}

	}
	mx.push_back(255);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int gr = img(i, j);
			int min = 255, value = -1;
			for (int k = 0; k < mx.size(); k++)
			{
				int diff = abs(gr - mx.at(k));
				if (diff < min) {
					min = diff;
					value = mx.at(k);
				}

			}
			dst(i, j) = value;
		}
	}

	return dst;
}

void testGrayReduction() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = grayReduction(img);

		imshow("original", img);
		imshow("gray reduction", dst);
		waitKey(0);
	}
}

Mat_<uchar> floydSteinberg(Mat_<uchar> img) {
	Mat_<uchar> gr = grayReduction(img);
	Mat_<uchar> dst = gr;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int oldpixel = img(i, j);
			int newpixel = gr(i, j);
			int error = oldpixel - newpixel;
			if (isInside(img, i, j + 1))
				dst(i, j + 1) = img(i, j + 1) + 7 * error / 16;
			if (isInside(img, i + 1, j - 1))
				dst(i + 1, j - 1) = img(i + 1, j - 1) + 3 * error / 16;
			if (isInside(img, i + 1, j))
				dst(i + 1, j) = img(i + 1, j) + 5 * error / 16;
			if (isInside(img, i + 1, j + 1))
				dst(i + 1, j + 1) = img(i + 1, j + 1) + error / 16;
		}
	}
	return dst;
}

void testFloydSteinberg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> gr = grayReduction(img);
		Mat_<uchar> dst = floydSteinberg(img);

		imshow("original", img);
		imshow("gray reduction", gr);
		imshow("Floyd Steinberg", dst);

		vector<int> hst = computeHistogram(gr);
		showHistogram("Histograma", hst.data(), 256, 256);

		waitKey(0);
	}
}


// laborator 04

Mat_<uchar> grayScaleToI(Mat_<uchar> img) {
	Mat_<uchar> dst = Mat_<uchar>(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) != 255)
				dst(i, j) = 1;
			else
				dst(i, j) = 0;
		}
	return dst;
}

Mat_<uchar> ItoGrayScale(Mat_<uchar> I) {
	Mat_<uchar> dst = Mat_<uchar>(I.rows, I.cols);
	for (int i = 0; i < I.rows; i++)
		for (int j = 0; j < I.cols; j++) {
			if (I(i, j) == 1)
				dst(i, j) = 0;
			else
				dst(i, j) = 255;
		}
	return dst;
}

int area(Mat_<uchar> I) {
	int A = 0;
	for (int r = 0; r < I.rows; r++)
		for (int c = 0; c < I.cols; c++)
			if (I(r, c) == 1) {
				A++;
			}
	return A;
}

int rCenter(Mat_<uchar> I) {
	int rn = 0;
	for (int r = 0; r < I.rows; r++)
		for (int c = 0; c < I.cols; c++)
		{
			rn = rn + r * I(r, c);
		}

	rn = rn / area(I);
	return rn;
}

int cCenter(Mat_<uchar> I) {
	int cn = 0;
	for (int r = 0; r < I.rows; r++)
		for (int c = 0; c < I.cols; c++)
		{
			cn = cn + c * I(r, c);
		}

	cn = cn / area(I);
	return cn;
}

//elongation axis
double computeFi(Mat_<uchar> I) {
	int Y = 0;
	int X = 0;
	int xL = 0, xR = 0;
	int rn = rCenter(I);
	int cn = cCenter(I);

	for (int r = 0; r < I.rows; r++)
	{
		for (int c = 0; c < I.cols; c++)
		{
			Y = Y + (r - rn) * (c - cn) * I(r, c);
			xL = xL + (c - cn) * (c - cn) * I(r, c);
			xR = xR + (r - rn) * (r - rn) * I(r, c);
		}
	}

	Y = 2 * Y;
	X = xL - xR;

	double twoFi = atan2((double)Y, (double)X);
	double Fi = twoFi / 2;
	return Fi;
}

bool eps(double a, double b, double e) {
	if (abs(a - b) < e) return true;
	return false;
}

vector<int> computeIntersectionPoints(Mat_<uchar> img) {
	//f(x) = ax+b
	vector<int> p;
	double e = 0.51;
	Mat_<uchar> I = grayScaleToI(img);
	double fi = computeFi(I);
	int r = rCenter(I);
	int c = cCenter(I);
	double a = fi / 0.785398f;
	double b = r - c * a;

	for (int i = 0; i < img.cols - 1; i++)
	{
		if (eps((a * i + b), 0.0f, e)) {
			p.push_back(0);
			p.push_back(i);
		}
	}
	for (int i = 0; i < img.rows - 1; i++)
	{
		if (eps((a * img.cols + b), (double)i, e)) {
			p.push_back(i);
			p.push_back(img.cols);
		}
	}
	for (int i = 1; i < img.cols; i++)
	{
		if (eps((a * i + b), (double)img.rows, e)) {
			p.push_back(img.rows);
			p.push_back(i);
		}
	}
	for (int i = 1; i < img.rows; i++)
	{
		if (eps((a * 0 + b), (double)i, e)) {
			p.push_back(i);
			p.push_back(0);
		}
	}
	return p;
}

bool onBorder(Mat_<uchar> I, int r, int c) {
	if (I(r, c) == 1) {
		if (r == 0 || r == I.rows - 1)
			return true;
		else if (c == 0 || c == I.cols - 1)
			return true;
		else if (I(r - 1, c) == 0 || I(r + 1, c) == 0 || I(r, c - 1) == 0 || I(r, c + 1) == 0)
			return true;
	}
	return false;
}

int perimeter(Mat_<uchar> I) {
	int P = 0;

	for (int r = 0; r < I.rows; r++)
	{
		for (int c = 0; c < I.cols; c++)
		{
			if (onBorder(I, r, c) == true)
				P++;
		}
	}

	return P * (M_PI / 4);
}

double thinnessRatio(Mat_<uchar> I) {
	int A = area(I);
	int P = perimeter(I);

	return 4 * M_PI * ((double)A / (P * P));
}

double aspectRatio(Mat_<uchar> I) {
	int cmin = I.cols, cmax = 0, rmin = I.rows, rmax = 0;

	for (int r = 0; r < I.rows; r++)
	{
		for (int c = 0; c < I.cols; c++)
		{
			if (I(r, c) == 1) {
				if (c < cmin)
					cmin = c;
				if (c > cmax)
					cmax = c;
				if (r < rmin)
					rmin = r;
				if (r > rmax)
					rmax = r;
			}
		}
	}
	return (double)(cmax - cmin + 1) / (rmax - rmin + 1);
}

void imgProperties() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> I = grayScaleToI(img);
		imshow("image", img);

		printf("Area = %d\n", area(I));
		printf("Center: r = %d  c = %d\n", rCenter(I), cCenter(I));
		printf("Elongation axis = %lf\n", computeFi(I));
		printf("Perimeter = %d\n", perimeter(I));
		printf("thinnessRatio = %lf\n", thinnessRatio(I));
		printf("aspectRatio = %lf\n\n", aspectRatio(I));

		waitKey();
	}
}

void borderCenterAxis() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> I = grayScaleToI(src);
		imshow("image", src);

		Mat_<uchar> img = Mat_<uchar>(I.rows, I.cols);

		for (int r = 0; r < I.rows; r++)
		{
			for (int c = 0; c < I.cols; c++)
			{
				img(r, c) = 255;

				// contur
				if (onBorder(I, r, c) == true)
					img(r, c) = 0;
			}
		}

		//centru de masa
		int rn = rCenter(I);
		int cn = cCenter(I);
		img(rn, cn) = 0;

		vector<int> p = computeIntersectionPoints(src);
		//line(img, Point(p.at(1), p.at(0)), Point(p.at(3), p.at(2)), (0, 0, 0), 2);

		imshow("imag", img);
		waitKey();
	}
}

vector<int> hProjection(Mat_<uchar> I) {
	std::vector<int> hi;
	int cnt = 0;
	for (int r = 0; r < I.rows; r++) {
		cnt = 0;
		for (int c = 0; c < I.cols; c++)
		{
			if (I(r, c) == 1) {
				cnt++;
			}
		}
		hi.push_back(cnt);
	}
	return hi;
}

vector<int> vProjection(Mat_<uchar> I) {
	std::vector<int> vi;
	int cnt = 0;
	for (int c = 0; c < I.cols; c++)
	{
		cnt = 0;
		for (int r = 0; r < I.rows; r++) {
			if (I(r, c) == 1) {
				cnt++;
			}
		}
		vi.push_back(cnt);
	}
	return vi;
}

void showProjections(Mat_<uchar> img, vector<int> vProj, vector<int> hProj) {
	Mat_<Vec3b> dst = Mat_<Vec3b>(img.rows, img.cols);
	Vec3b color = Vec3b(156, 185, 31);

	for (int r = 0; r < dst.rows; r++)
		for (int c = 0; c < dst.cols; c++)
			dst(r, c) = Vec3b(255, 255, 255);


	for (int i = 0; i < vProj.size(); i++)
	{
		for (int j = 0; j < vProj.at(i); j++)
		{
			dst(img.rows - 1 - j, i) = color;
		}
	}
	for (int i = 0; i < hProj.size(); i++)
	{
		for (int j = 0; j < hProj.at(i); j++)
		{
			dst(i, j) = color;
		}
	}
	imshow("original", img);
	imshow("projections", dst);
}

void testProjections() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> I = grayScaleToI(img);
		Mat_<uchar> dst = img;

		vector<int> vi = vProjection(I);
		vector<int> hi = hProjection(I);

		showProjections(img, vi, hi);
		waitKey(0);
	}
}

Mat_<uchar> colorToI(Mat_<Vec3b> img, Vec3b color) {
	Mat_<uchar> dst = Mat_<uchar>(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j)[0] == color[0] && img(i, j)[1] == color[1] && img(i, j)[2] == color[2])
				dst(i, j) = 1;
			else
				dst(i, j) = 0;
		}
	return dst;
}

bool isInLut(vector<Vec3b> lut, Vec3b color, int& pos) {
	for (int i = 0; i < lut.size(); i++)
		if (color[0] == lut.at(i)[0] && color[1] == lut.at(i)[1] && color[2] == lut.at(i)[2]) {
			pos = i;
			return true;
		}
	return false;
}

bool meetsConditions(Mat_<uchar> I, int THarea, double lowFi, double highFi) {
	if (area(I) < THarea && computeFi(I) < highFi && computeFi(I) > lowFi) return true;
	return false;
}

Mat_<Vec3b> keepObjectsThatMeetCondition(Mat_<Vec3b> img, int THarea, double lowFi, double highFi) {
	Mat_<Vec3b> dst = Mat_<Vec3b>(img.rows, img.cols);
	vector<Vec3b> lut;
	vector<bool> mCond;
	int q = 0;

	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++)
			if (!isInLut(lut, img(r, c), q))
				lut.push_back(img(r, c));

	for (int i = 0; i < lut.size(); i++) {
		Mat_<uchar> I = colorToI(img, lut.at(i));
		mCond.push_back(meetsConditions(I, THarea, lowFi, highFi));
	}

	// keep the objects that meets the conditions
	for (int r = 0; r < img.rows; r++)
		for (int c = 0; c < img.cols; c++) {
			Vec3b color = img(r, c);
			int pos = -1;
			isInLut(lut, color, pos);
			mCond.at(pos) ? (dst(r, c) = color) : (dst(r, c) = Vec3b(255, 255, 255));
		}
	return dst;
}

void testKeepObjThatMeetCond() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		int THarea;
		double lowFi, highFi;

		printf("THarea = ");
		scanf("%d", &THarea);
		printf("lowFi = ");
		scanf("%lf", &lowFi);
		printf("highFi = ");
		scanf("%lf", &highFi);

		Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
		imshow("meets cond", keepObjectsThatMeetCondition(img, THarea, lowFi, highFi));

		waitKey(0);
	}
}



// laborator 05

vector<Point_<int>> N4(Mat_<uchar> img, Point_<int> p) {
	vector<Point_<int>> neighbours;
	int i = p.x;
	int j = p.y;

	if (isInside(img, i - 1, j))
		neighbours.push_back(Point(i - 1, j));
	if (isInside(img, i, j - 1))
		neighbours.push_back(Point(i, j - 1));
	if (isInside(img, i + 1, j))
		neighbours.push_back(Point(i + 1, j));
	if (isInside(img, i, j + 1))
		neighbours.push_back(Point(i, j + 1));

	return neighbours;
}

vector<Point_<int>> N8(Mat_<uchar> img, Point_<int> p) {
	vector<Point_<int>> neighbours;
	for (int i = p.x - 1; i <= p.x + 1; i++)
		for (int j = p.y - 1; j <= p.y + 1; j++)
		{
			if (isInside(img, i, j) && (i != p.x || j != p.y))
				neighbours.push_back(Point(i, j));
		}
	return neighbours;
}

Mat_<Vec3b> breadthTraversal(Mat_<uchar> img) {
	int label = 0;
	Mat_<int> labels(img.rows, img.cols);
	labels = 0;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img(i, j) == 0 && labels(i, j) == 0) {
				label++;
				queue<Point_<int>> Q;
				labels(i, j) = label;
				Q.push(Point(i, j));
				while (!Q.empty()) {
					Point_<int> q = Q.front();
					Q.pop();
					//vector<Point_<int>> neighbours = N8(img, q);  //N8
					vector<Point_<int>> neighbours = N4(img, q);  //N4
					for (int n = 0; n < neighbours.size(); n++)
					{
						if (img(neighbours.at(n).x, neighbours.at(n).y) == 0
							&& (labels(neighbours.at(n).x, neighbours.at(n).y) == 0)) {
							labels(neighbours.at(n).x, neighbours.at(n).y) = label;
							Q.push(neighbours.at(n));
						}
					}
				}
			}
		}
	}


	vector<Vec3b> colors;
	colors.push_back(Vec3b(255, 255, 255));
	for (int i = 1; i <= label; i++)
	{
		Vec3b clr(rand() % 256, rand() % 256, rand() % 256);
		colors.push_back(clr);
	}

	Mat_<Vec3b> dst(img.rows, img.cols);
	for (int r = 0; r < dst.rows; r++)
		for (int c = 0; c < dst.cols; c++) {
			dst(r, c) = colors.at(labels(r, c));
		}
	return dst;
}

void testBreadthTraversal() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<Vec3b> dst = breadthTraversal(img);
		imshow("original", img);
		imshow("breadthTraversal", dst);

		waitKey(0);
	}
}

void twoPasses() {

	char fname[MAX_PATH];
	while (openFileDlg(fname)) {

		Mat_<uchar> image = imread(fname, IMREAD_GRAYSCALE);
		imshow("Display Image", image);
		int label = 0;
		Mat_<int> labels(image.rows, image.cols);
		labels = 0;
		vector<vector<int>> edges;
		edges.push_back({});

		int di[] = { 0, -1, -1, -1 };
		int dj[] = { -1, -1, 0, 1 };

#define NR_VEC 4

		int i, j, k;
		for (i = 0; i < image.rows; i++)  // RANDURI
			for (j = 0; j < image.cols; j++) { // COLOANE
				if ((image(i, j) == 0) && (labels(i, j) == 0)) {
					//                printf("gasit %d %d\n", i, j);
					vector<int> L;
					for (k = 0; k < NR_VEC; k++) {
						int vi = i + di[k];
						int vj = j + dj[k];
						if ((0 <= vi) && (vi < image.rows) && (0 <= vj) && (vj < image.cols)) {
							int label_vecin = labels(vi, vj);
							if (label_vecin > 0) {
								L.push_back(label_vecin);
							}
						}
					}//for all neighbors
					if (L.size() == 0) {
						label++;
						edges.push_back(vector<int>());
						labels(i, j) = label;
					}
					else {
						int x = L[0];
						for (k = 1; k < L.size(); k++) {
							x = min(x, L[k]);
						}
						labels(i, j) = x;
						for (k = 0; k < L.size(); k++) {
							int y = L[k];
							if (y != x) {
								edges[x].push_back(y);
								edges[y].push_back(x);
							}
						}
					}//end else

				}//end if pixel obiect.
			}//end for all pixels

		int newlabel = 0;
		vector<int> newlabels(label + 1, 0);
		for (i = 1; i < label; i++) {
			if (newlabels[i] == 0) {
				newlabel++;
				queue<int> Q;
				newlabels[i] = newlabel;
				Q.push(i);
				while (Q.size() > 0) {
					int x = Q.front();
					Q.pop();
					for (k = 0; k < edges[x].size(); k++) {
						int y = edges[x][k];
						if (newlabels[y] == 0) {
							newlabels[y] = newlabel;
							Q.push(y);
						}
					}
				}
			}
		}

		default_random_engine gen;
		uniform_int_distribution<int> d(0, 255);

#define NR_MAX_CULORI 300

		vector<Vec3b> culori(NR_MAX_CULORI);
		culori[0] = Vec3b(255, 255, 255);
		for (i = 1; i < NR_MAX_CULORI; i++) {
			culori[i][0] = d(gen);
			culori[i][1] = d(gen);
			culori[i][2] = d(gen);
		}

		Mat_<Vec3b> labels_for_show(image.rows, image.cols);
		for (i = 0; i < image.rows; i++)  // RANDURI
			for (j = 0; j < image.cols; j++) { // COLOANE
				labels_for_show(i, j) = culori[newlabels[labels(i, j)]];
			}
		//namedWindow("etichete", WINDOW_NORMAL | WINDOW_GUI_EXPANDED);
		imshow("etichete", labels_for_show);
		waitKey(0);
	}
}



// laborator 06

Mat_<uchar> contourTracking(Mat_<uchar> img) {

	Mat_<uchar> dst(img.rows, img.cols);
	vector<Point_<int>> P;
	vector<int> codes;
	int dir = 7;
	bool finish = false;

	int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Point current = Point(i, j);
			if (img(current.x, current.y) < 200) {

				dst(current.x, current.y) = 0; // black color
				P.push_back(current);
				codes.push_back(dir);
				int vi, vj;

				while (!finish) {

					(dir % 2 == 0) ? (dir = (dir + 7) % 8) : (dir = (dir + 6) % 8);

					for (int v = 0; v < 8; v++) {

						int tmpdir = (dir + v) % 8;
						vi = current.x + di[tmpdir];
						vj = current.y + dj[tmpdir];

						if (img(vi, vj) < 200) {
							dir = tmpdir;
							dst(vi, vj) = 0; // black color
							current = Point(vi, vj);
							P.push_back(current);
							codes.push_back(dir);
							break;
						}
					}

					if (P.size() > 2 && (current.x == P.at(1).x && current.y == P.at(1).y &&
						P.at(0).x == P.at(P.size() - 2).x && P.at(0).y == P.at(P.size() - 2).y))
						finish = true;

				}
			}
			if (finish) break;
		}
		if (finish) break;
	}

	vector<int> derivative;

	for (int i = 0; i < codes.size(); i++)
	{
		derivative.push_back((8 - codes.at(i % (codes.size())) + codes.at((i + 1) % (codes.size()))) % 8);
		printf("%d ", codes.at(i));
	}

	printf("\n\n");

	for (int i = 0; i < derivative.size(); i++)
		printf("%d ", derivative.at(i));
	return dst;
}

void testContourTracking() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst = contourTracking(img);
		imshow("original", img);
		imshow("contourTracking", dst);

		waitKey(0);
	}
}

void reconstructImage() {
	char fname[MAX_PATH];
	if (openFileDlg(fname)) {
		Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);

		FILE* f = fopen("D:\\UTC_an3_sem2\\PI\\Laborator\\lab06\\imaginiTest\\reconstruct.txt", "r");
		int si, sj, nr;

		// read start point and no of codes
		fscanf(f, "%d", &si);
		fscanf(f, "%d", &sj);
		fscanf(f, "%d", &nr);

		int di[] = { 0, -1, -1, -1, 0, 1, 1, 1 };
		int dj[] = { 1, 1, 0, -1, -1, -1, 0, 1 };

		for (int i = 0; i < nr; i++) {
			int dir;
			fscanf(f, "%d", &dir);
			si = si + di[dir];
			sj = sj + dj[dir];
			img(si, sj) = 0;
		}

		fclose(f);
		imshow("reconstruct", img);
		waitKey(0);
	}
}


// laborator 07

Mat_<uchar> dilation(Mat_<uchar> src, Mat_<uchar> mask) {
	Mat_<uchar> dst(src.rows, src.cols);
	int mi = mask.rows / 2;
	int mj = mask.cols / 2;

	// init 
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			dst(i, j) = 0;


	for (int i = mask.rows / 2; i < src.rows - (mask.rows / 2); i++)
	{
		for (int j = mask.cols / 2; j < src.cols - (mask.cols / 2); j++)
		{

			if (src(i, j) == 1) {

				for (int ii = 0; ii < mask.rows; ii++)
				{
					for (int jj = 0; jj < mask.cols; jj++)
					{
						dst((i + (ii - mi)), (j + (jj - mj))) = mask(ii, jj);
					}
				}
			}
		}
	}

	return dst;
}

Mat_<uchar> erosion(Mat_<uchar> src, Mat_<uchar> mask) {
	Mat_<uchar> dst(src.rows, src.cols);


	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			dst(i, j) = 0;

	int structPixels = 0, matchingPixels = 0;
	int mi = mask.rows / 2;
	int mj = mask.cols / 2;


	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			matchingPixels = 0;
			for (int ii = 0; ii < mask.rows; ii++)
			{
				for (int jj = 0; jj < mask.cols; jj++)
				{
					int di = i + (ii - mi);
					int dj = j + (jj - mj);

					if (isInside(src, di, dj))
						if (src(di, dj) == mask(ii, jj))
							matchingPixels++;
				}
			}
			// (9) numarul de pixeli din elemetul structural
			if (matchingPixels == 9)
				dst(i, j) = 1;
			else
				dst(i, j) = 0;

		}
	}
	return dst;
}

Mat_<uchar> opening(Mat_<uchar> src, Mat_<uchar> mask) {
	Mat_<uchar> dst(src.rows, src.cols);
	dst = erosion(src, mask);
	dst = dilation(dst, mask);
	return dst;
}

Mat_<uchar> closing(Mat_<uchar> src, Mat_<uchar> mask) {
	Mat_<uchar> dst(src.rows, src.cols);
	dst = dilation(src, mask);
	dst = erosion(dst, mask);
	return dst;
}

Mat_<uchar> difference(Mat_<uchar> A, Mat_<uchar> B) {
	Mat_<uchar> dst = A;

	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			if (A(i, j) == 1 && B(i, j) == 1)
				dst(i, j) = 0;
	return dst;
}

Mat_<uchar> extractContour(Mat_<uchar> src, Mat_<uchar> mask) {
	Mat_<uchar> dst(src.rows, src.cols);
	dst = difference(src, erosion(src, mask));
	return dst;
}

Mat_<uchar> intersection(Mat_<uchar> m1, Mat_<uchar> m2) {
	Mat_<uchar> dst(m1.rows, m1.cols);

	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			if (m1(i, j) == m2(i, j))
				dst(i, j) = m1(i, j);
			else
				dst(i, j) = 0; //background color

	return dst;
}

Mat_<uchar> reunion(Mat_<uchar> m1, Mat_<uchar> m2) {
	Mat_<uchar> dst(m1.rows, m1.cols);

	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			if (m1(i, j) == 1 || m2(i, j) == 1)
				dst(i, j) = 1;
			else
				dst(i, j) = 0;

	return dst;
}

Mat_<uchar> complement(Mat_<uchar> img) {
	Mat_<uchar> dst(img.rows, img.cols);

	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			if (img(i, j) == 1)
				dst(i, j) = 0;
			else
				dst(i, j) = 1;
	return dst;
}

bool areEqual(Mat_<uchar> img1, Mat_<uchar> img2) {
	for (int i = 0; i < img1.rows; i++)
		for (int j = 0; j < img1.cols; j++)
			if (img1(i, j) != img2(i, j)) return false;
	return true;
}

Mat_<uchar> fillRegion(Mat_<uchar> src, Mat_<uchar> mask) {
	Mat_<uchar> dst(src.rows, src.cols);
	Point p;

	//compute p
	bool found = false;
	for (int i = 1; i < src.rows; i++) {
		for (int j = 1; j < src.cols; j++) {
			if (src(i, j) == 0 && src(i - 1, j) == 1 && src(i, j - 1) == 1)
			{
				p.x = i;
				p.y = j;
				found = true;
			}
			if (found) break;
		}
		if (found) break;
	}

	Mat_<uchar> Xprev(src.rows, src.cols);
	Mat_<uchar> Xcurr(src.rows, src.cols);

	for (int i = 1; i < src.rows; i++)
		for (int j = 1; j < src.cols; j++) {
			Xprev(i, j) = 0;
			Xcurr(i, j) = 0;
		}

	Xcurr(p.x, p.y) = 1;
	Mat_<uchar> AC = complement(src);

	while (!areEqual(Xprev, Xcurr)) {
		Xprev = Xcurr;
		Xcurr = intersection(dilation(Xprev, mask), AC);
	}

	dst = reunion(src, Xcurr);
	return dst;
}

void test() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> I = grayScaleToI(src);
		Mat_<uchar> dst(src.rows, src.cols);

		int W = 3, H = 3;

		Mat_<uchar> mask(H, W);
		for (int i = 0; i < H; i++)
			for (int j = 0; j < W; j++)
				mask(i, j) = 1;

		/*
		mask(0, 0) = 0;
		mask(0, 1) = 1;
		mask(0, 2) = 0;
		mask(1, 0) = 1;
		mask(1, 1) = 1;
		mask(1, 2) = 1;
		mask(2, 0) = 0;
		mask(2, 1) = 1;
		mask(2, 2) = 0;*/

		int nr = 1;
		printf("nr = ");
		scanf("%d", &nr);

		for (int i = 0; i < nr; i++)
		{
			//dst = dilation(I, mask);
			//dst = erosion(I, mask);
			//dst = opening(I, mask);
			//dst = closing(I, mask);
			dst = extractContour(I, mask);
			I = dst;
		}

		dst = ItoGrayScale(dst);

		imshow("original", src);
		imshow("morph", dst);
		waitKey(0);
	}
}

void testRegionFilling() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		/*
		Mat_<uchar> i1 = imread("D:\\UTC_an3_sem2\\PI\\Laborator\\lab07\\Morphological_Op_Images\\6_RegionFilling\\t1.bmp", IMREAD_GRAYSCALE);
		Mat_<uchar> i2 = imread("D:\\UTC_an3_sem2\\PI\\Laborator\\lab07\\Morphological_Op_Images\\6_RegionFilling\\t2.bmp", IMREAD_GRAYSCALE);

		i1 = grayScaleToI(i1);
		i2 = grayScaleToI(i2);

		Mat_<uchar> ss = reunion(i1, i2);
		ss = ItoGrayScale(ss);

		imshow("ss", ss);
		waitKey(0);*/


		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> I = grayScaleToI(src);
		Mat_<uchar> dst(src.rows, src.cols);

		Mat_<uchar> mask(3, 3);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				mask(i, j) = 1;

		/*
		mask(0, 0) = 0;
		mask(0, 1) = 1;
		mask(0, 2) = 0;
		mask(1, 0) = 1;
		mask(1, 1) = 1;
		mask(1, 2) = 1;
		mask(2, 0) = 0;
		mask(2, 1) = 1;
		mask(2, 2) = 0;*/

		dst = fillRegion(I, mask);
		dst = ItoGrayScale(dst);

		imshow("original", src);
		imshow("fill", dst);
		waitKey(0);
	}
}


// laborator 08

int average(Mat_<uchar> img) {
	vector<int> histogram = computeHistogram(img);
	int average = 0;
	int M = img.cols * img.rows;

	for (int g = 0; g < 256; g++)
		average += g * histogram.at(g);
	average /= M;
	return average;
}

double deviation(Mat_<uchar> img) {
	vector<float> p = computeFDP(img);
	int m = average(img);
	int d = 0;

	for (int g = 0; g < 256; g++)
		d += (g - m) * (g - m) * p.at(g);

	return sqrt(d);
}

int cumulativeHistogram(Mat_<uchar> img, int g) {
	vector<int> histogram = computeHistogram(img);
	if (g < 0 || g >= histogram.size()) {
		printf("\nERROR: g out of range\n");
		exit(1);
	}
	int acc = 0;
	for (int j = 0; j <= g; j++)
		acc += histogram.at(j);
	return acc;
}

Mat_<uchar> automaticBinarization(Mat_<uchar> img) {
	Mat_<uchar> dst(img.rows, img.cols);
	vector<int> histogram = computeHistogram(img);
	int Imin, Imax, T, Tprev = 0;
	int m1 = 0, m2 = 0, n1 = 0, n2 = 0;
	int i = 0;
	float error = 0.1;

	while (histogram.at(i) == 0)
		i++;
	Imin = i;

	i = 255;
	while (histogram.at(i) == 0)
		i--;
	Imax = i;
	//printf("%d %d", Imin, Imax);
	T = (Imin + Imax) / 2;

	while (abs(T - Tprev) > error) {
		for (int g = Imin; g <= T; g++) {
			n1 += histogram[g];
			m1 += g * histogram[g];
		}

		for (int g = T + 1; g <= Imax; g++) {
			n2 += histogram[g];
			m2 += g * histogram[g];
		}

		m1 = m1 / n1;
		m2 = m2 / n2;

		Tprev = T;
		T = (m1 + m2) / 2;
	}

	printf("T = %d  \n", T);
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (img(i, j) < T)
				dst(i, j) = 0;
			else
				dst(i, j) = 255;
		}
	}

	return dst;
}

Mat_<uchar> negative(Mat_<uchar> img) {
	Mat_<uchar> dst(img.rows, img.cols);
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++)
			dst(i, j) = 255 - img(i, j);

	return dst;
}

Mat_<uchar> brightness(Mat_<uchar> img, int offset) {
	Mat_<uchar> dst(img.rows, img.cols);
	for (int i = 0; i < dst.rows; i++)
		for (int j = 0; j < dst.cols; j++) {
			int tmp = img(i, j) + offset;
			if (tmp < 0) tmp = 0;
			if (tmp > 255) tmp = 255;
			dst(i, j) = tmp;
		}

	return dst;
}

void testFeatures() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		imshow("original", src);
		showHistogram("Histograma", computeHistogram(src).data(), 256, 256);

		printf("average:%d  deviation:%f  cumulative histogram:%d\n",
			average(src), deviation(src), cumulativeHistogram(src, 100));

		waitKey(0);
	}
}

void testBinarization() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols);
		imshow("original", src);
		showHistogram("Histograma", computeHistogram(src).data(), 256, 256);

		dst = automaticBinarization(src);
		imshow("automatic binarization", dst);

		waitKey(0);
	}
}

void testNegative() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols);
		imshow("original", src);
		showHistogram("original H", computeHistogram(src).data(), 256, 256);

		dst = negative(src);
		imshow("negative", dst);
		showHistogram("negative H", computeHistogram(dst).data(), 256, 256);

		waitKey(0);
	}
}

void testBrightness() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols);
		imshow("original", src);
		showHistogram("original H", computeHistogram(src).data(), 256, 256);

		int offset = 0;
		printf("offset = ");
		scanf("%d", &offset);

		dst = brightness(src, offset);
		imshow("bright", dst);
		showHistogram("bright H", computeHistogram(dst).data(), 256, 256);

		waitKey(0);
	}
}

Mat_<uchar> changeContrast(Mat_<uchar>src, int gOutMin, int gOutMax) {

	if (gOutMin > gOutMax || gOutMin < 0 || gOutMin > 255 || gOutMax < 0 || gOutMax > 255) {
		printf("\nERROR: inconsistent data for gOutMin or gOutMax \n");
		exit(1);
	}
	Mat_<uchar> dst(src.rows, src.cols);

	vector<int> histogram = computeHistogram(src);
	int i = 0, gInMin, gInMax;

	while (histogram[i] == 0)
		i++;
	gInMin = i;

	i = 255;
	while (histogram[i] == 0)
		i--;
	gInMax = i;

	vector<int> LUT(256);

	for (int g = 0; g < 256; g++)
	{
		LUT[g] = gOutMin + (g - gInMin) * ((gOutMax - gOutMin) / (gInMax - gInMin));
	}

	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			//dst(i, j) = LUT[src(i, j)];
			dst(i, j) = gOutMin + (src(i, j) - gInMin) * ((gOutMax - gOutMin) / (gInMax - gInMin));
			if (dst(i, j) < 0) {
				dst(i, j) = 0;
			}
			if (dst.at<uchar>(i, j) > 255) {
				dst(i, j) = 255;
			}
		}
	}

	return dst;
}


void testChangeContrast() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols);
		imshow("original", src);
		showHistogram("original H", computeHistogram(src).data(), 256, 256);

		int gOutMin, gOutMax;
		printf("gOutMin = ");
		scanf("%d", &gOutMin);
		printf("gOutMax = ");
		scanf("%d", &gOutMax);

		dst = changeContrast(src, gOutMin, gOutMax);

		imshow("contrast", dst);
		showHistogram("contrast H", computeHistogram(dst).data(), 256, 256);

		waitKey(0);
	}
}


Mat_<uchar> gamma(Mat_<uchar> src, float g) {

	float L = 255;
	Mat_<uchar> dst(src.rows, src.cols);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float gOut = L * pow((src(i, j) / L), g);

			if (gOut < 0) {
				gOut = 0;
			}

			if (gOut > L) {
				gOut = L;
			}
			dst(i, j) = (int)gOut;
		}
	}

	return dst;
}


void testGamma() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);
		Mat_<uchar> dst(src.rows, src.cols);
		imshow("original", src);
		showHistogram("original H", computeHistogram(src).data(), 256, 256);

		float g;
		printf("gamma = ");
		scanf("%f", &g);

		dst = gamma(src, g);

		imshow("gamma", dst);
		showHistogram("gamma H", computeHistogram(dst).data(), 256, 256);

		waitKey(0);
	}
}



int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n\n");

		printf(" 10 - Lab 1: changing gray levels with an additive factor\n");
		printf(" 11 - Lab 1: changing gray levels with an multiplicative factor\n");
		printf(" 12 - Lab 1: 4 equal squares colored as follows: white, red, green, yellow.\n\n");

		printf(" 20 - Lab 2: copies the R, G, B channels of an image in three matrices\n");
		printf(" 21 - Lab 2: color image to grayscale\n");
		printf(" 22 - Lab 2: grayscale to binary\n");
		printf(" 23 - Lab 2: RGB to HSV\n\n");

		printf(" 30 - Lab 3: test histogram\n");
		printf(" 31 - Lab 3: test FDP\n");
		printf(" 32 - Lab 3: test gray reduction\n");
		printf(" 33 - Lab 3: test Floyd Steinberg\n\n");

		printf(" 40 - Lab 4: image properties\n");
		printf(" 41 - Lab 4: new image: border center axis \n");
		printf(" 42 - Lab 4: image projections \n");
		printf(" 43 - Lab 4: show objects that meet a certain condition\n\n");

		printf(" 50 - Lab 5: label objects -> breadth traversal \n");
		printf(" 51 - Lab 5: label objects -> two passes \n\n");

		printf(" 60 - Lab 6: contour tracking \n");
		printf(" 61 - Lab 6: reconstruct image \n\n");

		printf(" 70 - Lab 7:  dilation/erosion \n");
		printf(" 71 - Lab 7:  fill region \n\n");

		printf(" 80 - Lab 8:  mean, standard deviation, histogram and cumulative histogram\n");
		printf(" 81 - Lab 8:  automatic binarization\n");
		printf(" 82 - Lab 8:  test negative \n");
		printf(" 83 - Lab 8:  test brightness \n");
		printf(" 84 - Lab 8:  test change contrast \n");
		printf(" 85 - Lab 8:  test gamma correction \n");

		printf("\n 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;

		case 10:
			changeGrayAdditive();
			break;
		case 11:
			changeGrayMultiplicative();
			break;
		case 12:
			fourSquares();
			break;

		case 20:
			splitRGB();
			break;
		case 21:
			colorToGrayscale();
			break;
		case 22:
			grayscaleToBinary();
			break;
		case 23:
			RGBtoHSV();
			break;

		case 30:
			testHistogram();
			break;
		case 31:
			testFDP();
			break;
		case 32:
			testGrayReduction();
			break;
		case 33:
			testFloydSteinberg();
			break;

		case 40:
			imgProperties();
			break;
		case 41:
			borderCenterAxis();
			break;
		case 42:
			testProjections();
			break;
		case 43:
			testKeepObjThatMeetCond();
			break;

		case 50:
			testBreadthTraversal();
			break;
		case 51:
			twoPasses();
			break;

		case 60:
			testContourTracking();
			break;
		case 61:
			reconstructImage();
			break;

		case 70:
			test();
			break;
		case 71:
			testRegionFilling();
			break;

		case 80:
			testFeatures();
			break;
		case 81:
			testBinarization();
			break;
		case 82:
			testNegative();
			break;
		case 83:
			testBrightness();
			break;
		case 84:
			testChangeContrast();
			break;
		case 85:
			testGamma();
			break;
		}
	} while (op != 0);
	return 0;
}