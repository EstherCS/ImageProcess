#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/photo.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h> 
#include <stdlib.h>

using namespace std;
using namespace cv;

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale);
void detectEYE(Mat& img, CascadeClassifier& cascade, CascadeClassifier& eye_cascade, double scale);
void detectLips(Mat& img, CascadeClassifier& cascade, CascadeClassifier& eye_cascade, double scale);
void sketch(Mat &image);
void Shutters(Mat &image);
void BlackWhite(Mat & image);
void Fire(Mat & image);
void Ice(Mat & image);
void Colorful(Mat & image);
void Old(Mat & image);

String face_cascade_name = "haarcascade_frontalface_alt.xml";		// 人臉的訓臉數據
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";	// 人眼的訓練數據
String smile_cascade_name = "haarcascade_smile.xml";

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

bool isFace = false;

int main(int argc, const char** argv)
{
	Mat image, image2;
	double scale = 1;
	bool show = true;
	image = imread("a.jpg");


	if (!face_cascade.load(face_cascade_name))
	{
		cerr << "Error loading face cascade\n" << endl;
		return 0;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "Error loading eye cascade\n";
		return 0;
	}

	if (!image.empty()) // 有讀取到照片
	{
		int choose, case3, case5, Auto;
		cout << "請選擇要一鍵美化還是手動美化 : \n" << "1. 一鍵美化 \n" << "2. 手動美化 \n";
		cin >> Auto;
		
		if (Auto == 1)
		{
			imshow("美化前", image);
			detectAndDraw(image, face_cascade, scale);
			if (isFace == false)
			{
				Mat dst;

				int value1 = 3, value2 = 1;

				int dx = value1 * 5;		// 雙邊濾波參數
				double fc = value1 * 12.5;  // 雙邊濾波參數  
				int p = 50;//透明度  
				Mat temp1, temp2, temp3, temp4;

				// 雙邊濾波
				bilateralFilter(image, temp1, dx, fc, fc);

				temp2 = (temp1 - image + 128);

				// 高斯模糊 
				GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

				temp4 = image + 2 * temp3 - 255;

				dst = (image*(100 - p) + temp4 * p) / 100;
				dst.copyTo(image);
			}
			detectEYE(image, face_cascade, eyes_cascade, scale);
			detectLips(image, face_cascade, eyes_cascade, scale);
			int temp = rand() % 4;
			switch (temp)
			{
			case 1:
				image = image + Scalar(0, 0, 25);
				break;
			case 2:
				image = image + Scalar(25, 0, 0);
				break;
			case 3:
				image = image + Scalar(75, 75, 75);
				break;
			case 4:
				image = image + Scalar(-75, -75, -75);
				break;
			}
			imshow("結果", image);
		}
		else if (Auto == 2)
		{
			cout << "請輸入想選擇的功能: " << endl;
			cout << "1. 臉部磨皮\n" << "2. 去除黑眼圈\n" << "3. 美白\n" << "4. 上唇彩\n" << "5. 特殊處理\n";
			cin >> choose;
			switch (choose)
			{
			case 1:
				imshow("臉部磨皮前", image);
				detectAndDraw(image, face_cascade, scale);
				if (isFace == false)
				{
					Mat dst;

					int value1 = 3, value2 = 1;

					int dx = value1 * 5;		// 雙邊濾波參數
					double fc = value1 * 12.5;  // 雙邊濾波參數  
					int p = 50;//透明度  
					Mat temp1, temp2, temp3, temp4;

					// 雙邊濾波
					bilateralFilter(image, temp1, dx, fc, fc);

					temp2 = (temp1 - image + 128);

					// 高斯模糊 
					GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

					temp4 = image + 2 * temp3 - 255;

					dst = (image*(100 - p) + temp4 * p) / 100;
					dst.copyTo(image);
				}
				break;
			case 2:
				imshow("去除黑眼圈前", image);
				detectEYE(image, face_cascade, eyes_cascade, scale);
				break;
			case 3:
				cout << "請選想要的美白效果 : \n";
				cout << "1. 紅潤  " << "2. 自然  " << "3. 白皙  " << "4. 黑化  \n";
				cin >> case3;
				imshow("美白前", image);
				switch (case3)
				{
				case 1:
					image = image + Scalar(0, 0, 25);
					break;
				case 2:
					image = image + Scalar(25, 0, 0);
					break;
				case 3:
					image = image + Scalar(75, 75, 75);
					break;
				case 4:
					image = image + Scalar(-75, -75, -75);
					break;
				}
				break;
			case 4:
				imshow("上唇彩前", image);
				detectLips(image, face_cascade, eyes_cascade, scale);
				break;
			case 5:
				cout << "請選想要的影像處理效果 : \n";
				cout << "1. 風格化  " << "2. 百葉窗  " << "3. 黑白  " << "4. 熱情  " << "5. 高冷  " << "6. 多彩變換  " << "7. 輪廓描邊  "
					<< "8. 增強  " << "9. 懷舊  " << "10. 素描  \n";
				cin >> case5;
				imshow("各式濾鏡", image);
				switch (case5)
				{
				case 1:
					stylization(image, image, 50, 0.15);
					break;
				case 2:
					Shutters(image);
					break;
				case 3:
					BlackWhite(image);
					break;
				case 4:
					Fire(image);
					break;
				case 5:
					Ice(image);
					break;
				case 6:
					Colorful(image);
					show = false;
					break;
				case 7:
					pencilSketch(image, image, image2, 50, 0.15, 0.04);
					imshow("output2", image2);
					break;
				case 8:
					detailEnhance(image, image, 50, 0.15);
					break;
				case 9:
					Old(image);
					break;
				case 10:
					sketch(image);
					break;
				}
				break;
			}
			if (show)
				imshow("結果", image);
		}

		cv::waitKey(0);
	}
	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade, double scale)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] = {
		CV_RGB(0,0,255),
		CV_RGB(255,165,0),
		CV_RGB(0,245,255),
		CV_RGB(0,255,0),
		CV_RGB(202,225,255),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };// 有多個人臉會用不同顏色

	// 圖片縮小檢測
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);
	// 線性差值
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
	
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		isFace = true;
		srand(time(NULL));
		Scalar color = colors[rand() % 8];

		Point center, left, right;

		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);	// 還原原本大小
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);


		left.x = center.x - radius;
		left.y = cvRound(center.y - radius * 1.3);

		if (left.y<0)
		{
			left.y = 0;
		}

		right.x = center.x + radius;
		right.y = cvRound(center.y + radius * 1.3);

		if (right.y > img.rows)
		{
			right.y = img.rows;
		}

		//rectangle(img, left, right, color);

		Mat roi = img(Range(left.y, right.y), Range(left.x, right.x));
		Mat dst;

		int value1 = 3, value2 = 1;

		int dx = value1 * 5;		// 雙邊濾波參數
		double fc = value1 * 12.5;  // 雙邊濾波參數  
		int p = 50;//透明度  
		Mat temp1, temp2, temp3, temp4;

		// 雙邊濾波
		bilateralFilter(roi, temp1, dx, fc, fc);

		temp2 = (temp1 - roi + 128);

		// 高斯模糊 
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

		temp4 = roi + 2 * temp3 - 255;
		dst = (roi*(100 - p) + temp4 * p) / 100;
		dst.copyTo(roi);
	}
}

void detectLips(Mat& img, CascadeClassifier& cascade, CascadeClassifier& eye_cascade, double scale)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] = {
		CV_RGB(0,0,255),
		CV_RGB(255,165,0),
		CV_RGB(0,245,255),
		CV_RGB(0,255,0),
		CV_RGB(202,225,255),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };// 有多個人眼會用不同顏色

	// 圖片縮小檢測
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);
	// 線性差值
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	cascade.detectMultiScale(smallImg, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		isFace = true;
		Mat faceROI = smallImg(faces[i]);
		vector<Rect> eyes;
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		Point cpt[2];
		for (size_t j = 0; j < eyes.size(); j++)
		{
			srand(time(NULL));
			Scalar color = colors[rand() % 8];

			Rect eyeR;
			eyeR.x = faces[i].x + eyes[j].x;
			eyeR.y = faces[i].y + eyes[j].y;
			eyeR.height = eyes[j].height;
			eyeR.width = eyes[j].width;

			cpt[j].x = eyeR.x + cvRound(eyeR.width / 2.0);
			cpt[j].y = eyeR.y + cvRound(eyeR.height / 2.0);
			//circle(img, cpt[j], 3, CV_RGB(0, 255, 255));
		}

		Point eyeCenter = ((cpt[1] - cpt[0]) / 2) + cpt[0];
		//circle(img, eyeCenter, 3, CV_RGB(255, 0, 255));

		IplImage *image2 = &IplImage(img);
		CvScalar s2;
		s2 = cvGet2D(image2, eyeCenter.y, eyeCenter.x);
		int B2 = s2.val[0];
		int G2 = s2.val[1];
		int R2 = s2.val[2];
		//cout << R2 << " " << G2 << " " << B2 << endl;

		Point mouth;
		mouth.x = eyeCenter.x;
		mouth.y = eyeCenter.y + abs(cpt[1].x - cpt[0].x) + 0.5;
		int R, G, B;
		IplImage *image = &IplImage(img);
		CvScalar s;
		s = cvGet2D(image, mouth.y, mouth.x);
		B = s.val[0];
		G = s.val[1];
		R = s.val[2];
		//cout << R << " " << G << " " << B << endl;
		while (G > 100 && B > 100)
		{
			mouth.y += 1;
			s = cvGet2D(image, mouth.y, mouth.x);
			B = s.val[0];
			G = s.val[1];
			R = s.val[2];
			//cout << R << " " << G << " " << B << endl;
		}
		//circle(img, mouth, 3, CV_RGB(0, 255, 0));

		Point left, right, buttom;
		left.x = cpt[0].x;
		right.x = cpt[1].x;
		left.y = right.y = mouth.y + 0.3; 
		buttom.x = mouth.x;
		buttom.y = mouth.y + abs(right.x - left.x) / 4;
		//circle(img, left, 3, CV_RGB(0, 255, 0));
		//circle(img, right, 3, CV_RGB(0, 255, 0));
		//circle(img, buttom, 3, CV_RGB(255, 255, 0));
		Rect draw;
		draw.x = (left.x < right.x ? left.x : right.x);
		draw.y = mouth.y;
		draw.height = abs(buttom.y - mouth.y);
		draw.width = abs(right.x - left.x);
		//rectangle(img, draw, CV_RGB(255, 255, 255), 1, 1, 0);

		bool first = true;
		int r, g, b;
		for (int x = draw.x; x < draw.x + draw.width; x++)
		{
			for (int y = draw.y; y < draw.y + draw.height; y++)
			{
				IplImage *color = &IplImage(img);
				CvScalar xy;
				xy = cvGet2D(color, y, x);
				B = xy.val[0];
				G = xy.val[1];
				R = xy.val[2];
				int Y = 0.299 * R + 0.587 * G + 0.114 * B;
				int I = 0.596 * R - 0.275 * G - 0.321 * B;
				int Q = 0.212 * R - 0.523 * G + 0.311 * B;
				if ((Y >= 80 && Y <= 220 && I >= 12 && I <= 78 && Q >= 7 && Q <= 25))
				{
					if (first)
					{
						srand(time(NULL));
						r = rand() % 255;
						g = rand() % 255;
						b = rand() % 255;
						int y = 0.299 * r + 0.587 * g + 0.114 * b;
						int i = 0.596 * r - 0.275 * g - 0.321 * b;
						int q = 0.212 * r - 0.523 * g + 0.311 * b;
						while (y < 80 || y > 220 || i < 12 || i > 78 || q < 7 || q > 25)
						{
							r = rand() % 255;
							g = rand() % 255;
							b = rand() % 255;
							y = 0.299 * r + 0.587 * g + 0.114 * b;
							i = 0.596 * r - 0.275 * g - 0.321 * b;
							q = 0.212 * r - 0.523 * g + 0.311 * b;
						}
						first = false;
					}

					img.at<Vec3b>(y, x)[2] = r;
					img.at<Vec3b>(y, x)[0] = b;
					img.at<Vec3b>(y, x)[1] = g;
				}
			}
		}
	}
}

void detectEYE(Mat& img, CascadeClassifier& cascade, CascadeClassifier& eye_cascade, double scale)
{
	int i = 0;
	double t = 0;
	vector<Rect> faces;
	const static Scalar colors[] = {
		CV_RGB(0,0,255),
		CV_RGB(255,165,0),
		CV_RGB(0,245,255),
		CV_RGB(0,255,0),
		CV_RGB(202,225,255),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };// 有多個人眼會用不同顏色

	// 圖片縮小檢測
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);
	// 線性差值
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	cascade.detectMultiScale(smallImg, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		isFace = true;
		Mat faceROI = smallImg(faces[i]);
		vector<Rect> eyes;
		eye_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			srand(time(NULL));
			Scalar color = colors[rand() % 8];

			Rect eyeR;
			eyeR.x = faces[i].x + eyes[j].x;
			eyeR.y = faces[i].y + eyes[j].y;
			eyeR.height = eyes[j].height;
			eyeR.width = eyes[j].width;
			//rectangle(img, eyeR, color, 1, 1, 0);

			Mat roi = img(eyeR);
			Mat dst;

			int value1 = 6, value2 = 2;

			int dx = value1 * 5;		// 雙邊濾波參數
			double fc = value1 * 12.5;  // 雙邊濾波參數
			int p = 50;//透明度
			Mat temp1, temp2, temp3, temp4;

			// 雙邊濾波
			bilateralFilter(roi, temp1, dx, fc, fc);

			temp2 = (temp1 - roi + 128);

			// 高斯模糊
			GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

			temp4 = roi + 2 * temp3 - 255;
			dst = (roi*(100 - p) + temp4 * p) / 100;
			dst.copyTo(roi);
		}
	}
}

void sketch(Mat &image)
{
	Mat Desaturate, Gauss;
	// 去色  
	cvtColor(image, Desaturate, CV_BGR2GRAY);
	// 複製去色的圖並將他反色  
	addWeighted(Desaturate, -1, NULL, 0, 255, Gauss);
	// 對反色的圖做高斯模糊
	GaussianBlur(Gauss, Gauss, Size(11, 11), 0);
	// 將顏色調淡
	Mat output(Gauss.size(), CV_8UC1);
	for (int y = 0; y < image.rows; y++)
	{
		uchar* P0 = Desaturate.ptr<uchar>(y);
		uchar* P1 = Gauss.ptr<uchar>(y);
		uchar* P = output.ptr<uchar>(y);
		for (int x = 0; x < image.cols; x++)
		{
			int des = P0[x];
			int Gau = P1[x];
			// C =MIN( A +（A×B）/（255-B）,255)
			P[x] = (uchar)min((des + (des*Gau) / (255 - Gau)), 255);
		}
	}
	output.copyTo(image);
}

void Shutters(Mat &image)
{
	for (int y = 1; y < image.rows - 1; y++)
	{
		uchar *p0 = image.ptr<uchar>(y);
		uchar *q0 = image.ptr<uchar>(y);

		for (int x = 1; x < image.cols - 1; x++)
		{
			for (int i = 0; i<3; i++)
			{
				int tmp0 = p0[3 * (x + 1) + i] - p0[3 * (x - 1) + i] + 128;  
				if (tmp0<0)
					q0[3 * x + i] = 0;
				else if (tmp0>255)
					q0[3 * x + i] = 255;
				else
					q0[3 * x + i] = tmp0;
			}
		}
	}
}

void BlackWhite(Mat & image)
{
	int width = image.cols;
	int heigh = image.rows;
	RNG rng;
	Mat img(image.size(), CV_8UC3);
	for (int y = 0; y<heigh; y++)
	{
		uchar* P0 = image.ptr<uchar>(y);
		uchar* P1 = img.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			float B = P0[3 * x];
			float G = P0[3 * x + 1];
			float R = P0[3 * x + 2];
			float newB = abs(B - G + B + R)*G / 256;
			float newG = abs(B - G + B + R)*R / 256;
			float newR = abs(G - B + G + R)*R / 256;
			if (newB<0)newB = 0;
			if (newB>255)newB = 255;
			if (newG<0)newG = 0;
			if (newG>255)newG = 255;
			if (newR<0)newR = 0;
			if (newR>255)newR = 255;
			P1[3 * x] = (uchar)newB;
			P1[3 * x + 1] = (uchar)newG;
			P1[3 * x + 2] = (uchar)newR;
		}

	}
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	normalize(gray, image, 255, 0, CV_MINMAX);
}

void Fire(Mat & image)
{
	Mat img;
	image.copyTo(img);
	int width = image.cols;
	int heigh = image.rows;
	Mat dst(img.size(), CV_8UC3);
	for (int y = 0; y<heigh; y++)
	{
		uchar* imgP = img.ptr<uchar>(y);
		uchar* dstP = dst.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			float b0 = imgP[3 * x];
			float g0 = imgP[3 * x + 1];
			float r0 = imgP[3 * x + 2];

			float b = b0 * 255 / (g0 + r0 + 1);
			float g = g0 * 255 / (b0 + r0 + 1);
			float r = r0 * 255 / (g0 + b0 + 1);

			r = (r>255 ? 255 : (r<0 ? 0 : r));
			g = (g>255 ? 255 : (g<0 ? 0 : g));
			b = (b>255 ? 255 : (b<0 ? 0 : b));

			dstP[3 * x] = (uchar)b;
			dstP[3 * x + 1] = (uchar)g;
			dstP[3 * x + 2] = (uchar)r;
		}
	}
	dst.copyTo(image);
}

void Ice(Mat & image)
{
	Mat img;
	image.copyTo(img);
	int width = image.cols;
	int heigh = image.rows;
	Mat dst(img.size(), CV_8UC3);
	for (int y = 0; y<heigh; y++)
	{
		uchar* imgP = img.ptr<uchar>(y);
		uchar* dstP = dst.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			float b0 = imgP[3 * x];
			float g0 = imgP[3 * x + 1];
			float r0 = imgP[3 * x + 2];

			float b = (b0 - g0 - r0) * 3 / 2;
			float g = (g0 - b0 - r0) * 3 / 2;
			float r = (r0 - g0 - b0) * 3 / 2;

			r = (r>255 ? 255 : (r<0 ? -r : r));
			g = (g>255 ? 255 : (g<0 ? -g : g));
			b = (b>255 ? 255 : (b<0 ? -b : b));
			//          r = (r>255 ? 255 : (r<0? 0 : r));  
			//          g = (g>255 ? 255 : (g<0? 0 : g));  
			//          b = (b>255 ? 255 : (b<0? 0 : b));  
			dstP[3 * x] = (uchar)b;
			dstP[3 * x + 1] = (uchar)g;
			dstP[3 * x + 2] = (uchar)r;
		}
	}
	dst.copyTo(image);
}

void Colorful(Mat & image)
{
	int width = image.cols;
	int heigh = image.rows;
	Mat gray;
	Mat imgColor[12];
	Mat display(heigh * 1.5, width * 2, CV_8UC3);

	cvtColor(image, gray, CV_BGR2GRAY);
	for (int i = 0; i<12; i++)
	{
		applyColorMap(gray, imgColor[i], i);
		int x = i % 4;
		int y = i / 4;
		Mat displayROI = display(Rect(x*width / 2, y*heigh / 2, width / 2, heigh / 2));
		resize(imgColor[i], displayROI, displayROI.size());
	}
	imshow("output", display);
}

void Old(Mat & image)
{
	int width = image.cols;
	int heigh = image.rows;
	RNG rng;
	Mat img(image.size(), CV_8UC3);
	for (int y = 0; y<heigh; y++)
	{
		uchar* P0 = image.ptr<uchar>(y);
		uchar* P1 = img.ptr<uchar>(y);
		for (int x = 0; x<width; x++)
		{
			float B = P0[3 * x];
			float G = P0[3 * x + 1];
			float R = P0[3 * x + 2];
			float newB = 0.272*R + 0.534*G + 0.131*B;
			float newG = 0.349*R + 0.686*G + 0.168*B;
			float newR = 0.393*R + 0.769*G + 0.189*B;
			if (newB<0)newB = 0;
			if (newB>255)newB = 255;
			if (newG<0)newG = 0;
			if (newG>255)newG = 255;
			if (newR<0)newR = 0;
			if (newR>255)newR = 255;
			P1[3 * x] = (uchar)newB;
			P1[3 * x + 1] = (uchar)newG;
			P1[3 * x + 2] = (uchar)newR;
		}
	}
	img.copyTo(image);
}
