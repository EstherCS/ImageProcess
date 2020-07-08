#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	// 初始化
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat sobelX = Mat::zeros(image.size(), CV_8UC1); // sobel 的 Gx
	Mat sobelY = Mat::zeros(image.size(), CV_8UC1); // sobel 的 Gy
	Mat sobel = Mat::zeros(image.size(), CV_8UC1);
	// 指向第一個位置
	uchar *P = image.data;
	uchar *PX = sobelX.data;
	uchar *PY = sobelY.data;
	int step = image.step;
	int stepXY = sobel.step;
	for (int i = 1; i<image.rows - 1; i++)
	{
		for (int j = 1; j<image.cols - 1; j++)
		{
			// 通過 pointer 走遍圖像上每一個像素
			PX[i*sobelX.step + j * (stepXY / step)] = abs(P[(i - 1)*step + j + 1] + P[i*step + j + 1] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[i*step + j - 1] * 2 - P[(i + 1)*step + j - 1]);
			PY[i*sobelX.step + j * (stepXY / step)] = abs(P[(i + 1)*step + j - 1] + P[(i + 1)*step + j] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[(i - 1)*step + j] * 2 - P[(i - 1)*step + j + 1]);
		}
	}
	addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel); // 合併 X、Y 方向   
	Mat imageSobel;
	Sobel(image, imageSobel, CV_8UC1, 0, 1); // Opencv的Sobel函數
	imshow("org", image);
	imshow("Sobel", sobel);
	//imshow("realSobel", imageSobel);

	// 將亮度調亮
	Mat bright, last, last_Y;
	cvtColor(sobel, bright, CV_GRAY2BGR);
	cvtColor(bright, last, CV_BGR2YCrCb);
	// 調整 Y (亮度)
	vector<Mat> yuvChannels;
	split(last, yuvChannels);
	last_Y = yuvChannels[0] + 50;
	// 組合回去
	vector<Mat> channels;
	channels.push_back(last_Y);
	channels.push_back(last_Y);
	channels.push_back(last_Y);
	merge(channels, bright);
	//imshow("bright", bright);


	Mat Desaturate, Gauss;
	// 去色  
	cvtColor(bright, Desaturate, CV_BGR2GRAY);
	// 複製去色的圖並將他反色  
	addWeighted(Desaturate, -1, NULL, 0, 255, Gauss);
	// 對反色的圖做高斯模糊
	GaussianBlur(Gauss, Gauss, Size(11, 11), 0);
	// 將顏色調淡
	Mat output(Gauss.size(), CV_8UC1);
	for (int y = 0; y < bright.rows; y++)
	{
		uchar* P0 = Desaturate.ptr<uchar>(y);
		uchar* P1 = Gauss.ptr<uchar>(y);
		uchar* P = output.ptr<uchar>(y);
		for (int x = 0; x < bright.cols; x++)
		{
			int des = P0[x];
			int Gau = P1[x];
			// C = MIN( A +（A * B）/（255-B）,255)
			P[x] = (uchar)min((des + (des * Gau) / (255 - Gau)), 255);
		}
	}
	imshow("Contour Drawing", output);

	waitKey();
	return 0;
}