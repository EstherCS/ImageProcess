#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	// ��l��
	Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat sobelX = Mat::zeros(image.size(), CV_8UC1); // sobel �� Gx
	Mat sobelY = Mat::zeros(image.size(), CV_8UC1); // sobel �� Gy
	Mat sobel = Mat::zeros(image.size(), CV_8UC1);
	// ���V�Ĥ@�Ӧ�m
	uchar *P = image.data;
	uchar *PX = sobelX.data;
	uchar *PY = sobelY.data;
	int step = image.step;
	int stepXY = sobel.step;
	for (int i = 1; i<image.rows - 1; i++)
	{
		for (int j = 1; j<image.cols - 1; j++)
		{
			// �q�L pointer ���M�Ϲ��W�C�@�ӹ���
			PX[i*sobelX.step + j * (stepXY / step)] = abs(P[(i - 1)*step + j + 1] + P[i*step + j + 1] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[i*step + j - 1] * 2 - P[(i + 1)*step + j - 1]);
			PY[i*sobelX.step + j * (stepXY / step)] = abs(P[(i + 1)*step + j - 1] + P[(i + 1)*step + j] * 2 + P[(i + 1)*step + j + 1] - P[(i - 1)*step + j - 1] - P[(i - 1)*step + j] * 2 - P[(i - 1)*step + j + 1]);
		}
	}
	addWeighted(sobelX, 0.5, sobelY, 0.5, 0, sobel); // �X�� X�BY ��V   
	Mat imageSobel;
	Sobel(image, imageSobel, CV_8UC1, 0, 1); // Opencv��Sobel���
	imshow("org", image);
	imshow("Sobel", sobel);
	//imshow("realSobel", imageSobel);

	// �N�G�׽իG
	Mat bright, last, last_Y;
	cvtColor(sobel, bright, CV_GRAY2BGR);
	cvtColor(bright, last, CV_BGR2YCrCb);
	// �վ� Y (�G��)
	vector<Mat> yuvChannels;
	split(last, yuvChannels);
	last_Y = yuvChannels[0] + 50;
	// �զX�^�h
	vector<Mat> channels;
	channels.push_back(last_Y);
	channels.push_back(last_Y);
	channels.push_back(last_Y);
	merge(channels, bright);
	//imshow("bright", bright);


	Mat Desaturate, Gauss;
	// �h��  
	cvtColor(bright, Desaturate, CV_BGR2GRAY);
	// �ƻs�h�⪺�ϨñN�L�Ϧ�  
	addWeighted(Desaturate, -1, NULL, 0, 255, Gauss);
	// ��Ϧ⪺�ϰ������ҽk
	GaussianBlur(Gauss, Gauss, Size(11, 11), 0);
	// �N�C��ղH
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
			// C = MIN( A +�]A * B�^/�]255-B�^,255)
			P[x] = (uchar)min((des + (des * Gau) / (255 - Gau)), 255);
		}
	}
	imshow("Contour Drawing", output);

	waitKey();
	return 0;
}