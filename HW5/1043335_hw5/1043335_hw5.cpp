#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;

void DFT(Mat img)
{
	int Col = img.cols;
	int Row = img.rows;
	Mat padded;
	copyMakeBorder(img, padded, 0, Row, 0, Col, BORDER_CONSTANT, Scalar::all(0));
	//imshow("padding", padded);
	int M = padded.cols; // Col * 2 = 128
	int N = padded.rows; // Row * 2 = 128

	Mat Real = Mat::zeros(padded.size(), CV_8U);
	Mat Imag = Mat::zeros(padded.size(), CV_8U);
	Mat Spectrum = Mat::zeros(padded.size(), CV_8U);
	Mat Phase = Mat::zeros(padded.size(), CV_8U);

	// 傅立葉開始, 外圈循環計算 F(u,v), 內圈循環 f(x,y)
	for (int v = 0; v < N; ++v)
	{
		for (int u = 0; u < M; ++u)
		{
			double realSum = 0.0;
			double imagSum = 0.0; 
			for (int y = 0; y < Row; ++y)
			{
				for (int x = 0; x < Col; ++x)
				{
					// 用公式分別計算實部虛部
					realSum += (double)(pow(-1, (x + y))*padded.ptr(y)[x] * cos(2.0 * CV_PI*((double)(x)*(double)(u)*(1.0 / M) + (double)(y)*(double)(v)*(1.0 / N))));
					imagSum -= (double)(pow(-1, (x + y))*padded.ptr(y)[x] * sin(2.0 * CV_PI*((double)(x)*(double)(u)*(1.0 / M) + (double)(y)*(double)(v)*(1.0 / N))));
				}
			}
			Real.ptr(v)[u] = realSum;
			Imag.ptr(v)[u] = imagSum;

			//Spectrum.ptr(v)[u] = log((double)(sqrt(realSum * realSum + imagSum * imagSum))) * 20;
			Spectrum.ptr(v)[u] = log((double)(sqrt(realSum * realSum + imagSum * imagSum))+1) * 15;
			Phase.ptr(v)[u] = (double)(atan2(imagSum, realSum));
		}
	}
	Mat OSpectrum, OPhase;
	resize(Spectrum, OSpectrum, Size(Col, Row), 0, 0, INTER_LINEAR);
	resize(Phase, OPhase, Size(Col, Row), 0, 0, INTER_LINEAR);
	imshow("Spectrum Image", OSpectrum);
	imshow("Phase angle Image", OPhase);
}


int main()
{
	Mat inputImg = imread("mo.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	// 將圖像擴展到最佳尺寸，邊界用0補充
	Mat padded;
	// m 為大於等於 inputImg.rows 裡的最小值，且須為2、3、5的次方相乘
	int m = getOptimalDFTSize(inputImg.rows);  
	int n = getOptimalDFTSize(inputImg.cols);
	// 為了效率，所以對影像邊界拓展
	copyMakeBorder(inputImg, padded, 0, m - inputImg.rows, 0, n - inputImg.cols, BORDER_CONSTANT, Scalar::all(0)); 

	// 儲存實部、虛部
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);

	split(complexImg, planes);                  // 分離通道，planes[0]為實數部分，planes[1]為虛數部分 
	magnitude(planes[0], planes[1], planes[0]); // planes[0] = sqrt((planes[0])^2 + (planes[1])^2
	Mat output = planes[0];
	output += Scalar::all(1);                   // 進行對數尺度缩放，output = log(1+planes[0])
	log(output, output);
	// 令邊長為偶數
	output = output(Rect(0, 0, output.cols & -2, output.rows & -2));  

	// 將區塊重排，讓原點在影像的中央
	int cx = output.cols / 2;
	int cy = output.rows / 2;

	Mat q0(output, Rect(0, 0, cx, cy));    // Top-Left 
	Mat q1(output, Rect(cx, 0, cx, cy));   // Top-Right
	Mat q2(output, Rect(0, cy, cx, cy));   // Bottom-Left
	Mat q3(output, Rect(cx, cy, cx, cy));  // Bottom-Right

	Mat tmp;
	// 交換象限 ( 左上與右下 )
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	// 交換象限 ( 右上與左下 )
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(output, output, 0, 1, CV_MINMAX);

	imshow("inputImg", inputImg);
	imshow("dft", output);

	DFT(inputImg);

	waitKey();

	return 0;
}
