/*
* File:   main.cpp
* Author: sagar
*
* Created on 10 September, 2012, 7:48 PM
*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;

Mat imagen;
Mat imagen_bn;
Mat imagen_prom;
Mat imagen_gauss;
Mat imagen_mediano;
Mat imagen_laplace;
Mat imagen_sombrero;
Mat imagen_bordes;
Mat imagen_enfatizador;

//Funcion que saca blanco y negro de imagen
void blanco_negro(const Mat &sourceImage, Mat &destinationImage)
{

int prom = 0;

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols ; ++x){
            //se suman los valores r g y b de cada pixel y se promedian
            prom = 0;
			for (int i = 0; i < sourceImage.channels(); ++i){
                prom += sourceImage.at<Vec3b>(y, x)[i];
			}prom = prom/3;
			for (int i = 0; i < sourceImage.channels(); ++i){
                destinationImage.at<Vec3b>(y, x)[i] = prom;
			}
        }

}

//Funcion que saca blanco y negro de imagen
void promedio(const Mat &sourceImage, Mat &destinationImage)
{

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

blur( sourceImage, destinationImage, Size( 10, 10 ), Point(-1,-1) );

}

//Funcion que saca blanco y negro de imagen
void gauss(const Mat &sourceImage, Mat &destinationImage)
{

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

GaussianBlur(sourceImage, destinationImage, Size( 5, 5 ), 0, 0 );

}

void mediano(const Mat &sourceImage, Mat &destinationImage)
{

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

medianBlur ( sourceImage, destinationImage, 3 );

}

void laplace(const Mat &sourceImage, Mat &destinationImage)
{
Mat procesado;

int kernel_size = 3;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;


if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

if (procesado.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

Laplacian( sourceImage, procesado, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
convertScaleAbs( procesado, destinationImage );

}

void sombrero(const Mat &sourceImage, Mat &destinationImage)
{
Mat gaussiana;
Mat procesado;

int kernel_size = 3;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;


if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

if (procesado.empty())
    procesado = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

if (gaussiana.empty())
    gaussiana = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

GaussianBlur(sourceImage, gaussiana, Size( 5, 5 ), 0, 0 );

Laplacian( gaussiana, procesado, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
convertScaleAbs( procesado, destinationImage );

}

void bordes(const Mat &sourceImage, Mat &destinationImage)
{
Mat procesado;


int edgeThresh = 1;
int lowThreshold = 20;
int ratio = 3;
int kernel_size = 3;

if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

if (procesado.empty())
    procesado = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());


/// Canny detector
  Canny( sourceImage, procesado, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  destinationImage = Scalar::all(0);

  sourceImage.copyTo(destinationImage, procesado);

}

void sobel(const Mat &sourceImage, Mat &destinationImage)
{
Mat procesado;

int scale = 1;
int delta = 0;
int ddepth = CV_16S;

if (destinationImage.empty())
    destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

if (procesado.empty())
    procesado = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

 /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;

  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( sourceImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );

  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( sourceImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, destinationImage );

}

//Esta funcion toma una matriz y llama a las demás para modificarla
void tratamiento_imagen(const Mat &sourceImage)
{
blanco_negro(sourceImage,imagen_bn);
promedio(imagen_bn,imagen_prom);
gauss(imagen_bn,imagen_gauss);
mediano(imagen_bn, imagen_mediano);
laplace(imagen_bn, imagen_laplace);
sombrero(imagen_bn, imagen_sombrero);
bordes(imagen_bn,imagen_bordes);
sobel(imagen_bn, imagen_enfatizador);
}

int main() {

imagen = imread("lena.jpg");  //0 is the id of video device.0 if you have only one camera.

//unconditional loop
while (true) {

tratamiento_imagen(imagen);
imshow("Original", imagen);
imshow("imagen blanco negro", imagen_bn);
imshow("imagen promedio", imagen_prom);
imshow("imagen gauss", imagen_gauss);
imshow("imagen mediano", imagen_mediano);
imshow("imagen laplace", imagen_laplace);
imshow("imagen sombrero", imagen_sombrero);
imshow("imagen bordes", imagen_bordes);
imshow("imagen enfatizador", imagen_enfatizador);


if (waitKey(30) >= 0)
break;
}
return 0;
}

