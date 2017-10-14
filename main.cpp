/*
* File:   main.cpp
* Author: sagar
*
* Created on 10 September, 2012, 7:48 PMf
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
Mat imagen_derivador;
Mat imagen_dilatacion;
Mat imagen_erosion;
Mat imagen_apertura;
Mat imagen_cerradura;

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

blur(sourceImage, destinationImage, Size( 10, 10 ), Point(-1,-1) );

}

//Funcion que saca blanco y negro de imagen
void gauss(const Mat &sourceImage, Mat &destinationImage)
{

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

GaussianBlur(sourceImage, destinationImage, Size( 9, 9 ), 0, 0 );

}

void mediano(const Mat &sourceImage, Mat &destinationImage)
{

if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

medianBlur ( sourceImage, destinationImage, 5 );

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
Mat umbral;

if (umbral.empty())
    umbral = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

int threshold_value = 100;

/*0: Binary
1: Binary Inverted
2: Threshold Truncated
3: Threshold to Zero
4: Threshold to Zero Inverted */

int threshold_type = 0;
int max_BINARY_value = 255;

cvtColor( sourceImage, umbral, CV_BGR2GRAY );

threshold( umbral, umbral, threshold_value, max_BINARY_value,threshold_type );

laplace(umbral, destinationImage);

}

void enfatizador(const Mat &sourceImage, Mat &destinationImage)
{

Mat lapla;

if (lapla.empty())
    lapla= Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());
promedio(sourceImage,lapla);
laplace(lapla, lapla);
addWeighted( sourceImage, 0.5, lapla, 0.5, 0, destinationImage );

}


void derivador(const Mat &sourceImage, Mat &destinationImage)
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

  imshow("imagen derivador en X", abs_grad_x);
  imshow("imagen derivador en Y", abs_grad_y);

  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, destinationImage );

}

void dilatacion(const Mat &sourceImage, Mat &destinationImage)
{
  int dilation_type;
  int dilation_size = 4;
  int dilation_elem = 0;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );

  dilate( sourceImage, destinationImage, element);

}

void erosion(const Mat &sourceImage, Mat &destinationImage)
{
  int erosion_type;
  int erosion_size = 4;
  int erosion_elem = 0;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  erode( sourceImage, destinationImage, element);

}

void apertura(const Mat &sourceImage, Mat &destinationImage)
{
Mat procesado;

if (procesado.empty())
    procesado = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());


dilatacion(sourceImage, procesado);
erosion(procesado, destinationImage);

}

void cerradura(const Mat &sourceImage, Mat &destinationImage)
{
Mat procesado;

if (procesado.empty())
    procesado = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());


erosion(sourceImage, procesado);
dilatacion(procesado, destinationImage);


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
derivador(imagen_bn, imagen_derivador);
enfatizador(imagen_bn, imagen_enfatizador);
dilatacion(imagen_bn,imagen_dilatacion);
erosion(imagen_bn, imagen_erosion);
apertura(imagen_bn, imagen_apertura);
cerradura(imagen_bn, imagen_cerradura);
}

int main() {

imagen = imread("Carmina.jpeg");  //0 is the id of video device.0 if you have only one camera.

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
imshow("imagen derivador X y Y", imagen_derivador);
imshow("imagen enfatizador", imagen_enfatizador);
imshow("imagen dilatacion", imagen_dilatacion);
imshow("imagen erosion", imagen_erosion);
imshow("imagen apertura", imagen_apertura);
imshow("imagen cerradura", imagen_cerradura);

if (waitKey(30) >= 0)
break;
}
return 0;
}

