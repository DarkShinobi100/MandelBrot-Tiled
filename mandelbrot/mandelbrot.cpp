// Mandelbrot set example
// Adam Sampson <a.sampson@abertay.ac.uk>

//standard mandelbrot timed at 2819 ms
//zoomed in mandelbrot timed at 5064 ms
//both of these are with a size of 1920x1200 (widthxheight)
//function compute_mandelbrot is at least O(N)^2 as where each pixel corresponds onto the screen is calculated within a nested for loop

//for the 16 slices normal
	/*Computing the first Mandelbrot set took 6 ms.
	Computing the second Mandelbrot set took 10 ms.
	Computing the third Mandelbrot set took 48 ms.
	Computing the fourth Mandelbrot set took 102 ms.
	Computing the fifth Mandelbrot set took 218 ms.
	Computing the sixth Mandelbrot set took 269 ms.
	Computing the seventh Mandelbrot set took 356 ms.
	Computing the eighth Mandelbrot set took 418 ms.
	Computing the ninth Mandelbrot set took 397 ms.
	Computing the tenth Mandelbrot set took 326 ms.
	Computing the eleventh Mandelbrot set took 245 ms.
	Computing the twelfth Mandelbrot set took 206 ms.
	Computing the thirteenth Mandelbrot set took 98 ms.
	Computing the fourteenth Mandelbrot set took 48 ms.
	Computing the fifteenth Mandelbrot set took 11 ms.
	Computing the sixteenth Mandelbrot set took 6 ms.*/

//for the 16 slices zoomed in
	//Computing the first Mandelbrot set took 352 ms.
	//Computing the second Mandelbrot set took 336 ms.
	//Computing the third Mandelbrot set took 317 ms.
	//Computing the fourth Mandelbrot set took 301 ms.
	//Computing the fifth Mandelbrot set took 369 ms.
	//Computing the sixth Mandelbrot set took 418 ms.
	//Computing the seventh Mandelbrot set took 452 ms.
	//Computing the eighth Mandelbrot set took 399 ms.
	//Computing the ninth Mandelbrot set took 336 ms.
	//Computing the tenth Mandelbrot set took 323 ms.
	//Computing the eleventh Mandelbrot set took 266 ms.
	//Computing the twelfth Mandelbrot set took 267 ms.
	//Computing the thirteenth Mandelbrot set took 224 ms.
	//Computing the fourteenth Mandelbrot set took 197 ms.
	//Computing the fifteenth Mandelbrot set took 233 ms.
	//Computing the sixteenth Mandelbrot set took 320 ms.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <iostream>
#include "string.h"
#include <amp.h>
#include <amp_math.h>

// Import things we need from the standard library
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::complex;
using std::cout;
using std::endl;
using std::ofstream;
using namespace concurrency;

// Define the alias "the_clock" for the clock type we're going to use.
typedef std::chrono::steady_clock the_clock;


// The size of the image to generate.
const int WIDTH = 1920;
const int HEIGHT = 1200;
//const int WIDTH = 192;
//const int HEIGHT = 120;

// The number of times to iterate before we assume that a point isn't in the
// Mandelbrot set.
// (You may need to turn this up if you zoom further into the set.)
const int MAX_ITERATIONS = 500;

// The image data.
// Each pixel is represented as 0xRRGGBB.
uint32_t image[HEIGHT][WIDTH];


// Write the image to a TGA file with the given name.
// Format specification: http://www.gamers.org/dEngine/quake3/TGA.txt
void write_tga(const char *filename)
{
	ofstream outfile(filename, ofstream::binary);

	uint8_t header[18] = {
		0, // no image ID
		0, // no colour map
		2, // uncompressed 24-bit image
		0, 0, 0, 0, 0, // empty colour map specification
		0, 0, // X origin
		0, 0, // Y origin
		WIDTH & 0xFF, (WIDTH >> 8) & 0xFF, // width
		HEIGHT & 0xFF, (HEIGHT >> 8) & 0xFF, // height
		24, // bits per pixel
		0, // image descriptor
	};
	outfile.write((const char *)header, 18);

	for (int y = 0; y < HEIGHT; ++y)
	{
		for (int x = 0; x < WIDTH; ++x)
		{
			uint8_t pixel[3] = {
				image[y][x] & 0xFF, // blue channel
				(image[y][x] >> 8) & 0xFF, // green channel
				(image[y][x] >> 16) & 0xFF, // red channel
			};
			outfile.write((const char *)pixel, 3);
		}
	}

	outfile.close();
	if (!outfile)
	{
		// An error has occurred at some point since we opened the file.
		cout << "Error writing to " << filename << endl;
		exit(1);
	}
}

unsigned long makeColour(int r, int g, int b)
{
	return ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);
}

// using our own Complex number structure and definitions as the Complex type is not available in the Concurrency namespace

struct Complex1 {

	float x;

	float y;

};


Complex1 c_add(Complex1 c1, Complex1 c2) restrict(cpu, amp) // restrict keyword - able to execute this function on the GPU and CPU

{

	Complex1 tmp;

	float a = c1.x;

	float b = c1.y;

	float c = c2.x;

	float d = c2.y;

	tmp.x = a + c;

	tmp.y = b + d;

	return tmp;

} // c_add


float c_abs(Complex1 c) restrict(cpu, amp)

{

	return concurrency::fast_math::sqrt(c.x * c.x + c.y * c.y);

} // c_abs


Complex1 c_mul(Complex1 c1, Complex1 c2) restrict(cpu, amp)

{

	Complex1 tmp;

	float a = c1.x;

	float b = c1.y;

	float c = c2.x;

	float d = c2.y;

	tmp.x = a * c - b * d;

	tmp.y = b * c + a * d;

	return tmp;

} // c_mul


// Render the Mandelbrot set into the image array.
// The parameters specify the region on the complex plane to plot.
void compute_mandelbrot(double left, double right, double top, double bottom, int ymin, int ymax)
{
	uint32_t* pImage = &(image[0][0]);
	array_view<uint32_t, 2> a(HEIGHT, WIDTH, pImage); //this is called "a" for 'reasons *head nod* ;) aubergene(eggplant) *water dropplets sfx*(;' 
	a.discard_data();
	   
	parallel_for_each(a.extent, [=](concurrency::index<2> idx)
		restrict(amp) {
			//compute madelbrot here i.e madelbrot kernel/shader
				//USE THREAD ID/INDEX TO MAP INTO THE COMPLEX PLANE
			int h = idx[0];
			int w = idx[1];
			Complex1 c;

			// Work out the point in the complex plane that
			// corresponds to this pixel in the output image.
			c.x = left + (h * (right - left) / WIDTH);
			c.y = top + (w * (bottom - top) / HEIGHT);

			// Start off z at (0, 0)
			Complex1 z;
			z.x = 0.0f;
			z.y = 0.0f;

			// Iterate z = z^2 + c until z moves more than 2 units
			// away from (0, 0), or we've iterated too many times.
			int iterations = 0;
			while (c_abs(z) < 2.0 && iterations < MAX_ITERATIONS)
			{
				z.x = (z.x * z.x) + c.x;
				z.y = (z.y * z.y) + c.y;

				++iterations;
			}

			if (iterations == MAX_ITERATIONS)
			{
				// z didn't escape from the circle.
				// This point is in the Mandelbrot set.
				a[h][w] = 0x000000; // black
			}
			else
			{
				// z escaped within less than MAX_ITERATIONS
				// iterations. This point isn't in the set.
				a[h][w] = 0xFFFFFF; // white
				//image[y][x] = makeColour(0xAF, 0xEC, 0xDB);

			}
		});


	//for (int y = ymin; y < ymax; ++y)
	//{
	//	for (int x = 0; x < WIDTH; ++x)
	//	{
	//		// Work out the point in the complex plane that
	//		// corresponds to this pixel in the output image.
	//		complex<double> c(left + (x * (right - left) / WIDTH),
	//			top + (y * (bottom - top) / HEIGHT));

	//		// Start off z at (0, 0).
	//		complex<double> z(0.0, 0.0);

	//		// Iterate z = z^2 + c until z moves more than 2 units
	//		// away from (0, 0), or we've iterated too many times.
	//		int iterations = 0;
	//		while (abs(z) < 2.0 && iterations < MAX_ITERATIONS)
	//		{
	//			z = (z * z) + c;

	//			++iterations;
	//		}

	//		if (iterations == MAX_ITERATIONS)
	//		{
	//			// z didn't escape from the circle.
	//			// This point is in the Mandelbrot set.
	//			//image[y][x] = 0x000000; // black
	//			image[y][x] = makeColour(0xFA, 0x09, 0xCA);
	//		}
	//		else
	//		{
	//			// z escaped within less than MAX_ITERATIONS
	//			// iterations. This point isn't in the set.
	//			//image[y][x] = 0xFFFFFF; // white
	//			//image[y][x] = makeColour(0xAF, 0xEC, 0xDB);
	//			if (iterations >= 0 && iterations <= 100)
	//			{
	//				image[y][x] = makeColour(0x00, 0x00, 0xFE);
	//			}
	//			else if (iterations >= 101 && iterations <= 200)
	//			{
	//				image[y][x] = makeColour(0x00, 0x00, 0xEB);
	//			}
	//			else if (iterations >= 201 && iterations <= 450)
	//			{
	//				image[y][x] = makeColour(0x00, 0x00, 0xDA);
	//			}
	//			else if (iterations >= 451 && iterations <= 499)
	//			{
	//				image[y][x] = makeColour(0x00, 0x00, 0xCC);
	//			}
	//			else if (iterations >= 499 && iterations <= 500)
	//			{
	//				image[y][x] = makeColour(0x00, 0x00, 0xBC);
	//			}


	//		}
	//	
	//
	//	}
	//}
}




int main(int argc, char *argv[])
{
	cout << "Please wait..." << endl;

	// Start timing
	//the_clock::time_point start = the_clock::now();

	// This shows the whole set.
	//compute_mandelbrot(-2.0, 1.0, 1.125, -1.125, (HEIGHT / 16) * multiply, (HEIGHT / 16) * (multiply+1));

	for (int multiply = 0; multiply < 16; multiply++)
	{
		// This shows the whole set.
		the_clock::time_point start = the_clock::now();
		//compute_mandelbrot(-2.0, 1.0, 1.125, -1.125, (HEIGHT / 16) * multiply, (HEIGHT / 16) * (multiply+1));
		compute_mandelbrot(-0.751085, -0.734975, 0.118378, 0.134488, (HEIGHT / 16) * multiply, (HEIGHT / 16) * (multiply + 1));
		the_clock::time_point end = the_clock::now();
		// Compute the difference between the two times in milliseconds
		auto time_taken = duration_cast<milliseconds>(end - start).count();
		cout << "Computing the Mandelbrot set took " << time_taken << " ms." << endl;
	}

	// This zooms in on an interesting bit of detail.
	//compute_mandelbrot(-0.751085, -0.734975, 0.118378, 0.134488);

	// Stop timing
	//the_clock::time_point end = the_clock::now();

	// Compute the difference between the two times in milliseconds
	//auto time_taken = duration_cast<milliseconds>(end - start).count();
	//cout << "Computing the Mandelbrot set took " << time_taken << " ms." << endl;

	write_tga("output.tga");

	return 0;
}
