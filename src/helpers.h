/*----------------------------------------------------------------------------

  LSD - Line Segment Detector on digital images

  This code is part of the following publication and was subject
  to peer review:

    "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
    Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
    Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
    http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd

  Copyright (c) 2007-2011 rafael grompone von gioi <grompone@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  ----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** @file lsd.cpp
    LSD module code
    @author rafael grompone von gioi <grompone@gmail.com>
 */
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** @mainpage LSD code documentation

    This is an implementation of the Line Segment Detector described
    in the paper:

      "LSD: A Fast Line Segment Detector with a False Detection Control"
      by Rafael Grompone von Gioi, Jeremie Jakubowicz, Jean-Michel Morel,
      and Gregory Randall, IEEE Transactions on Pattern Analysis and
      Machine Intelligence, vol. 32, no. 4, pp. 722-732, April, 2010.

    and in more details in the CMLA Technical Report:

      "LSD: A Line Segment Detector, Technical Report",
      by Rafael Grompone von Gioi, Jeremie Jakubowicz, Jean-Michel Morel,
      Gregory Randall, CMLA, ENS Cachan, 2010.

    The version implemented here includes some further improvements
    described in the following publication, of which this code is part:

      "LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
      Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
      Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
      http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd

    The module's main function is lsd().

    The source code is contained in two files: lsd.h and lsd.c.

    HISTORY:
    - version 1.6 - nov 2011:
                              - changes in the interface,
                              - max_grad parameter removed,
                              - the factor 11 was added to the number of test
                                to consider the different precision values
                                tested,
                              - a minor bug corrected in the gradient sorting
                                code,
                              - the algorithm now also returns p and log_nfa
                                for each detection,
                              - a minor bug was corrected in the image scaling,
                              - the angle comparison in "isaligned" changed
                                from < to <=,
                              - "eps" variable renamed "log_eps",
                              - "lsd_scale_region" interface was added,
                              - minor changes to comments.
    - version 1.5 - dec 2010: Changes in 'refine', -W option added,
                              and more comments added.
    - version 1.4 - jul 2010: lsd_scale interface added and doxygen doc.
    - version 1.3 - feb 2010: Multiple bug correction and improved code.
    - version 1.2 - dec 2009: First full Ansi C Language version.
    - version 1.1 - sep 2009: Systematic subsampling to scale 0.8 and
                              correction to partially handle "angle problem".
    - version 1.0 - jan 2009: First complete Megawave2 and Ansi C Language
                              version.

    @author rafael grompone von gioi <grompone@gmail.com>
 */
/*----------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>
#include <iostream>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <functional>
#include <future>

#include "lsd.h"

/** ln(10) */
#ifndef M_LN10
#define M_LN10 2.30258509299404568402
#endif /* !M_LN10 */

/** PI */
#ifndef M_PI
#define M_PI   3.14159265358979323846
#endif /* !M_PI */

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

/** Label for pixels with undefined gradient. */
#define NOTDEF -1024.0

/** 3/2 pi */
#define M_3_2_PI 4.71238898038

/** 2 pi */
#define M_2__PI  6.28318530718

/** Label for pixels not used in yet. */
#define NOTUSED 0

/** Label for pixels already used in detection. */
#define USED    1

double UPM_GRADIENT_THRESHOLD_LSD = 5.2262518595055063;

/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist {
  int x, y;
  struct coorlist *next;
};

/*----------------------------------------------------------------------------*/
/** A point (or pixel).
 */
struct point { int x, y; };


/*----------------------------------------------------------------------------*/
/*------------------------- Miscellaneous functions --------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
static void error(const char *msg) {
  fprintf(stderr, "LSD Error: %s\n", msg);
  exit(EXIT_FAILURE);
}

/*----------------------------------------------------------------------------*/
/** Doubles relative error factor
 */
#define RELATIVE_ERROR_FACTOR 100.0

/*----------------------------------------------------------------------------*/
/** Compare doubles by relative error.

    The resulting rounding error after floating point computations
    depend on the specific operations done. The same number computed by
    different algorithms could present different rounding errors. For a
    useful comparison, an estimation of the relative rounding error
    should be considered and compared to a factor times EPS. The factor
    should be related to the cumulated rounding error in the chain of
    computation. Here, as a simplification, a fixed factor is used.
 */
static int double_equal(double a, double b) {
  double abs_diff, aa, bb, abs_max;

  /* trivial case */
  if (a == b) return TRUE;

  abs_diff = fabs(a - b);
  aa = fabs(a);
  bb = fabs(b);
  abs_max = aa > bb ? aa : bb;

  /* DBL_MIN is the smallest normalized number, thus, the smallest
     number whose relative error is bounded by DBL_EPSILON. For
     smaller numbers, the same quantization steps as for DBL_MIN
     are used. Then, for smaller numbers, a meaningful "relative"
     error should be computed by dividing the difference by DBL_MIN. */
  if (abs_max < DBL_MIN) abs_max = DBL_MIN;

  /* equal if relative error <= factor x eps */
  return (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON);
}

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */
static double dist(double x1, double y1, double x2, double y2) {
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/*----------------------------------------------------------------------------*/
/*----------------------------- Image Data Types -----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** char image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_char_s {
  unsigned char *data;
  unsigned int xsize, ysize;
} *image_char;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_char 'i'.
 */
static void free_image_char(image_char i) {
  if (i == nullptr || i->data == nullptr)
    error("free_image_char: invalid input image.");
  free((void *) i->data);
  free((void *) i);
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize'.
 */
static image_char new_image_char(unsigned int xsize, unsigned int ysize) {
  image_char image;

  /* check parameters */
  if (xsize == 0 || ysize == 0) error("new_image_char: invalid image size.");

  /* get memory */
  image = (image_char) malloc(sizeof(struct image_char_s));
  if (image == nullptr) error("not enough memory.");
  image->data = (unsigned char *) calloc(xsize * ysize,
                                         sizeof(unsigned char));
  if (image->data == nullptr) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_char of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_char new_image_char_ini(unsigned int xsize, unsigned int ysize,
                                     unsigned char fill_value) {
  image_char image = new_image_char(xsize, ysize); /* create image */
  unsigned int N = xsize * ysize;
  unsigned int i;

  /* check parameters */
  if (image == nullptr || image->data == nullptr)
    error("new_image_char_ini: invalid image.");

  /* initialize */
  for (i = 0; i < N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** int image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_int_s {
  int *data;
  unsigned int xsize, ysize;
} *image_int;

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize'.
 */
static image_int new_image_int(unsigned int xsize, unsigned int ysize) {
  image_int image;

  /* check parameters */
  if (xsize == 0 || ysize == 0) error("new_image_int: invalid image size.");

  /* get memory */
  image = (image_int) malloc(sizeof(struct image_int_s));
  if (image == nullptr) error("not enough memory.");
  image->data = (int *) calloc(xsize * ysize, sizeof(int));
  if (image->data == nullptr) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_int of size 'xsize' times 'ysize',
    initialized to the value 'fill_value'.
 */
static image_int new_image_int_ini(unsigned int xsize, unsigned int ysize,
                                   int fill_value) {
  image_int image = new_image_int(xsize, ysize); /* create image */
  unsigned int N = xsize * ysize;
  unsigned int i;

  /* initialize */
  for (i = 0; i < N; i++) image->data[i] = fill_value;

  return image;
}

/*----------------------------------------------------------------------------*/
/** double image data type

    The pixel value at (x,y) is accessed by:

      image->data[ x + y * image->xsize ]

    with x and y integer.
 */
typedef struct image_double_s {
  double *data;
  unsigned int xsize, ysize;
} *image_double;

/*----------------------------------------------------------------------------*/
/** Free memory used in image_double 'i'.
 */
static void free_image_double(image_double i) {
  if (i == nullptr || i->data == nullptr)
    error("free_image_double: invalid input image.");
  free((void *) i->data);
  free((void *) i);
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'.
 */
static image_double new_image_double(unsigned int xsize, unsigned int ysize) {
  image_double image;

  /* check parameters */
  if (xsize == 0 || ysize == 0) error("new_image_double: invalid image size.");

  /* get memory */
  image = (image_double) malloc(sizeof(struct image_double_s));
  if (image == nullptr) error("not enough memory.");
  image->data = (double *) calloc(xsize * ysize, sizeof(double));
  if (image->data == nullptr) error("not enough memory.");

  /* set image size */
  image->xsize = xsize;
  image->ysize = ysize;

  return image;
}

/*----------------------------------------------------------------------------*/
/** Create a new image_double of size 'xsize' times 'ysize'
    with the data pointed by 'data'.
 */
static image_double new_image_double_ptr(unsigned int xsize,
                                         unsigned int ysize, double *data) {
  image_double image;

  /* check parameters */
  if (xsize == 0 || ysize == 0)
    error("new_image_double_ptr: invalid image size.");
  if (data == nullptr) error("new_image_double_ptr: nullptr data pointer.");

  /* get memory */
  image = (image_double) malloc(sizeof(struct image_double_s));
  if (image == nullptr) error("not enough memory.");

  /* set image */
  image->xsize = xsize;
  image->ysize = ysize;
  image->data = data;

  return image;
}

/*----------------------------------------------------------------------------*/
/*--------------------------------- Gradient ---------------------------------*/
/*----------------------------------------------------------------------------*/

/**
 * Computes the gradient magnitude and orientation of an image.
 * @param in The input image
 * @param threshold The gradient threshold used to avoid some values in the gradient orientation map
 * @param g The gradient orientation map
 * @param modgrad the module of the gradient
 */
static void grad_angle_orientation(image_double in, double threshold, image_double& g, image_double& modgrad){
  unsigned int n, p, x, y, adr;
  double com1, com2, gx, gy, norm, norm2;
  n = in->ysize;
  p = in->xsize;

  /* allocate output image */
  g = new_image_double(in->xsize, in->ysize);

  /* get memory for the image of gradient modulus */
  modgrad = new_image_double(in->xsize, in->ysize);

  /* 'undefined' on the up and left boundaries */
  for (x = 0; x < p; x++) g->data[x] = NOTDEF;
  for (y = 0; y < n; y++) g->data[p * y] = NOTDEF;

  /* compute gradient on the remaining pixels */
  for (x = 1; x < p; x++)
    for (y = 1; y < n; y++) {
      adr = y * p + x;

      /*
         Norm 2 computation using 2x2 pixel window:
           A B
           C D
         and
           com1 = D-A,  com2 = B-C.
         Then
           gx = B+D - (A+C)   horizontal difference
           gy = C+D - (A+B)   vertical difference
         com1 and com2 are just to avoid 2 additions.
       */
      com1 = in->data[adr] - in->data[adr - p - 1];
      com2 = in->data[adr - p] - in->data[adr - 1];

      gx = com1 + com2; /* gradient x component */
      gy = com1 - com2; /* gradient y component */
      norm2 = gx * gx + gy * gy;
      norm = sqrt(norm2 / 4.0); /* gradient norm */

      modgrad->data[adr] = norm; /* store gradient norm */

      if (norm <= threshold) /* norm too small, gradient no defined */
        g->data[adr] = NOTDEF; /* gradient angle not defined */
      else {
        /* gradient angle computation */
        //g->data[adr] = atan2(gx, -gy);
        g->data[adr] = atan2(-gx, gy);
      }
    }
}

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point.

    The result is:
    - an image_double with the angle at each pixel, or NOTDEF if not defined.
    - the image_double 'modgrad' with the gradient magnitude at each point.
    - a list of pixels 'list_p' roughly ordered by decreasing
      gradient magnitude. (The order is made by classifying points
      into bins by gradient magnitude. The parameters 'n_bins' and
      'max_grad' specify the number of bins and the gradient modulus
      at the highest bin. The pixels in the list would be in
      decreasing gradient magnitude, up to a precision of the size of
      the bins.)
    - a pointer 'mem_p' to the memory used by 'list_p' to be able to
      free the memory when it is not used anymore.
 */
static image_double ll_angle(image_double in, double threshold,
                             struct coorlist **list_p, void **mem_p,
                             image_double& modgrad, image_double& g, unsigned int n_bins) {
  unsigned int n, p, x, y, i;
  double norm;
  /* the rest of the variables are used for pseudo-ordering
     the gradient magnitude values */
  int list_count = 0;
  struct coorlist *list;
  struct coorlist **range_l_s; /* array of pointers to start of bin list */
  struct coorlist **range_l_e; /* array of pointers to end of bin list */
  struct coorlist *start;
  struct coorlist *end;
  double max_grad = 0.0;

  /* check parameters */
  if (in == nullptr || in->data == nullptr || in->xsize == 0 || in->ysize == 0)
    error("ll_angle: invalid image.");
  if (threshold < 0.0) error("ll_angle: 'threshold' must be positive.");
  if (list_p == nullptr) error("ll_angle: nullptr pointer 'list_p'.");
  if (mem_p == nullptr) error("ll_angle: nullptr pointer 'mem_p'.");
  if (n_bins == 0) error("ll_angle: 'n_bins' must be positive.");

  /* image size shortcuts */
  n = in->ysize;
  p = in->xsize;

  /* get memory for "ordered" list of pixels */
  list = (struct coorlist *) calloc(n * p, sizeof(struct coorlist));
  *mem_p = (void *) list;
  range_l_s = (struct coorlist **) calloc((size_t) n_bins,
                                          sizeof(struct coorlist *));
  range_l_e = (struct coorlist **) calloc((size_t) n_bins,
                                          sizeof(struct coorlist *));
  if (list == nullptr || range_l_s == nullptr || range_l_e == nullptr)
    error("not enough memory.");
  for (i = 0; i < n_bins; i++) range_l_s[i] = range_l_e[i] = nullptr;

  if (modgrad == nullptr || g == nullptr) {
    grad_angle_orientation(in, threshold, g, modgrad);
  }

  for (i = 0; i < modgrad->xsize * modgrad->ysize; i++) {
    if (modgrad->data[i] > max_grad) max_grad = modgrad->data[i];
  }

  /* compute histogram of gradient values */
  for (x = 0; x < p - 1; x++)
    for (y = 0; y < n - 1; y++) {
      norm = modgrad->data[y * p + x];

      /* store the point in the right bin according to its norm */
      i = (unsigned int) (norm * (double) n_bins / max_grad);
      if (i >= n_bins) i = n_bins - 1;
      if (range_l_e[i] == nullptr)
        range_l_s[i] = range_l_e[i] = list + list_count++;
      else {
        range_l_e[i]->next = list + list_count;
        range_l_e[i] = list + list_count++;
      }
      range_l_e[i]->x = (int) x;
      range_l_e[i]->y = (int) y;
      range_l_e[i]->next = nullptr;
    }

  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  for (i = n_bins - 1; i > 0 && range_l_s[i] == nullptr; i--);
  start = range_l_s[i];
  end = range_l_e[i];
  if (start != nullptr)
    while (i > 0) {
      --i;
      if (range_l_s[i] != nullptr) {
        end->next = range_l_s[i];
        end = range_l_e[i];
      }
    }
  *list_p = start;

  /* free memory */
  free((void *) range_l_s);
  free((void *) range_l_e);

  return g;
}

/*----------------------------------------------------------------------------*/
/** Is point (x,y) aligned to angle theta, up to precision 'prec'?
 */
static int isaligned(int x, int y, image_double angles, double theta,
                     double prec) {
  double a;

  /* check parameters */
  if (angles == nullptr || angles->data == nullptr)
    error("isaligned: invalid image 'angles'.");
  if (x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize)
    error("isaligned: (x,y) out of the image.");
  if (prec < 0.0) error("isaligned: 'prec' must be positive.");

  /* angle at pixel (x,y) */
  a = angles->data[x + y * angles->xsize];

  /* pixels whose level-line angle is not defined
     are considered as NON-aligned */
  if (a == NOTDEF) return FALSE;  /* there is no need to call the function
                                      'double_equal' here because there is
                                      no risk of problems related to the
                                      comparison doubles, we are only
                                      interested in the exact NOTDEF value */

  /* it is assumed that 'theta' and 'a' are in the range [-pi,pi] */
  theta -= a;
  if (theta < 0.0) theta = -theta;
  if (theta > M_3_2_PI) {
    theta -= M_2__PI;
    if (theta < 0.0) theta = -theta;
  }

  return theta <= prec;
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
static double angle_diff(double a, double b) {
  a -= b;
  while (a <= -M_PI) a += M_2__PI;
  while (a > M_PI) a -= M_2__PI;
  if (a < 0.0) a = -a;
  return a;
}

/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
static double angle_diff_signed(double a, double b) {
  a -= b;
  while (a <= -M_PI) a += M_2__PI;
  while (a > M_PI) a -= M_2__PI;
  return a;
}


/*----------------------------------------------------------------------------*/
/*----------------------------- NFA computation ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using the Lanczos approximation.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
      \Gamma(x) = \frac{ \sum_{n=0}^{N} q_n x^n }{ \Pi_{n=0}^{N} (x+n) }
                  (x+5.5)^{x+0.5} e^{-(x+5.5)}
    @f]
    so
    @f[
      \log\Gamma(x) = \log\left( \sum_{n=0}^{N} q_n x^n \right)
                      + (x+0.5) \log(x+5.5) - (x+5.5) - \sum_{n=0}^{N} \log(x+n)
    @f]
    and
      q0 = 75122.6331530,
      q1 = 80916.6278952,
      q2 = 36308.2951477,
      q3 = 8687.24529705,
      q4 = 1168.92649479,
      q5 = 83.8676043424,
      q6 = 2.50662827511.
 */
static double log_gamma_lanczos(double x) {
  static double q[7] = {75122.6331530, 80916.6278952, 36308.2951477,
                        8687.24529705, 1168.92649479, 83.8676043424,
                        2.50662827511};
  double a = (x + 0.5) * log(x + 5.5) - (x + 5.5);
  double b = 0.0;
  int n;

  for (n = 0; n < 7; n++) {
    a -= log(x + (double) n);
    b += q[n] * pow(x, (double) n);
  }
  return a + log(b);
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x using Windschitl method.
    See http://www.rskey.org/gamma.htm

    The formula used is
    @f[
        \Gamma(x) = \sqrt{\frac{2\pi}{x}} \left( \frac{x}{e}
                    \sqrt{ x\sinh(1/x) + \frac{1}{810x^6} } \right)^x
    @f]
    so
    @f[
        \log\Gamma(x) = 0.5\log(2\pi) + (x-0.5)\log(x) - x
                      + 0.5x\log\left( x\sinh(1/x) + \frac{1}{810x^6} \right).
    @f]
    This formula is a good approximation when x > 15.
 */
static double log_gamma_windschitl(double x) {
  return 0.918938533204673 + (x - 0.5) * log(x) - x
      + 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x. When x>15 use log_gamma_windschitl(),
    otherwise use log_gamma_lanczos().
 */
#define log_gamma(x) ((x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x))

/*----------------------------------------------------------------------------*/
/** Size of the table to store already computed inverse values.
 */
#define TABSIZE 100000

/*----------------------------------------------------------------------------*/
/** Computes -log10(NFA).

    NFA stands for Number of False Alarms:
    @f[
        \mathrm{NFA} = NT \cdot B(n,k,p)
    @f]

    - NT       - number of tests
    - B(n,k,p) - tail of binomial distribution with parameters n,k and p:
    @f[
        B(n,k,p) = \sum_{j=k}^n
                   \left(\begin{array}{c}n\\j\end{array}\right)
                   p^{j} (1-p)^{n-j}
    @f]

    The value -log10(NFA) is equivalent but more intuitive than NFA:
    - -1 corresponds to 10 mean false alarms
    -  0 corresponds to 1 mean false alarm
    -  1 corresponds to 0.1 mean false alarms
    -  2 corresponds to 0.01 mean false alarms
    -  ...

    Used this way, the bigger the value, better the detection,
    and a logarithmic scale is used.

    @param n,k,p binomial parameters.
    @param logNT logarithm of Number of Tests

    The computation is based in the gamma function by the following
    relation:
    @f[
        \left(\begin{array}{c}n\\k\end{array}\right)
        = \frac{ \Gamma(n+1) }{ \Gamma(k+1) \cdot \Gamma(n-k+1) }.
    @f]
    We use efficient algorithms to compute the logarithm of
    the gamma function.

    To make the computation faster, not all the sum is computed, part
    of the terms are neglected based on a bound to the error obtained
    (an error of 10% in the result is accepted).
 */
static double nfa(int n, int k, double p, double logNT) {
  static double inv[TABSIZE];   /* table to keep computed inverse values */
  double tolerance = 0.1;       /* an error of 10% in the result is accepted */
  double log1term, term, bin_term, mult_term, bin_tail, err, p_term;
  int i;

  /* check parameters */
  if (n < 0 || k < 0 || k > n || p <= 0.0 || p >= 1.0)
    error("nfa: wrong n, k or p values.");

  /* trivial cases */
  if (n == 0 || k == 0) return -logNT;
  if (n == k) return -logNT - (double) n * log10(p);

  /* probability term */
  p_term = p / (1.0 - p);

  /* compute the first term of the series */
  /*
     binomial_tail(n,k,p) = sum_{i=k}^n bincoef(n,i) * p^i * (1-p)^{n-i}
     where bincoef(n,i) are the binomial coefficients.
     But
       bincoef(n,k) = gamma(n+1) / ( gamma(k+1) * gamma(n-k+1) ).
     We use this to compute the first term. Actually the log of it.
   */
  log1term = log_gamma((double) n + 1.0) - log_gamma((double) k + 1.0)
      - log_gamma((double) (n - k) + 1.0)
      + (double) k * log(p) + (double) (n - k) * log(1.0 - p);
  term = exp(log1term);

  /* in some cases no more computations are needed */
  if (double_equal(term, 0.0))              /* the first term is almost zero */
  {
    if ((double) k > (double) n * p)     /* at begin or end of the tail?  */
      return -log1term / M_LN10 - logNT;  /* end: use just the first term  */
    else
      return -logNT;                      /* begin: the tail is roughly 1  */
  }

  /* compute more terms if needed */
  bin_tail = term;
  for (i = k + 1; i <= n; i++) {
    /*
       As
         term_i = bincoef(n,i) * p^i * (1-p)^(n-i)
       and
         bincoef(n,i)/bincoef(n,i-1) = n-1+1 / i,
       then,
         term_i / term_i-1 = (n-i+1)/i * p/(1-p)
       and
         term_i = term_i-1 * (n-i+1)/i * p/(1-p).
       1/i is stored in a table as they are computed,
       because divisions are expensive.
       p/(1-p) is computed only once and stored in 'p_term'.
     */
    bin_term = (double) (n - i + 1) * (i < TABSIZE ?
                                       (inv[i] != 0.0 ? inv[i] : (inv[i] = 1.0 / (double) i)) :
                                       1.0 / (double) i);

    mult_term = bin_term * p_term;
    term *= mult_term;
    bin_tail += term;
    if (bin_term < 1.0) {
      /* When bin_term<1 then mult_term_j<mult_term_i for j>i.
         Then, the error on the binomial tail when truncated at
         the i term can be bounded by a geometric series of form
         term_i * sum mult_term_i^j.                            */
      err = term * ((1.0 - pow(mult_term, (double) (n - i + 1))) /
          (1.0 - mult_term) - 1.0);

      /* One wants an error at most of tolerance*final_result, or:
         tolerance * abs(-log10(bin_tail)-logNT).
         Now, the error that can be accepted on bin_tail is
         given by tolerance*final_result divided by the derivative
         of -log10(x) when x=bin_tail. that is:
         tolerance * abs(-log10(bin_tail)-logNT) / (1/bin_tail)
         Finally, we truncate the tail if the error is less than:
         tolerance * abs(-log10(bin_tail)-logNT) * bin_tail        */
      if (err < tolerance * fabs(-log10(bin_tail) - logNT) * bin_tail) break;
    }
  }
  return -log10(bin_tail) - logNT;
}


/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangle structure ----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct rect {
  double x1, y1, x2, y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double x, y;          /* center of the rectangle */
  double theta;        /* angle */
  double dx, dy;        /* (dx,dy) is vector oriented as the line segment */
  double prec;         /* tolerance angle */
  double p;            /* probability of a point with angle within 'prec' */
};

/*----------------------------------------------------------------------------*/
/** Copy one rectangle structure to another.
 */
static void rect_copy(struct rect *in, struct rect *out) {
  /* check parameters */
  if (in == nullptr || out == nullptr) error("rect_copy: invalid 'in' or 'out'.");

  /* copy values */
  out->x1 = in->x1;
  out->y1 = in->y1;
  out->x2 = in->x2;
  out->y2 = in->y2;
  out->width = in->width;
  out->x = in->x;
  out->y = in->y;
  out->theta = in->theta;
  out->dx = in->dx;
  out->dy = in->dy;
  out->prec = in->prec;
  out->p = in->p;
}

/*----------------------------------------------------------------------------*/
/** Rectangle points iterator.

    The integer coordinates of pixels inside a rectangle are
    iteratively explored. This structure keep track of the process and
    functions ri_ini(), ri_inc(), ri_end(), and ri_del() are used in
    the process. An example of how to use the iterator is as follows:
    \code

      struct rect * rec = XXX; // some rectangle
      rect_iter * i;
      for( i=ri_ini(rec); !ri_end(i); ri_inc(i) )
        {
          // your code, using 'i->x' and 'i->y' as coordinates
        }
      ri_del(i); // delete iterator

    \endcode
    The pixels are explored 'column' by 'column', where we call
    'column' a set of pixels with the same x value that are inside the
    rectangle. The following is an schematic representation of a
    rectangle, the 'column' being explored is marked by colons, and
    the current pixel being explored is 'x,y'.
    \verbatim

              vx[1],vy[1]
                 *   *
                *       *
               *           *
              *               ye
             *                :  *
        vx[0],vy[0]           :     *
               *              :        *
                  *          x,y          *
                     *        :              *
                        *     :            vx[2],vy[2]
                           *  :                *
        y                     ys              *
        ^                        *           *
        |                           *       *
        |                              *   *
        +---> x                      vx[3],vy[3]

    \endverbatim
    The first 'column' to be explored is the one with the smaller x
    value. Each 'column' is explored starting from the pixel of the
    'column' (inside the rectangle) with the smallest y value.

    The four corners of the rectangle are stored in order that rotates
    around the corners at the arrays 'vx[]' and 'vy[]'. The first
    point is always the one with smaller x value.

    'x' and 'y' are the coordinates of the pixel being explored. 'ys'
    and 'ye' are the start and end values of the current column being
    explored. So, 'ys' < 'ye'.
 */
typedef struct {
  double vx[4];  /* rectangle's corner X coordinates in circular order */
  double vy[4];  /* rectangle's corner Y coordinates in circular order */
  double ys, ye;  /* start and end Y values of current 'column' */
  int x, y;       /* coordinates of currently explored pixel */
} rect_iter;

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the smaller
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
static double inter_low(double x, double x1, double y1, double x2, double y2) {
  /* check parameters */
  if (x1 > x2 || x < x1 || x > x2)
    error("inter_low: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if (double_equal(x1, x2) && y1 < y2) return y1;
  if (double_equal(x1, x2) && y1 > y2) return y2;
  return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

/*----------------------------------------------------------------------------*/
/** Interpolate y value corresponding to 'x' value given, in
    the line 'x1,y1' to 'x2,y2'; if 'x1=x2' return the larger
    of 'y1' and 'y2'.

    The following restrictions are required:
    - x1 <= x2
    - x1 <= x
    - x  <= x2
 */
static double inter_hi(double x, double x1, double y1, double x2, double y2) {
  /* check parameters */
  if (x1 > x2 || x < x1 || x > x2)
    error("inter_hi: unsuitable input, 'x1>x2' or 'x<x1' or 'x>x2'.");

  /* interpolation */
  if (double_equal(x1, x2) && y1 < y2) return y2;
  if (double_equal(x1, x2) && y1 > y2) return y1;
  return y1 + (x - x1) * (y2 - y1) / (x2 - x1);
}

/*----------------------------------------------------------------------------*/
/** Free memory used by a rectangle iterator.
 */
static void ri_del(rect_iter *iter) {
  if (iter == nullptr) error("ri_del: nullptr iterator.");
  free((void *) iter);
}

/*----------------------------------------------------------------------------*/
/** Check if the iterator finished the full iteration.

    See details in \ref rect_iter
 */
static int ri_end(rect_iter *i) {
  /* check input */
  if (i == nullptr) error("ri_end: nullptr iterator.");

  /* if the current x value is larger than the largest
     x value in the rectangle (vx[2]), we know the full
     exploration of the rectangle is finished. */
  return (double) (i->x) > i->vx[2];
}

/*----------------------------------------------------------------------------*/
/** Increment a rectangle iterator.

    See details in \ref rect_iter
 */
static void ri_inc(rect_iter *i) {
  /* check input */
  if (i == nullptr) error("ri_inc: nullptr iterator.");

  /* if not at end of exploration,
     increase y value for next pixel in the 'column' */
  if (!ri_end(i)) i->y++;

  /* if the end of the current 'column' is reached,
     and it is not the end of exploration,
     advance to the next 'column' */
  while ((double) (i->y) > i->ye && !ri_end(i)) {
    /* increase x, next 'column' */
    i->x++;

    /* if end of exploration, return */
    if (ri_end(i)) return;

    /* update lower y limit (start) for the new 'column'.

       We need to interpolate the y value that corresponds to the
       lower side of the rectangle. The first thing is to decide if
       the corresponding side is

         vx[0],vy[0] to vx[3],vy[3] or
         vx[3],vy[3] to vx[2],vy[2]

       Then, the side is interpolated for the x value of the
       'column'. But, if the side is vertical (as it could happen if
       the rectangle is vertical and we are dealing with the first
       or last 'columns') then we pick the lower value of the side
       by using 'inter_low'.
     */
    if ((double) i->x < i->vx[3])
      i->ys = inter_low((double) i->x, i->vx[0], i->vy[0], i->vx[3], i->vy[3]);
    else
      i->ys = inter_low((double) i->x, i->vx[3], i->vy[3], i->vx[2], i->vy[2]);

    /* update upper y limit (end) for the new 'column'.

       We need to interpolate the y value that corresponds to the
       upper side of the rectangle. The first thing is to decide if
       the corresponding side is

         vx[0],vy[0] to vx[1],vy[1] or
         vx[1],vy[1] to vx[2],vy[2]

       Then, the side is interpolated for the x value of the
       'column'. But, if the side is vertical (as it could happen if
       the rectangle is vertical and we are dealing with the first
       or last 'columns') then we pick the lower value of the side
       by using 'inter_low'.
     */
    if ((double) i->x < i->vx[1])
      i->ye = inter_hi((double) i->x, i->vx[0], i->vy[0], i->vx[1], i->vy[1]);
    else
      i->ye = inter_hi((double) i->x, i->vx[1], i->vy[1], i->vx[2], i->vy[2]);

    /* new y */
    i->y = (int) ceil(i->ys);
  }
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
static rect_iter *ri_ini(struct rect *r) {
  double vx[4], vy[4];
  int n, offset;
  rect_iter *i;

  /* check parameters */
  if (r == nullptr) error("ri_ini: invalid rectangle.");

  /* get memory */
  i = (rect_iter *) malloc(sizeof(rect_iter));
  if (i == nullptr) error("ri_ini: Not enough memory.");

  /* build list of rectangle corners ordered
     in a circular way around the rectangle */
  vx[0] = r->x1 - r->dy * r->width / 2.0;
  vy[0] = r->y1 + r->dx * r->width / 2.0;
  vx[1] = r->x2 - r->dy * r->width / 2.0;
  vy[1] = r->y2 + r->dx * r->width / 2.0;
  vx[2] = r->x2 + r->dy * r->width / 2.0;
  vy[2] = r->y2 - r->dx * r->width / 2.0;
  vx[3] = r->x1 + r->dy * r->width / 2.0;
  vy[3] = r->y1 - r->dx * r->width / 2.0;

  /* compute rotation of index of corners needed so that the first
     point has the smaller x.

     if one side is vertical, thus two corners have the same smaller x
     value, the one with the largest y value is selected as the first.
   */
  if (r->x1 < r->x2 && r->y1 <= r->y2) offset = 0;
  else if (r->x1 >= r->x2 && r->y1 < r->y2) offset = 1;
  else if (r->x1 > r->x2 && r->y1 >= r->y2) offset = 2;
  else offset = 3;

  /* apply rotation of index. */
  for (n = 0; n < 4; n++) {
    i->vx[n] = vx[(offset + n) % 4];
    i->vy[n] = vy[(offset + n) % 4];
  }

  /* Set an initial condition.

     The values are set to values that will cause 'ri_inc' (that will
     be called immediately) to initialize correctly the first 'column'
     and compute the limits 'ys' and 'ye'.

     'y' is set to the integer value of vy[0], the starting corner.

     'ys' and 'ye' are set to very small values, so 'ri_inc' will
     notice that it needs to start a new 'column'.

     The smallest integer coordinate inside of the rectangle is
     'ceil(vx[0])'. The current 'x' value is set to that value minus
     one, so 'ri_inc' (that will increase x by one) will advance to
     the first 'column'.
   */
  i->x = (int) ceil(i->vx[0]) - 1;
  i->y = (int) ceil(i->vy[0]);
  i->ys = i->ye = -DBL_MAX;

  /* advance to the first pixel */
  ri_inc(i);

  return i;
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
static double rect_nfa(struct rect *rec, image_double angles, double logNT) {
  rect_iter *i;
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if (rec == nullptr) error("rect_nfa: invalid rectangle.");
  if (angles == nullptr) error("rect_nfa: invalid 'angles'.");

  /* compute the total number of pixels and of aligned points in 'rec' */
  for (i = ri_ini(rec); !ri_end(i); ri_inc(i)) /* rectangle iterator */
    if (i->x >= 0 && i->y >= 0 &&
        i->x < (int) angles->xsize && i->y < (int) angles->ysize) {
      ++pts; /* total number of pixels counter */
      if (isaligned(i->x, i->y, angles, rec->theta, rec->prec))
        ++alg; /* aligned points counter */
    }
  ri_del(i); /* delete iterator */

  return nfa(pts, alg, rec->p, logNT); /* compute NFA value */
}