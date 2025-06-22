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
#include <thread>
#include <chrono>
#include <functional>
#include <future>

#include "lsd.h"

constexpr bool with_gaussian = false;

constexpr unsigned int stride_ll_angle_x = 2;
constexpr unsigned int stride_ll_angle_y = 2;
constexpr unsigned int numberThreads = 2;
constexpr unsigned int early_stop_iterations = 25000;

constexpr double m_ln10 = 2.30258509299404568402;
constexpr double m_pi = 3.14159265358979323846;
constexpr int NOTDEF = -1024;
constexpr int NOTUSED = 0;
constexpr int USED = 1;

double UPM_GRADIENT_THRESHOLD_LSD = 5.2262518595055063;

/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist {
  unsigned int x, y;
  struct coorlist *next;
};

struct point { int x, y; };

static void error(const char *msg) {
  std::cerr << "LSD error: " << msg << "\n";
  std::exit(EXIT_FAILURE);
}

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
  constexpr double relative_error_factor = 100.0;
  double abs_diff, aa, bb, abs_max;

  /* trivial case */
  if (a == b) return true;

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
  return (abs_diff / abs_max) <= (relative_error_factor * DBL_EPSILON);
}

/*----------------------------------------------------------------------------*/
/** Computes Euclidean distance between point (x1,y1) and point (x2,y2).
 */

// TODO: Add concept here
template<typename T>
static T dist(T x1, T y1, T x2, T y2) {
  return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

/*----------------------------------------------------------------------------*/
/*-----------------------------    Image type    -----------------------------*/
/*----------------------------------------------------------------------------*/

template <typename  T>
class Image {

public:
  Image(unsigned int in_xsize, unsigned int in_ysize) : xsize(in_xsize), ysize(in_ysize) {
    if (xsize == 0 || ysize == 0) error("new Image_char: invalid image size.");

    this->data = static_cast<T*>(calloc(xsize * ysize,
                                           sizeof(T)));
    if (this->data == nullptr) error("not enough memory.");

    this->borrowed = false;
  }

  Image(unsigned int in_xsize, unsigned int in_ysize, T fill_value): Image(in_xsize, in_ysize) {
    for (int i = 0; i < this->xsize * this->ysize; i++) {
      this->data[i] = fill_value;
    }
    this->borrowed = false;
  }

  Image(unsigned in_xsize, unsigned int in_ysize, T* in_data): xsize(in_xsize), ysize(in_ysize), data(in_data) {
    borrowed = true;
  }

 ~Image() {
    if(!this->borrowed) {
      free((void *) this->data);
    }
 }

  // TODO: Delete the assignment operator
  unsigned int xsize, ysize;
  T *data;
  bool borrowed;
};

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

static void grad_angle_orientation(Image<double>& in, double threshold, Image<double>*& g, Image<double>*& modgrad){
  unsigned int n, p;
  n = in.ysize;
  p = in.xsize;

  /* allocate output image */
  g = new Image<double>(in.xsize, in.ysize);

  /* get memory for the image of gradient modulus */
  modgrad = new Image<double>(in.xsize, in.ysize);

  /* 'undefined' on the up and left boundaries */
  for (int x = 0; x < p; x++) g->data[x] = NOTDEF;
  for (int y = 0; y < n; y++) g->data[p * y] = NOTDEF;

  const unsigned int numberThreads = 16;

  /* compute gradient on the remaining pixels */
  std::vector<std::pair<int,int>> ranges(numberThreads);

  int stride = ceil(((double)p) / numberThreads);
  std::function worker = [&](int idx) {
    auto [from, to] = std::make_pair(idx * stride + 1, std::min((idx+1) * stride + 1, static_cast<int>(p)));
    for (int x = from; x < to; x++) {
      for (int y = 1; y < n; y++) {
        unsigned adr = y * p + x;

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
        double com1 = in.data[adr] - in.data[adr - p - 1];
        double com2 = in.data[adr - p] - in.data[adr - 1];

        double gx = com1 + com2; /* gradient x component */
        double gy = com1 - com2; /* gradient y component */
        double norm2 = gx * gx + gy * gy;
        double norm = sqrt(norm2 / 4.0); /* gradient norm */

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
  };

  std::vector<std::thread> threads(numberThreads);
  for(int i = 0; i < numberThreads;i++) {
    threads[i] = std::thread(worker, i);
  }

  for(std::thread &t: threads) {
      t.join();
  }
}

/*----------------------------------------------------------------------------*/
/** Computes the direction of the level line of 'in' at each point.

    The result is:
    - an Image<double>* with the angle at each pixel, or NOTDEF if not defined.
    - the Image<double>* 'modgrad' with the gradient magnitude at each point.
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
void ll_angle(Image<double>& in, double threshold,
                             struct coorlist **list_p, void **mem_p,
                             Image<double>*& modgrad, Image<double>*& g, unsigned int n_bins) {
  /* the rest of the variables are used for pseudo-ordering
     the gradient magnitude values */
  int list_count = 0;
  struct coorlist *start;
  struct coorlist *end;
  double max_grad = 0.0;

  /* check parameters */
  if (in.data == nullptr || in.xsize == 0 || in.ysize == 0)
    error("ll_angle: invalid image.");
  if (threshold < 0.0) error("ll_angle: 'threshold' must be positive.");
  if (list_p == nullptr) error("ll_angle: nullptr pointer 'list_p'.");
  if (mem_p == nullptr) error("ll_angle: nullptr pointer 'mem_p'.");
  if (n_bins == 0) error("ll_angle: 'n_bins' must be positive.");

  /* image size shortcuts */
  unsigned int n = in.ysize;
  unsigned int p = in.xsize;

  /* get memory for "ordered" list of pixels */
  struct coorlist *list = (struct coorlist *) calloc(n * p, sizeof(struct coorlist));
  *mem_p = (void *) list;

  /* array of pointers to start of bin list */
  std::vector<struct coorlist*> range_l_s(n_bins);
  std::vector<struct coorlist*> range_l_e(n_bins);

  if (list == nullptr)
    error("not enough memory.");
  for (unsigned int i = 0; i < n_bins; i++) range_l_s[i] = range_l_e[i] = nullptr;

  if (modgrad == nullptr || g == nullptr) {
    grad_angle_orientation(in, threshold, g, modgrad);
  }

  for (unsigned i = 0; i < modgrad->xsize * modgrad->ysize; i++) {
    if(max_grad < modgrad->data[i]) max_grad = modgrad->data[i];
  }

  /* compute histogram of gradient values */
  for (unsigned int x = 0; x < p - 1; x += stride_ll_angle_x) {
    for (unsigned int y = 0; y < n - 1; y += stride_ll_angle_y) {
      double norm = modgrad->data[y * p + x];

      /* store the point in the right bin according to its norm */
      unsigned i = static_cast<unsigned int>(norm * static_cast<double>(n_bins) / max_grad);
      if (i >= n_bins) i = n_bins - 1;
      if (range_l_e[i] == nullptr)
        range_l_s[i] = range_l_e[i] = list + list_count++;
      else {
        range_l_e[i]->next = list + list_count;
        range_l_e[i] = list + list_count++;
      }
      range_l_e[i]->x = x;
      range_l_e[i]->y = y;
      range_l_e[i]->next = nullptr;
    }
  }

  /* Make the list of pixels (almost) ordered by norm value.
     It starts by the larger bin, so the list starts by the
     pixels with the highest gradient value. Pixels would be ordered
     by norm value, up to a precision given by max_grad/n_bins.
   */
  unsigned int i;
  for (i = n_bins - 1; i > 0 && range_l_s[i] == nullptr; i--);
  start = range_l_s[i];
  end = range_l_e[i];
  if (start != nullptr) {
    while (i > 0) {
      --i;
      if (range_l_s[i] != nullptr) {
        end->next = range_l_s[i];
        end = range_l_e[i];
      }
    }
  }

  *list_p = start;
}

/*----------------------------------------------------------------------------*/
/** Is point (x,y) aligned to angle theta, up to precision 'prec'?
 */
static int isaligned(int x, int y, Image<double>* angles, double theta,
                     double prec) {
  /* check parameters */
  if (angles == nullptr || angles->data == nullptr)
    error("isaligned: invalid image 'angles'.");
  if (x < 0 || y < 0 || x >= angles->xsize || y >= angles->ysize)
    error("isaligned: (x,y) out of the image.");
  if (prec < 0.0) error("isaligned: 'prec' must be positive.");

  /* angle at pixel (x,y) */
  double a = angles->data[x + y * angles->xsize];

  /* pixels whose level-line angle is not defined
     are considered as NON-aligned */
  if (a == NOTDEF) return false;  /* there is no need to call the function
                                      'double_equal' here because there is
                                      no risk of problems related to the
                                      comparison doubles, we are only
                                      interested in the exact NOTDEF value */

  /* it is assumed that 'theta' and 'a' are in the range [-pi,pi] */
  theta -= a;
  if (theta < 0.0) theta = -theta;
  if (theta > 3 * m_pi / 2) {
    theta -= 2 * m_pi;
    theta = abs(theta);
  }

  return theta <= prec;
}

/*----------------------------------------------------------------------------*/
/** Absolute value angle difference.
 */
static inline double angle_diff(double a, double b) {
  a -= b;
  while (a <= -M_PI) a += 2 * m_pi;
  while (a > M_PI) a -= 2 * m_pi;
  if (a < 0.0) a = -a;
  return a;
}

/*----------------------------------------------------------------------------*/
/** Signed angle difference.
 */
static inline double angle_diff_signed(double a, double b) {
  a -= b;
  while (a <= -M_PI) a += 2 * m_pi;
  while (a > M_PI) a -= 2 * m_pi;
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

  for (int n = 0; n < 7; n++) {
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
static inline double log_gamma_windschitl(double x) {
  return 0.918938533204673 + (x - 0.5) * log(x) - x
      + 0.5 * x * log(x * sinh(1 / x) + 1 / (810.0 * pow(x, 6.0)));
}

/*----------------------------------------------------------------------------*/
/** Computes the natural logarithm of the absolute value of
    the gamma function of x. When x>15 use log_gamma_windschitl(),
    otherwise use log_gamma_lanczos().
 */
static inline double log_gamma(double x) {
  return (x)>15.0?log_gamma_windschitl(x):log_gamma_lanczos(x);
}

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
  //  Size of the table to store already computed inverse values.
  constexpr int tabsize = 100000;
  static double inv[tabsize];   /* table to keep computed inverse values */
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
    bin_term = (double) (n - i + 1) * (i < tabsize ?
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
/*---------------------------   Rectangle class   ----------------------------*/
/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
class rect {
public:
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
// TODO: Write as a copy constructor
static void rect_copy(rect *in, rect *out) {
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
    i->y = ceil(i->ys);
  }
}

/*----------------------------------------------------------------------------*/
/** Create and initialize a rectangle iterator.

    See details in \ref rect_iter
 */
static rect_iter *ri_ini(struct rect *r) {
  double vx[4], vy[4];
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
  int offset = 0;
  if (r->x1 < r->x2 && r->y1 <= r->y2) offset = 0;
  else if (r->x1 >= r->x2 && r->y1 < r->y2) offset = 1;
  else if (r->x1 > r->x2 && r->y1 >= r->y2) offset = 2;
  else offset = 3;

  /* apply rotation of index. */
  for (int n = 0; n < 4; n++) {
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
  i->x = static_cast<int>(ceil(i->vx[0]) - 1);
  i->y = static_cast<int>(ceil(i->vy[0]));
  i->ys = i->ye = -DBL_MAX;

  /* advance to the first pixel */
  ri_inc(i);

  return i;
}

/*----------------------------------------------------------------------------*/
/** Compute a rectangle's NFA value.
 */
static double rect_nfa(struct rect *rec, Image<double>* angles, double logNT) {
  int pts = 0;
  int alg = 0;

  /* check parameters */
  if (rec == nullptr) error("rect_nfa: invalid rectangle.");
  if (angles == nullptr) error("rect_nfa: invalid 'angles'.");

  /* compute the total number of pixels and of aligned points in 'rec' */
  rect_iter *i;
  for (i = ri_ini(rec); !ri_end(i); ri_inc(i)) /* rectangle iterator */
    if (i->x >= 0 && i->y >= 0 &&
        i->x < angles->xsize && i->y < angles->ysize) {
      ++pts; /* total number of pixels counter */
      if (isaligned(i->x, i->y, angles, rec->theta, rec->prec))
        ++alg; /* aligned points counter */
    }
  ri_del(i); /* delete iterator */

  return nfa(pts, alg, rec->p, logNT); /* compute NFA value */
}

/*----------------------------------------------------------------------------*/
/*----------------------------- Gaussian filter ------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute a Gaussian kernel of length 'kernel->dim',
    standard deviation 'sigma', and centered at value 'mean'.

    For example, if mean=0.5, the Gaussian will be centered
    in the middle point between values 'kernel->values[0]'
    and 'kernel->values[1]'.
 */
static void gaussian_kernel(std::vector<double>& kernel, double sigma, double mean) {
  double sum = 0.0;

  /* check parameters */
  if (sigma <= 0.0) error("gaussian_kernel: 'sigma' must be positive.");

  for (unsigned int i = 0; i < kernel.size(); i++) {
    double val = (static_cast<double>(i) - mean) / sigma;
    kernel[i] = exp(-0.5 * val * val);
    sum += kernel[i];
  }

  /* normalization */
  if (sum >= 0.0) for (unsigned int i = 0; i < kernel.size(); i++) kernel[i] /= sum;
}

/*----------------------------------------------------------------------------*/
/** Scale the input image 'in' by a factor 'scale' by Gaussian sub-sampling.

    For example, scale=0.8 will give a result at 80% of the original size.

    The image is convolved with a Gaussian kernel
    @f[
        G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
    @f]
    before the sub-sampling to prevent aliasing.

    The standard deviation sigma given by:
    -  sigma = sigma_scale / scale,   if scale <  1.0
    -  sigma = sigma_scale,           if scale >= 1.0

    To be able to sub-sample at non-integer steps, some interpolation
    is needed. In this implementation, the interpolation is done by
    the Gaussian kernel, so both operations (filtering and sampling)
    are done at the same time. The Gaussian kernel is computed
    centered on the coordinates of the required sample. In this way,
    when applied, it gives directly the result of convolving the image
    with the kernel and interpolated to that particular position.

    A fast algorithm is done using the separability of the Gaussian
    kernel. Applying the 2D Gaussian kernel is equivalent to applying
    first a horizontal 1D Gaussian kernel and then a vertical 1D
    Gaussian kernel (or the other way round). The reason is that
    @f[
        G(x,y) = G(x) * G(y)
    @f]
    where
    @f[
        G(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}.
    @f]
    The algorithm first applies a combined Gaussian kernel and sampling
    in the x axis, and then the combined Gaussian kernel and sampling
    in the y axis.
 */
static Image<double> gaussian_sampler(Image<double>& in, double scale,
                                     double sigma_scale) {
  /* check parameters */
  if (in.data == nullptr || in.xsize == 0 || in.ysize == 0)
    error("gaussian_sampler: invalid image.");
  if (scale <= 0.0) error("gaussian_sampler: 'scale' must be positive.");
  if (sigma_scale <= 0.0)
    error("gaussian_sampler: 'sigma_scale' must be positive.");

  /* compute new image size and get memory for images */
  if (in.xsize * scale > UINT_MAX ||
      in.ysize * scale > UINT_MAX)
    error("gaussian_sampler: the output image size exceeds the handled size.");
  unsigned int N = static_cast<unsigned int>(ceil(in.xsize * scale));
  unsigned int M = static_cast<unsigned int>(ceil(in.ysize * scale));

  Image aux = Image<double>(N, in.ysize);
  Image out = Image<double>(N, M);

  /* sigma, kernel size and memory for the kernel */
  double sigma = scale < 1.0 ? sigma_scale / scale : sigma_scale;
  /*
     The size of the kernel is selected to guarantee that the
     the first discarded term is at least 10^prec times smaller
     than the central value. For that, h should be larger than x, with
       e^(-x^2/2sigma^2) = 1/10^prec.
     Then,
       x = sigma * sqrt( 2 * prec * ln(10) ).
   */
  double prec = 3.0;
  unsigned int h = static_cast<unsigned int>(ceil(sigma * sqrt(2.0 * prec * log(10.0))));
  unsigned int n = 1 + 2 * h; /* kernel size */
  std::vector<double> kernel = std::vector<double>(n, 0);

  /* auxiliary double image size variables */
  int double_x_size = static_cast<int>(2 * in.xsize);
  int double_y_size = static_cast<int>(2 * in.ysize);

  /* First subsampling: x axis */
  for (unsigned int x = 0; x < aux.xsize; x++) {
    /*
       x   is the coordinate in the new image.
       xx  is the corresponding x-value in the original size image.
       xc  is the integer value, the pixel coordinate of xx.
     */
    double xx = static_cast<double>(x) / scale;
    /* coordinate (0.0,0.0) is in the center of pixel (0,0),
       so the pixel with xc=0 get the values of xx from -0.5 to 0.5 */
    int xc = static_cast<int>(floor(xx + 0.5));
    gaussian_kernel(kernel, sigma, static_cast<double>(h) + xx - static_cast<double>(xc));
    /* the kernel must be computed for each x because the fine
       offset xx-xc is different in each case */

    for (unsigned int y = 0; y < aux.ysize; y++) {
      double sum = 0.0;
      for (unsigned int i = 0; i < kernel.size(); i++) {
        int j = xc - h + i;

        /* symmetry boundary condition */
        while (j < 0) j += double_x_size;
        while (j >= double_x_size) j -= double_x_size;
        if (j >= in.xsize) j = double_x_size - 1 - j;

        sum += in.data[j + y * in.xsize] * kernel[i];
      }
      aux.data[x + y * aux.xsize] = sum;
    }
  }

  /* Second subsampling: y axis */
  for (unsigned int y = 0; y < out.ysize; y++) {
    /*
       y   is the coordinate in the new image.
       yy  is the corresponding x-value in the original size image.
       yc  is the integer value, the pixel coordinate of xx.
     */
    double yy = (double) y / scale;
    /* coordinate (0.0,0.0) is in the center of pixel (0,0),
       so the pixel with yc=0 get the values of yy from -0.5 to 0.5 */
    int yc = floor(yy + 0.5);
    gaussian_kernel(kernel, sigma, (double) h + yy - (double) yc);
    /* the kernel must be computed for each y because the fine
       offset yy-yc is different in each case */

    for (unsigned int x = 0; x < out.xsize; x++) {
      double sum = 0.0;
      for (unsigned int i = 0; i < kernel.size(); i++) {
        int j = yc - h + i;

        /* symmetry boundary condition */
        while (j < 0) j += double_y_size;
        while (j >= double_y_size) j -= double_y_size;
        if (j >= in.ysize) j = double_y_size - 1 - j;

        sum += aux.data[x + j * aux.xsize] * kernel[i];
      }
      out.data[x + y * out.xsize] = sum;
    }
  }

  return out;
}
/*----------------------------------------------------------------------------*/
/*---------------------------------- Regions ---------------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Compute region's angle as the principal inertia axis of the region.

    The following is the region inertia matrix A:
    @f[

        A = \left(\begin{array}{cc}
                                    Ixx & Ixy \\
                                    Ixy & Iyy \\
             \end{array}\right)

    @f]
    where

      Ixx =   sum_i G(i).(y_i - cx)^2

      Iyy =   sum_i G(i).(x_i - cy)^2

      Ixy = - sum_i G(i).(x_i - cx).(y_i - cy)

    and
    - G(i) is the gradient norm at pixel i, used as pixel's weight.
    - x_i and y_i are the coordinates of pixel i.
    - cx and cy are the coordinates of the center of th region.

    lambda1 and lambda2 are the eigenvalues of matrix A,
    with lambda1 >= lambda2. They are found by solving the
    characteristic polynomial:

      det( lambda I - A) = 0

    that gives:

      lambda1 = ( Ixx + Iyy + sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

      lambda2 = ( Ixx + Iyy - sqrt( (Ixx-Iyy)^2 + 4.0*Ixy*Ixy) ) / 2

    To get the line segment direction we want to get the angle the
    eigenvector associated to the smallest eigenvalue. We have
    to solve for a,b in:

      a.Ixx + b.Ixy = a.lambda2

      a.Ixy + b.Iyy = b.lambda2

    We want the angle theta = atan(b/a). It can be computed with
    any of the two equations:

      theta = atan( (lambda2-Ixx) / Ixy )

    or

      theta = atan( Ixy / (lambda2-Iyy) )

    When |Ixx| > |Iyy| we use the first, otherwise the second (just to
    get better numeric precision).
 */
static double get_theta(struct point *reg, int reg_size, double x, double y,
                        Image<double>* modgrad, double reg_angle, double prec) {
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Ixy = 0.0;

  /* check parameters */
  if (reg == nullptr) error("get_theta: invalid region.");
  if (reg_size <= 1) error("get_theta: region size <= 1.");
  if (modgrad == nullptr || modgrad->data == nullptr)
    error("get_theta: invalid 'modgrad'.");
  if (prec < 0.0) error("get_theta: 'prec' must be positive.");

  /* compute inertia matrix */
  for (int i = 0; i < reg_size; i++) {
    double weight = modgrad->data[reg[i].x + reg[i].y * modgrad->xsize];
    Ixx += ((double) reg[i].y - y) * ((double) reg[i].y - y) * weight;
    Iyy += ((double) reg[i].x - x) * ((double) reg[i].x - x) * weight;
    Ixy -= ((double) reg[i].x - x) * ((double) reg[i].y - y) * weight;
  }
  if (double_equal(Ixx, 0.0) && double_equal(Iyy, 0.0) && double_equal(Ixy, 0.0))
    error("get_theta: null inertia matrix.");

  /* compute smallest eigenvalue */
  double lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

  /* compute angle */
  double theta = fabs(Ixx) > fabs(Iyy) ? atan2(lambda - Ixx, Ixy) : atan2(Ixy, lambda - Iyy);

  /* The previous procedure doesn't cares about orientation,
     so it could be wrong by 180 degrees. Here is corrected if necessary. */
  if (angle_diff(theta, reg_angle) > prec) theta += M_PI;

  return theta;
}

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of points.
 */
static void region2rect(struct point *reg, int reg_size,
                        Image<double>* modgrad, double reg_angle,
                        double prec, double p, struct rect *rec) {

  /* check parameters */
  if (reg == nullptr) error("region2rect: invalid region.");
  if (reg_size <= 1) error("region2rect: region size <= 1.");
  if (modgrad == nullptr || modgrad->data == nullptr)
    error("region2rect: invalid image 'modgrad'.");
  if (rec == nullptr) error("region2rect: invalid 'rec'.");

  /* center of the region:

     It is computed as the weighted sum of the coordinates
     of all the pixels in the region. The norm of the gradient
     is used as the weight of a pixel. The sum is as follows:
       cx = \sum_i G(i).x_i
       cy = \sum_i G(i).y_i
     where G(i) is the norm of the gradient of pixel i
     and x_i,y_i are its coordinates.
   */
  double x = 0.0;
  double y  = 0.0;
  double sum = 0.0;
  for (int i = 0; i < reg_size; i++) {
    double weight = modgrad->data[reg[i].x + reg[i].y * modgrad->xsize];
    x += (double) reg[i].x * weight;
    y += (double) reg[i].y * weight;
    sum += weight;
  }
  if (sum <= 0.0) error("region2rect: weights sum equal to zero.");
  x /= sum;
  y /= sum;

  /* theta */
  double theta = get_theta(reg, reg_size, x, y, modgrad, reg_angle, prec);

  /* length and width:

     'l' and 'w' are computed as the distance from the center of the
     region to pixel i, projected along the rectangle axis (dx,dy) and
     to the orthogonal axis (-dy,dx), respectively.

     The length of the rectangle goes from l_min to l_max, where l_min
     and l_max are the minimum and maximum values of l in the region.
     Analogously, the width is selected from w_min to w_max, where
     w_min and w_max are the minimum and maximum of w for the pixels
     in the region.
   */
  double dx = cos(theta);
  double dy = sin(theta);
  double l_min = 0.0;
  double l_max = 0.0;
  double w_min = 0.0;
  double w_max = 0.0;
  for (int i = 0; i < reg_size; i++) {
    double l = ((double) reg[i].x - x) * dx + ((double) reg[i].y - y) * dy;
    double w = -((double) reg[i].x - x) * dy + ((double) reg[i].y - y) * dx;

    if (l > l_max) l_max = l;
    if (l < l_min) l_min = l;
    if (w > w_max) w_max = w;
    if (w < w_min) w_min = w;
  }

  /* store values */
  rec->x1 = x + l_min * dx;
  rec->y1 = y + l_min * dy;
  rec->x2 = x + l_max * dx;
  rec->y2 = y + l_max * dy;
  rec->width = w_max - w_min;
  rec->x = x;
  rec->y = y;
  rec->theta = theta;
  rec->dx = dx;
  rec->dy = dy;
  rec->prec = prec;
  rec->p = p;

  /* we impose a minimal width of one pixel

     A sharp horizontal or vertical step would produce a perfectly
     horizontal or vertical region. The width computed would be
     zero. But that corresponds to a one pixels width transition in
     the image.
   */
  if (rec->width < 1.0) rec->width = 1.0;
}

/*----------------------------------------------------------------------------*/
/** Build a region of pixels that share the same angle, up to a
    tolerance 'prec', starting at point (x,y).
 */
static void region_grow(int x, int y, Image<double>* angles, struct point *reg,
                        int *reg_size, double *reg_angle, Image<char> *used,
                        double prec) {

  /* check parameters */
  if (x < 0 || y < 0 || x >= angles->xsize || y >= angles->ysize)
    error("region_grow: (x,y) out of the image.");
  if (angles == nullptr || angles->data == nullptr)
    error("region_grow: invalid image 'angles'.");
  if (reg == nullptr) error("region_grow: invalid 'reg'.");
  if (reg_size == nullptr) error("region_grow: invalid pointer 'reg_size'.");
  if (reg_angle == nullptr) error("region_grow: invalid pointer 'reg_angle'.");
  if (used == nullptr || used->data == nullptr)
    error("region_grow: invalid image 'used'.");

  /* first point of the region */
  *reg_size = 1;
  reg[0].x = x;
  reg[0].y = y;
  *reg_angle = angles->data[x + y * angles->xsize];  /* region's angle */
  double sumdx = cos(*reg_angle);
  double sumdy = sin(*reg_angle);
  used->data[x + y * used->xsize] = USED;

  /* try neighbors as new region points */
  for (unsigned int i = 0; i < *reg_size; i++)
    for (unsigned int xx = reg[i].x - 1; xx <= reg[i].x + 1; xx++)
      for (unsigned int yy = reg[i].y - 1; yy <= reg[i].y + 1; yy++)
        if (xx >= 0 && yy >= 0 && xx < used->xsize && yy < used->ysize &&
            used->data[xx + yy * used->xsize] != USED &&
            isaligned(xx, yy, angles, *reg_angle, prec)) {
          /* add point */
          used->data[xx + yy * used->xsize] = USED;
          reg[*reg_size].x = xx;
          reg[*reg_size].y = yy;
          ++(*reg_size);

          /* update region's angle */
          sumdx += cos(angles->data[xx + yy * angles->xsize]);
          sumdy += sin(angles->data[xx + yy * angles->xsize]);
          *reg_angle = atan2(sumdy, sumdx);
        }
}

/*----------------------------------------------------------------------------*/
/** Try some rectangles variations to improve NFA value. Only if the
    rectangle is not meaningful (i.e., log_nfa <= log_eps).
 */
static double rect_improve(struct rect *rec, Image<double>* angles,
                           double logNT, double log_eps) {
  double delta = 0.5;
  double delta_2 = delta / 2.0;

  double log_nfa = rect_nfa(rec, angles, logNT);

  if (log_nfa > log_eps) return log_nfa;

  /* try finer precisions */
  struct rect r;
  rect_copy(rec, &r);
  for (int n = 0; n < 5; n++) {
    r.p /= 2.0;
    r.prec = r.p * M_PI;
    double log_nfa_new = rect_nfa(&r, angles, logNT);
    if (log_nfa_new > log_nfa) {
      log_nfa = log_nfa_new;
      rect_copy(&r, rec);
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try to reduce width */
  rect_copy(rec, &r);
  for (int n = 0; n < 5; n++) {
    if ((r.width - delta) >= 0.5) {
      r.width -= delta;
      double log_nfa_new = rect_nfa(&r, angles, logNT);
      if (log_nfa_new > log_nfa) {
        rect_copy(&r, rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect_copy(rec, &r);
  for (int n = 0; n < 5; n++) {
    if ((r.width - delta) >= 0.5) {
      r.x1 += -r.dy * delta_2;
      r.y1 += r.dx * delta_2;
      r.x2 += -r.dy * delta_2;
      r.y2 += r.dx * delta_2;
      r.width -= delta;
      double log_nfa_new = rect_nfa(&r, angles, logNT);
      if (log_nfa_new > log_nfa) {
        rect_copy(&r, rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect_copy(rec, &r);
  for (int n = 0; n < 5; n++) {
    if ((r.width - delta) >= 0.5) {
      r.x1 -= -r.dy * delta_2;
      r.y1 -= r.dx * delta_2;
      r.x2 -= -r.dy * delta_2;
      r.y2 -= r.dx * delta_2;
      r.width -= delta;
      double log_nfa_new = rect_nfa(&r, angles, logNT);
      if (log_nfa_new > log_nfa) {
        rect_copy(&r, rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try even finer precisions */
  rect_copy(rec, &r);
  for (int n = 0; n < 5; n++) {
    r.p /= 2.0;
    r.prec = r.p * M_PI;
    double log_nfa_new = rect_nfa(&r, angles, logNT);
    if (log_nfa_new > log_nfa) {
      log_nfa = log_nfa_new;
      rect_copy(&r, rec);
    }
  }

  return log_nfa;
}

/*----------------------------------------------------------------------------*/
/** Reduce the region size, by elimination the points far from the
    starting point, until that leads to rectangle with the right
    density of region points or to discard the region if too small.
 */
static int reduce_region_radius(struct point *reg, int *reg_size,
                                Image<double>* modgrad, double reg_angle,
                                double prec, double p, struct rect *rec,
                                Image<char> *used, Image<double>* angles,
                                double density_th) {

  /* check parameters */
  if (reg == nullptr) error("reduce_region_radius: invalid pointer 'reg'.");
  if (reg_size == nullptr)
    error("reduce_region_radius: invalid pointer 'reg_size'.");
  if (prec < 0.0) error("reduce_region_radius: 'prec' must be positive.");
  if (rec == nullptr) error("reduce_region_radius: invalid pointer 'rec'.");
  if (used == nullptr || used->data == nullptr)
    error("reduce_region_radius: invalid image 'used'.");
  if (angles == nullptr || angles->data == nullptr)
    error("reduce_region_radius: invalid image 'angles'.");

  /* compute region points density */
  double density = (double) *reg_size /
      (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);

  /* if the density criterion is satisfied there is nothing to do */
  if (density >= density_th) return true;

  /* compute region's radius */
  double xc = (double) reg[0].x;
  double yc = (double) reg[0].y;
  double rad1 = dist(xc, yc, rec->x1, rec->y1);
  double rad2 = dist(xc, yc, rec->x2, rec->y2);
  double rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while (density < density_th) {
    rad *= 0.75; /* reduce region's radius to 75% of its value */

    /* remove points from the region and update 'used' map */
    for (unsigned  int i = 0; i < *reg_size; i++)
      if (dist(xc, yc, (double) reg[i].x, (double) reg[i].y) > rad) {
        /* point not kept, mark it as NOTUSED */
        used->data[reg[i].x + reg[i].y * used->xsize] = NOTUSED;
        /* remove point from the region */
        reg[i].x = reg[*reg_size - 1].x; /* if i==*reg_size-1 copy itself */
        reg[i].y = reg[*reg_size - 1].y;
        --(*reg_size);
        --i; /* to avoid skipping one point */
      }

    /* reject if the region is too small.
       2 is the minimal region size for 'region2rect' to work. */
    if (*reg_size < 2) return false;

    /* re-compute rectangle */
    region2rect(reg, *reg_size, modgrad, reg_angle, prec, p, rec);

    /* re-compute region points density */
    density = (double) *reg_size /
        (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);
  }

  /* if this point is reached, the density criterion is satisfied */
  return true;
}

/*----------------------------------------------------------------------------*/
/** Refine a rectangle.

    For that, an estimation of the angle tolerance is performed by the
    standard deviation of the angle at points near the region's
    starting point. Then, a new region is grown starting from the same
    point, but using the estimated angle tolerance. If this fails to
    produce a rectangle with the right density of region points,
    'reduce_region_radius' is called to try to satisfy this condition.
 */
static int refine(struct point *reg, int *reg_size, Image<double>* modgrad,
                  double reg_angle, double prec, double p, struct rect *rec,
                  Image<char> *used, Image<double>* angles, double density_th) {

  /* check parameters */
  if (reg == nullptr) error("refine: invalid pointer 'reg'.");
  if (reg_size == nullptr) error("refine: invalid pointer 'reg_size'.");
  if (prec < 0.0) error("refine: 'prec' must be positive.");
  if (rec == nullptr) error("refine: invalid pointer 'rec'.");
  if (used == nullptr || used->data == nullptr)
    error("refine: invalid image 'used'.");
  if (angles == nullptr || angles->data == nullptr)
    error("refine: invalid image 'angles'.");

  /* compute region points density */
  double density = (double) *reg_size /
      (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);

  /* if the density criterion is satisfied there is nothing to do */
  if (density >= density_th) return true;

  /*------ First try: reduce angle tolerance ------*/

  /* compute the new mean angle and tolerance */
  double xc = (double) reg[0].x;
  double yc = (double) reg[0].y;
  double ang_c = angles->data[reg[0].x + reg[0].y * angles->xsize];
  double sum = 0.0;
  double s_sum = 0.0;
  unsigned int n = 0;
  for (unsigned int i = 0; i < *reg_size; i++) {
    used->data[reg[i].x + reg[i].y * used->xsize] = NOTUSED;
    if (dist(xc, yc, (double) reg[i].x, (double) reg[i].y) < rec->width) {
      double angle = angles->data[reg[i].x + reg[i].y * angles->xsize];
      double ang_d = angle_diff_signed(angle, ang_c);
      sum += ang_d;
      s_sum += ang_d * ang_d;
      ++n;
    }
  }
  double mean_angle = sum / (double) n;
  double tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / (double) n
                       + mean_angle * mean_angle); /* 2 * standard deviation */

  /* find a new region from the same starting point and new angle tolerance */
  region_grow(reg[0].x, reg[0].y, angles, reg, reg_size, &reg_angle, used, tau);

  /* if the region is too small, reject */
  if (*reg_size < 2) return false;

  /* re-compute rectangle */
  region2rect(reg, *reg_size, modgrad, reg_angle, prec, p, rec);

  /* re-compute region points density */
  density = (double) *reg_size /
      (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);

  /*------ Second try: reduce region radius ------*/
  if (density < density_th)
    return reduce_region_radius(reg, reg_size, modgrad, reg_angle, prec, p,
                                rec, used, angles, density_th);

  /* if this point is reached, the density criterion is satisfied */
  return true;
}


/*----------------------------------------------------------------------------*/
/*-------------------------- Line Segment Detector ---------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** LSD full interface.
 */
double *LineSegmentDetection(int *n_out,
                             double *img, int X, int Y,
                             double scale, double sigma_scale, double quant,
                             double ang_th, double log_eps, double density_th,
                             int n_bins, bool grad_nfa,
                             double * modgrad_ptr, double * angles_ptr,
                             int **reg_img, int *reg_x, int *reg_y) {
  Image<int> * region = nullptr;
  struct coorlist *list_p, *list_pp;
  void *mem_p, *mem_pp;
  struct point *reg;
  int ls_count = 0;                   /* line segments are numbered 1,2,3,... */

  /* check parameters */
  if (img == nullptr || X <= 0 || Y <= 0) error("invalid image input.");
  if (scale <= 0.0) error("'scale' value must be positive.");
  if (sigma_scale <= 0.0) error("'sigma_scale' value must be positive.");
  if (quant < 0.0) error("'quant' value must be positive.");
  if (ang_th <= 0.0 || ang_th >= 180.0)
    error("'ang_th' value must be in the range (0,180).");
  if (density_th < 0.0 || density_th > 1.0)
    error("'density_th' value must be in the range [0,1].");
  if (n_bins <= 0) error("'n_bins' value must be positive.");


  /* angle tolerance */
  double prec = M_PI * ang_th / 180.0;
  double p = ang_th / 180.0;
  double rho = quant / sin(prec); /* gradient magnitude threshold */
  // Iago
  rho = UPM_GRADIENT_THRESHOLD_LSD;
  // std::cout << "LSD Gradient threshold: " << rho << std::endl;

  Image<double>* modgrad = nullptr;
  Image<double>* angles = nullptr;
  Image<double>* img_gradnorm = nullptr;
  Image<double>* img_grad_angle = nullptr;

  // TODO: In the future handle case with modgrad_ptr and angles_ptr
  if (modgrad_ptr) {
    modgrad = new Image<double>(X, Y, modgrad_ptr);
  }
  if (angles_ptr) {
    angles = new Image<double>(X, Y, angles_ptr);
  }

  auto start = std::chrono::high_resolution_clock::now();

  /* load and scale image (if necessary) and compute angle at each pixel */
  Image image = Image<double>((unsigned ) X, (unsigned int) Y, img);

  if (scale != 1.0 || with_gaussian) {
    Image<double> scaled_image = gaussian_sampler(image, scale, sigma_scale);
    if (grad_nfa) {
      ll_angle(scaled_image, rho, &list_pp, &mem_pp, img_gradnorm, img_grad_angle, (unsigned int) n_bins);
    }
    ll_angle(scaled_image, rho, &list_p, &mem_p, modgrad, angles, (unsigned int) n_bins);
  } else {
    if (grad_nfa) {
      ll_angle(image, rho, &list_pp, &mem_pp, img_gradnorm, img_grad_angle, (unsigned int) n_bins);
    }
    ll_angle(image, rho, &list_p, &mem_p, modgrad, angles, (unsigned int) n_bins);
  }

  const unsigned int xsize = angles->xsize;
  const unsigned int ysize = angles->ysize;

  /* Number of Tests - NT

 The theoretical number of tests is Np.(XY)^(5/2)
 where X and Y are number of columns and rows of the image.
 Np corresponds to the number of angle precisions considered.
 As the procedure 'rect_improve' tests 5 times to halve the
 angle precision, and 5 more times after improving other factors,
 11 different precision values are potentially tested. Thus,
 the number of tests is
   11 * (X*Y)^(5/2)
 whose logarithm value is
   log10(11) + 5/2 * (log10(X) + log10(Y)).
*/
  double logNT = 5.0 * (log10((double) xsize) + log10((double) ysize)) / 2.0
    + log10(11.0);
  int min_reg_size = static_cast<int>(-logNT / log10(p)); /* minimal number of points in region
                                             that can give a meaningful event */

  std::vector<point*> registers_vector(numberThreads);
  for(int i = 0; i < numberThreads;i++) {
    registers_vector[i] = static_cast<point *>(malloc((xsize * ysize) * sizeof(point)));
  }

  /* initialize some structures */
  if (reg_img != nullptr && reg_x != nullptr && reg_y != nullptr) /* save region data */
    region = new Image<int>(angles->xsize, angles->ysize, 0);
  auto used = new Image<char>(xsize, ysize, static_cast<char>(0));

  reg = static_cast<struct point*>(malloc((size_t) (xsize * ysize) * sizeof(struct point)));
  if (reg == nullptr) error("not enough memory!");

  /* search for line segments */
  std::function<std::vector<double>(int)> worker = [&](int index) {
    std::vector<double> output;
    auto list_p2 = list_p;
    unsigned int counter = 0;
    for (; list_p2 != nullptr; list_p2 = list_p2->next) {
      if(counter++ % numberThreads != index) continue;
      if (counter >= early_stop_iterations) break;

    auto [x,y] = std::make_pair(list_p2->x, list_p2->y);
    if (used->data[x + y * used->xsize] == NOTUSED &&
        angles->data[x + y * angles->xsize] != NOTDEF)
      /* there is no risk of double comparison problems here
         because we are only interested in the exact NOTDEF value */
    {
      // We don't want to share those value accross the threads
      // Otherwise the threads will fights for it
      rect rec;
      int reg_size;
      double reg_angle;
      /* find the region of connected point and ~equal angle */
      region_grow(x, y, angles, registers_vector[index], &reg_size,
                  &reg_angle, used, prec);

      /* reject small regions */
      if (reg_size < min_reg_size) continue;

      /* construct rectangular approximation for the region */
      region2rect(registers_vector[index], reg_size, modgrad, reg_angle, prec, p, &rec);

      double log_nfa;
      /* compute NFA value */
      if(grad_nfa)
        log_nfa = rect_improve(&rec, img_grad_angle, logNT, log_eps);
      else
        log_nfa = rect_improve(&rec, angles, logNT, log_eps);
      if (log_nfa <= log_eps) continue;

      /* A New Line Segment was found! */
      ++ls_count;  /* increase line segment counter */

      /* scale the result values if a subsampling was performed */
      if (scale != 1.0) {
        rec.x1 /= scale;
        rec.y1 /= scale;
        rec.x2 /= scale;
        rec.y2 /= scale;
        rec.width /= scale;
      }

      /* add region number to 'region' image if needed */
      if (region != nullptr)
        for (int i = 0; i < reg_size; i++)
          region->data[reg[i].x + reg[i].y * region->xsize] = ls_count;

      /* add line segment found to output */
      output.push_back(rec.x1);
      output.push_back(rec.y1);
      output.push_back(rec.x2);
      output.push_back(rec.y2);
      output.push_back(rec.width);
      output.push_back(rec.p);
      output.push_back(log_nfa);
    }
   }

   return output;
  };

  std::vector<std::future<std::vector<double>>> futures(numberThreads);
  for(int i = 0; i < numberThreads;i++) {
    futures[i] = std::async(std::launch::async,worker, i);
  }

  std::vector<std::vector<double>> values(numberThreads);
  for(int i = 0; i < numberThreads;i++) {
    values[i] = futures[i].get();
  }

  *n_out = 0;
  for(std::vector<double> &points: values) *n_out += points.size();

  double* buffer = static_cast<double*>(malloc(*n_out * sizeof(double)));

  size_t idx = 0;
  for(std::vector<double> &points: values) {
    for(double entry: points) {
      buffer[idx++] = entry;
    }
  } 

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  if(false) {
    std::cout << "Time elapsed: " << duration.count() << " seconds\n";
  }

  angles_ptr ? free(angles) : delete angles;
  modgrad_ptr ? free(modgrad) : delete modgrad;
  if (grad_nfa) {
    delete img_gradnorm;
    delete img_grad_angle;
  }


  free((void *) reg);
  free((void *) mem_p);
  if (grad_nfa)
    free((void *) mem_pp);

  /* return the result */
  if (reg_img != nullptr && reg_x != nullptr && reg_y != nullptr) {
    if (region == nullptr) error("'region' should be a valid image.");
    *reg_img = region->data;
    if (region->xsize > (unsigned int) INT_MAX)
      error("region image to big to fit in INT sizes.");
    *reg_x = region->xsize;
    *reg_y = region->ysize;

    /* free the 'region' structure.
       we cannot use the function 'free_Image<int> *' because we need to keep
       the memory with the image data to be returned by this function. */
    delete region;
  }

  *n_out /= 7;
  return buffer;
}

/*----------------------------------------------------------------------------*/
/** LSD Simple Interface.
 */
double *lsd(int *n_out, double *img, int X, int Y, double gradientThreshold, double log_eps) {
  /* LSD parameters */
  double scale = 1.0;       /* Scale the image by Gaussian filter to 'scale'. */
  double sigma_scale = 0.6; /* Sigma for Gaussian filter is computed as
                                sigma = sigma_scale/scale.                    */
  double quant = 2.0;       /* Bound to the quantization error on the
                                gradient norm.                                */
  double ang_th = 22.5;     /* Gradient angle tolerance in degrees.           */
  // double log_eps = 0.0;     /* Detection threshold: -log10(NFA) > log_eps     */
  double density_th = 0.7;  /* Minimal density of region points in rectangle. */
  constexpr int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
                               modulus.                                       */

  double prev_grad_val = UPM_GRADIENT_THRESHOLD_LSD;
  UPM_GRADIENT_THRESHOLD_LSD = gradientThreshold;
  double *result = LineSegmentDetection(n_out, img, X, Y, scale, sigma_scale, quant,
                              ang_th, log_eps, density_th, n_bins, false,
                              nullptr, nullptr, nullptr, nullptr, nullptr);
  UPM_GRADIENT_THRESHOLD_LSD = prev_grad_val;
  return result;
}
/*----------------------------------------------------------------------------*/