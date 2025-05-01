#include "helpers.h"

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
                        image_double modgrad, double reg_angle, double prec) {
  double lambda, theta, weight;
  double Ixx = 0.0;
  double Iyy = 0.0;
  double Ixy = 0.0;
  int i;

  /* check parameters */
  if (reg == nullptr) error("get_theta: invalid region.");
  if (reg_size <= 1) error("get_theta: region size <= 1.");
  if (modgrad == nullptr || modgrad->data == nullptr)
    error("get_theta: invalid 'modgrad'.");
  if (prec < 0.0) error("get_theta: 'prec' must be positive.");

  /* compute inertia matrix */
  for (i = 0; i < reg_size; i++) {
    weight = modgrad->data[reg[i].x + reg[i].y * modgrad->xsize];
    Ixx += ((double) reg[i].y - y) * ((double) reg[i].y - y) * weight;
    Iyy += ((double) reg[i].x - x) * ((double) reg[i].x - x) * weight;
    Ixy -= ((double) reg[i].x - x) * ((double) reg[i].y - y) * weight;
  }
  if (double_equal(Ixx, 0.0) && double_equal(Iyy, 0.0) && double_equal(Ixy, 0.0))
    error("get_theta: null inertia matrix.");

  /* compute smallest eigenvalue */
  lambda = 0.5 * (Ixx + Iyy - sqrt((Ixx - Iyy) * (Ixx - Iyy) + 4.0 * Ixy * Ixy));

  /* compute angle */
  theta = fabs(Ixx) > fabs(Iyy) ? atan2(lambda - Ixx, Ixy) : atan2(Ixy, lambda - Iyy);

  /* The previous procedure doesn't cares about orientation,
     so it could be wrong by 180 degrees. Here is corrected if necessary. */
  if (angle_diff(theta, reg_angle) > prec) theta += M_PI;

  return theta;
}

/*----------------------------------------------------------------------------*/
/** Computes a rectangle that covers a region of points.
 */
static void region2rect(struct point *reg, int reg_size,
                        image_double modgrad, double reg_angle,
                        double prec, double p, struct rect *rec) {
  double x, y, dx, dy, l, w, theta, weight, sum, l_min, l_max, w_min, w_max;
  int i;

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
  x = y = sum = 0.0;
  for (i = 0; i < reg_size; i++) {
    weight = modgrad->data[reg[i].x + reg[i].y * modgrad->xsize];
    x += (double) reg[i].x * weight;
    y += (double) reg[i].y * weight;
    sum += weight;
  }
  if (sum <= 0.0) error("region2rect: weights sum equal to zero.");
  x /= sum;
  y /= sum;

  /* theta */
  theta = get_theta(reg, reg_size, x, y, modgrad, reg_angle, prec);

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
  dx = cos(theta);
  dy = sin(theta);
  l_min = l_max = w_min = w_max = 0.0;
  for (i = 0; i < reg_size; i++) {
    l = ((double) reg[i].x - x) * dx + ((double) reg[i].y - y) * dy;
    w = -((double) reg[i].x - x) * dy + ((double) reg[i].y - y) * dx;

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
static void region_grow(int x, int y, image_double angles, struct point *reg,
                        int *reg_size, double *reg_angle, image_char used,
                        double prec) {
  double sumdx, sumdy;
  int xx, yy, i;

  /* check parameters */
  if (x < 0 || y < 0 || x >= (int) angles->xsize || y >= (int) angles->ysize)
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
  sumdx = cos(*reg_angle);
  sumdy = sin(*reg_angle);
  used->data[x + y * used->xsize] = USED;

  /* try neighbors as new region points */
  for (i = 0; i < *reg_size; i++)
    for (xx = reg[i].x - 1; xx <= reg[i].x + 1; xx++)
      for (yy = reg[i].y - 1; yy <= reg[i].y + 1; yy++)
        if (xx >= 0 && yy >= 0 && xx < (int) used->xsize && yy < (int) used->ysize &&
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
static double rect_improve(struct rect *rec, image_double angles,
                           double logNT, double log_eps) {
  struct rect r;
  double log_nfa, log_nfa_new;
  double delta = 0.5;
  double delta_2 = delta / 2.0;
  int n;

  log_nfa = rect_nfa(rec, angles, logNT);

  if (log_nfa > log_eps) return log_nfa;

  /* try finer precisions */
  rect_copy(rec, &r);
  for (n = 0; n < 5; n++) {
    r.p /= 2.0;
    r.prec = r.p * M_PI;
    log_nfa_new = rect_nfa(&r, angles, logNT);
    if (log_nfa_new > log_nfa) {
      log_nfa = log_nfa_new;
      rect_copy(&r, rec);
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try to reduce width */
  rect_copy(rec, &r);
  for (n = 0; n < 5; n++) {
    if ((r.width - delta) >= 0.5) {
      r.width -= delta;
      log_nfa_new = rect_nfa(&r, angles, logNT);
      if (log_nfa_new > log_nfa) {
        rect_copy(&r, rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try to reduce one side of the rectangle */
  rect_copy(rec, &r);
  for (n = 0; n < 5; n++) {
    if ((r.width - delta) >= 0.5) {
      r.x1 += -r.dy * delta_2;
      r.y1 += r.dx * delta_2;
      r.x2 += -r.dy * delta_2;
      r.y2 += r.dx * delta_2;
      r.width -= delta;
      log_nfa_new = rect_nfa(&r, angles, logNT);
      if (log_nfa_new > log_nfa) {
        rect_copy(&r, rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try to reduce the other side of the rectangle */
  rect_copy(rec, &r);
  for (n = 0; n < 5; n++) {
    if ((r.width - delta) >= 0.5) {
      r.x1 -= -r.dy * delta_2;
      r.y1 -= r.dx * delta_2;
      r.x2 -= -r.dy * delta_2;
      r.y2 -= r.dx * delta_2;
      r.width -= delta;
      log_nfa_new = rect_nfa(&r, angles, logNT);
      if (log_nfa_new > log_nfa) {
        rect_copy(&r, rec);
        log_nfa = log_nfa_new;
      }
    }
  }

  if (log_nfa > log_eps) return log_nfa;

  /* try even finer precisions */
  rect_copy(rec, &r);
  for (n = 0; n < 5; n++) {
    r.p /= 2.0;
    r.prec = r.p * M_PI;
    log_nfa_new = rect_nfa(&r, angles, logNT);
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
                                image_double modgrad, double reg_angle,
                                double prec, double p, struct rect *rec,
                                image_char used, image_double angles,
                                double density_th) {
  double density, rad1, rad2, rad, xc, yc;
  int i;

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
  density = (double) *reg_size /
      (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);

  /* if the density criterion is satisfied there is nothing to do */
  if (density >= density_th) return TRUE;

  /* compute region's radius */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  rad1 = dist(xc, yc, rec->x1, rec->y1);
  rad2 = dist(xc, yc, rec->x2, rec->y2);
  rad = rad1 > rad2 ? rad1 : rad2;

  /* while the density criterion is not satisfied, remove farther pixels */
  while (density < density_th) {
    rad *= 0.75; /* reduce region's radius to 75% of its value */

    /* remove points from the region and update 'used' map */
    for (i = 0; i < *reg_size; i++)
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
    if (*reg_size < 2) return FALSE;

    /* re-compute rectangle */
    region2rect(reg, *reg_size, modgrad, reg_angle, prec, p, rec);

    /* re-compute region points density */
    density = (double) *reg_size /
        (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);
  }

  /* if this point is reached, the density criterion is satisfied */
  return TRUE;
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
static int refine(struct point *reg, int *reg_size, image_double modgrad,
                  double reg_angle, double prec, double p, struct rect *rec,
                  image_char used, image_double angles, double density_th) {
  double angle, ang_d, mean_angle, tau, density, xc, yc, ang_c, sum, s_sum;
  int i, n;

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
  density = (double) *reg_size /
      (dist(rec->x1, rec->y1, rec->x2, rec->y2) * rec->width);

  /* if the density criterion is satisfied there is nothing to do */
  if (density >= density_th) return TRUE;

  /*------ First try: reduce angle tolerance ------*/

  /* compute the new mean angle and tolerance */
  xc = (double) reg[0].x;
  yc = (double) reg[0].y;
  ang_c = angles->data[reg[0].x + reg[0].y * angles->xsize];
  sum = s_sum = 0.0;
  n = 0;
  for (i = 0; i < *reg_size; i++) {
    used->data[reg[i].x + reg[i].y * used->xsize] = NOTUSED;
    if (dist(xc, yc, (double) reg[i].x, (double) reg[i].y) < rec->width) {
      angle = angles->data[reg[i].x + reg[i].y * angles->xsize];
      ang_d = angle_diff_signed(angle, ang_c);
      sum += ang_d;
      s_sum += ang_d * ang_d;
      ++n;
    }
  }
  mean_angle = sum / (double) n;
  tau = 2.0 * sqrt((s_sum - 2.0 * mean_angle * sum) / (double) n
                       + mean_angle * mean_angle); /* 2 * standard deviation */

  /* find a new region from the same starting point and new angle tolerance */
  region_grow(reg[0].x, reg[0].y, angles, reg, reg_size, &reg_angle, used, tau);

  /* if the region is too small, reject */
  if (*reg_size < 2) return FALSE;

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
  return TRUE;
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
  image_double image;
  double *return_value;
  image_double scaled_image;
  image_char used;
  image_int region = nullptr;
  struct coorlist *list_p, *list_pp;
  void *mem_p, *mem_pp;
  struct point *reg;
  int min_reg_size, i;
  unsigned int xsize = 1024, ysize = 768;
  double rho, reg_angle, prec, p, log_nfa, logNT;
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
  prec = M_PI * ang_th / 180.0;
  p = ang_th / 180.0;
  rho = quant / sin(prec); /* gradient magnitude threshold */
  // Iago
  rho = UPM_GRADIENT_THRESHOLD_LSD;
  // std::cout << "LSD Gradient threshold: " << rho << std::endl;

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
  logNT = 5.0 * (log10((double) xsize) + log10((double) ysize)) / 2.0
    + log10(11.0);
  min_reg_size = (int) (-logNT / log10(p)); /* minimal number of points in region
                                             that can give a meaningful event */

  image_double modgrad{}, angles{};
  image_double img_gradnorm{}, img_grad_angle{};
  if (modgrad_ptr) {
    modgrad = new_image_double_ptr(X, Y, modgrad_ptr);
  }
  if (angles_ptr) {
    angles = new_image_double_ptr(X, Y, angles_ptr);
  }

  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr((unsigned int) X, (unsigned int) Y, img);

  std::vector<point*> registers_vector(numberThreads);
  for(int i = 0; i < numberThreads;i++) {
    registers_vector[i] = static_cast<point *>(malloc((xsize * ysize) * sizeof(point)));
  }

  /* initialize some structures */
  if (reg_img != nullptr && reg_x != nullptr && reg_y != nullptr) /* save region data */
    region = new_image_int_ini(angles->xsize, angles->ysize, 0);
  used = new_image_char_ini(xsize, ysize, NOTUSED);

  reg = static_cast<struct point*>(malloc((size_t) (xsize * ysize) * sizeof(struct point)));
  if (reg == nullptr) error("not enough memory!");

  auto start = std::chrono::high_resolution_clock::now();

  /* load and scale image (if necessary) and compute angle at each pixel */
  image = new_image_double_ptr((unsigned int) X, (unsigned int) Y, img);

  if (scale != 1.0) {
    scaled_image = gaussian_sampler(image, scale, sigma_scale);
    if (grad_nfa)
      ll_angle(scaled_image, rho, &list_pp, &mem_pp, img_gradnorm, img_grad_angle, (unsigned int) n_bins);
    ll_angle(scaled_image, rho, &list_p, &mem_p, modgrad, angles, (unsigned int) n_bins);
    free_image_double(scaled_image);
  } else {
    if (grad_nfa)
      ll_angle(image, rho, &list_pp, &mem_pp, img_gradnorm, img_grad_angle, (unsigned int) n_bins);
    ll_angle(image, rho, &list_p, &mem_p, modgrad, angles, (unsigned int) n_bins);
  }

  xsize = angles->xsize;
  ysize = angles->ysize;

  /* search for line segments */
  std::function<std::vector<double>(int)> worker = [&](int index) {
    std::vector<double> output;
    auto list_p2 = list_p;
    unsigned int counter = 0;
    for (; list_p2 != nullptr; list_p2 = list_p2->next) {
      if(counter++ % numberThreads != index) continue;
      if(counter >= 100000) break;

    auto [x,y] = std::make_pair(list_p2->x, list_p2->y);
    if (used->data[x + y * used->xsize] == NOTUSED &&
        angles->data[x + y * angles->xsize] != NOTDEF)
      /* there is no risk of double comparison problems here
         because we are only interested in the exact NOTDEF value */
    {
      // We don't want to share those value accross the threads
      // Otherwise the threads will fights for it
      struct rect rec;
      int reg_size;
      double reg_angle;
      /* find the region of connected point and ~equal angle */
      region_grow(x, y, angles, registers_vector[index], &reg_size,
                  &reg_angle, used, prec);

      /* reject small regions */
      if (reg_size < min_reg_size) continue;

      /* construct rectangular approximation for the region */
      region2rect(registers_vector[index], reg_size, modgrad, reg_angle, prec, p, &rec);

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
        for (i = 0; i < reg_size; i++)
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

  std::cout << "Time elapsed: " << duration.count() << " seconds\n";

  /* free memory */
  free((void *) image);   /* only the double_image structure should be freed,
                               the data pointer was provided to this functions
                               and should not be destroyed.                 */
  angles_ptr ? free(angles) : free_image_double(angles);
  modgrad_ptr ? free(modgrad) : free_image_double(modgrad);
  if (grad_nfa) {
    free_image_double(img_gradnorm);
    free_image_double(img_grad_angle);
  }

  free_image_char(used);
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
    *reg_x = (int) (region->xsize);
    *reg_y = (int) (region->ysize);

    /* free the 'region' structure.
       we cannot use the function 'free_image_int' because we need to keep
       the memory with the image data to be returned by this function. */
    free((void *) region);
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
  int n_bins = 1024;        /* Number of bins in pseudo-ordering of gradient
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