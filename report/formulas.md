# Formulas used (copy into the report)

**Windowing** (level L, width W): g = 255 x (clip(HU, L - W/2, L + W/2) - (L - W/2)) / W

**PSNR**: MSE = (1/MN) sum (f - g)^2 ; PSNR = 10 log10(255^2 / MSE) dB

**CNR**: CNR = |mean_A - mean_B| / sqrt(var_A + var_B)  (A = liver or tumour, B = surroundings or healthy liver)

**Bilateral filter**: g(p) = (1/W_p) sum_q f(q) exp(-|p-q|^2 / 2 sigma_s^2) exp(-(f(p)-f(q))^2 / 2 sigma_r^2)

**Anisotropic diffusion (Perona-Malik)**: I_{t+1} = I_t + gamma sum_{dir} c(dI) dI ,  c(dI) = exp(-(dI / kappa)^2)

**Gaussian**: G(x, y) = (1 / 2 pi sigma^2) exp(-(x^2 + y^2) / 2 sigma^2)

**Histogram equalisation**: s_k = 255 x sum_{j<=k} p(r_j)   (CLAHE: the same per tile, histogram clipped at the clip limit)

**2D DFT**: F(u, v) = sum_x sum_y f(x, y) exp(-j 2 pi (ux/M + vy/N)) ; spectrum shown as log(1 + |F|)

**Butterworth lowpass**: H(u, v) = 1 / (1 + (D(u, v) / D0)^(2n))

**Homomorphic filter**: H(u, v) = (gamma_H - gamma_L)(1 - exp(-c D^2 / D0^2)) + gamma_L , applied to log(f), then exp

**Otsu / Multi-Otsu**: choose thresholds that maximise the between-class variance sigma_B^2 = sum_k w_k (mu_k - mu_T)^2

**Adaptive threshold**: T(x, y) = mean of the b x b neighbourhood - C

**Morphology**: erosion A (-) B, dilation A (+) B, opening (A (-) B) (+) B, closing (A (+) B) (-) B

**Dice** = 2|A n B| / (|A| + |B|) ; **IoU (Jaccard)** = |A n B| / |A u B|

**Shape**: circularity = 4 pi A / P^2 ; compactness = P^2 / A ; solidity = A / A_convex ; eccentricity = c / a of the fitted ellipse

**GLCM** (P = normalised co-occurrence matrix, angles 0/45/90/135 degrees):
energy = sum P(i,j)^2 ; contrast = sum (i-j)^2 P(i,j) ; homogeneity = sum P(i,j) / (1 + (i-j)^2) ;
correlation = sum (i - mu_i)(j - mu_j) P(i,j) / (sigma_i sigma_j) ; entropy = -sum P(i,j) log2 P(i,j)

**First-order**: mean, variance, skewness = E[(x-mu)^3]/sigma^3, kurtosis = E[(x-mu)^4]/sigma^4 - 3

**Classification**: accuracy = (TP+TN)/(TP+TN+FP+FN) ; sensitivity (recall) = TP/(TP+FN) ; specificity = TN/(TN+FP) ;
precision = TP/(TP+FP) ; F1 = 2 x precision x recall / (precision + recall)
