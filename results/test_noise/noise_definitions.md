## Noise Definitions

- Image resolution: 32×32
- sqrt_alpha = 0.701630
- sqrt_one_minus_alpha = 0.712541

### Gaussian (baseline) noise
- Formulation: $x_t = \sqrt{\alpha_t} \, x_0 + \sqrt{1-\alpha_t} \, \varepsilon$, with $\varepsilon \sim \mathcal{N}(0, I)$ sampled i.i.d. per pixel.

### Uniform spectral noise
- For FFT frequency indices $(k, \ell)$, define $r(k, \ell) = \sqrt{(k/H)^2 + (\ell/W)^2}$.
- Let $r_\text{min} = \min_{r>0} r(k, \ell) = 3.125000e-02$. The mask is 
  $m(k, \ell) = \sqrt{\left(\frac{r(k, \ell)}{r_\text{min}}\right)^2 + 1}$ with the DC bin   $m(0,0)$ set to $1.0$.
- The noise FFT is scaled by this mask prior to the inverse FFT so that higher frequencies receive   proportionally more energy while the DC component remains bounded.

### Saved figures
- `noise_gaussian.png`, `corrupted_gaussian.png`
- `noise_uniform.png`, `corrupted_uniform.png`
- `noise_difference_uniform_minus_gaussian.png`, `corrupted_difference_uniform_minus_gaussian.png`