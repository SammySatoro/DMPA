require 'matrix'


def gaussian_kernel(size, sigma)
  Matrix.build(size) do |x, y|
    (1 / (2 * Math::PI * sigma ** 2)) *
      Math.exp(-((x - (size-1)/2)**2 + (y - (size-1)/2)**2) / (2 * sigma**2))
  end
  
end

def normalize_kernel(kernel)
  kernel / kernel.to_a.flatten.reduce(:+)
end

kernel_size = 5
sigma = 1
gauss_kernel = gaussian_kernel(kernel_size, sigma)
norm_gauss_kernel = normalize_kernel(gauss_kernel)
print(gauss_kernel)
print("\n")
print(gauss_kernel.to_a.flatten.reduce(:+))
print("\n")
print(norm_gauss_kernel)
print("\n")
print(norm_gauss_kernel.to_a.flatten.reduce(:+))


