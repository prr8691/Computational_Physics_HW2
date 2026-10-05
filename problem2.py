import math


# ---------------------------------------------------------
# Problem 2: Wien's displacement constant
# Solve 5e^(-x) + x - 5 = 0 using binary search
# ---------------------------------------------------------


# Function whose nonzero root we want to find
def f(x):
    return 5 * math.exp(-x) + x - 5


# ---------------------------------------------------------
# Binary search
# ---------------------------------------------------------

# The physical root is between x = 1 and x = 10
x1 = 1.0
x2 = 10.0

# Accuracy required by the problem
accuracy = 1e-6

iterations = 0


# Keep cutting the interval in half until it is small enough
while x2 - x1 > accuracy:

    midpoint = (x1 + x2) / 2

    # Check which half contains the root
    if f(x1) * f(midpoint) < 0:
        x2 = midpoint

    else:
        x1 = midpoint

    iterations += 1


# Best estimate of the root
x = (x1 + x2) / 2


# ---------------------------------------------------------
# Calculate Wien's displacement constant
# ---------------------------------------------------------

# Physical constants
h = 6.62607015e-34       # Planck constant, J s
c = 2.99792458e8         # speed of light, m/s
k_B = 1.380649e-23       # Boltzmann constant, J/K


# Wien's displacement constant:
# b = hc / (k_B x)
b = h * c / (k_B * x)


# ---------------------------------------------------------
# Estimate the temperature of the Sun
# ---------------------------------------------------------

# Peak wavelength of the Sun
wavelength = 502e-9      # meters

# Wien's law: lambda_max * T = b
T_sun = b / wavelength


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("Solution for x =", x)
print("Number of binary search iterations =", iterations)

print()
print("Wien displacement constant =", b, "m K")

print()
print("Estimated surface temperature of the Sun =", T_sun, "K")
