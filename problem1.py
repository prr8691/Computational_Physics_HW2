import math


# ---------------------------------------------------------
# Problem 1: Overrelaxation
# Solve x = 1 - exp(-cx) for c = 2
# ---------------------------------------------------------

c = 2.0
tolerance = 1e-6


# Function f(x) in the equation x = f(x)
def f(x):
    return 1 - math.exp(-c * x)


# Derivative of f(x)
# This is used in the error estimate
def f_prime(x):
    return c * math.exp(-c * x)


# ---------------------------------------------------------
# Relaxation / overrelaxation method
# ---------------------------------------------------------

def solve(omega):
    """
    Solve x = 1 - exp(-2x).

    omega = 0 gives ordinary relaxation.
    omega > 0 gives overrelaxation.
    """

    # Starting guess
    x = 1.0
    iterations = 0

    while True:

        # Overrelaxation formula
        x_new = (1 + omega) * f(x) - omega * x

        # Derivative of the overrelaxation function
        q = (1 + omega) * f_prime(x) - omega

        # Error estimate from the overrelaxation formula
        error = abs(
            (x - x_new) /
            (1 - 1 / q)
        )

        iterations += 1

        # Stop when the error is small enough
        if error < tolerance:
            break

        # Use the new value for the next iteration
        x = x_new

    return x_new, iterations


# ---------------------------------------------------------
# Ordinary relaxation
# ---------------------------------------------------------

solution, steps = solve(0.0)

print("Ordinary relaxation")
print("Solution =", solution)
print("Iterations =", steps)


# ---------------------------------------------------------
# Try several overrelaxation values
# ---------------------------------------------------------

omega_values = [
    0.2,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

print()
print("Overrelaxation results")

for omega in omega_values:

    solution, steps = solve(omega)

    print(
        "omega =", omega,
        " solution =", solution,
        " iterations =", steps
    )
