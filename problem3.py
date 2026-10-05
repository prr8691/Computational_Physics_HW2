import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Problem 3: Gradient descent and Schechter-function fitting
# ---------------------------------------------------------


# ---------------------------------------------------------
# Load the COSMOS data
# ---------------------------------------------------------

# Column 1 = log10(M_gal)
# Column 2 = n(M_gal)
# Column 3 = error in n(M_gal)

data = np.loadtxt("smf_cosmos.dat")

logM_data = data[:, 0]
n_data = data[:, 1]
error_data = data[:, 2]

# Convert log10(M) into M
M_data = 10**logM_data


# ---------------------------------------------------------
# Numerical gradient
# ---------------------------------------------------------

def numerical_gradient(function, parameters, h=1e-5):
    """
    Calculate the gradient using central differences.
    """

    gradient = np.zeros(len(parameters))

    for i in range(len(parameters)):

        p_plus = parameters.copy()
        p_minus = parameters.copy()

        p_plus[i] += h
        p_minus[i] -= h

        gradient[i] = (
            function(p_plus) - function(p_minus)
        ) / (2 * h)

    return gradient


# ---------------------------------------------------------
# Gradient descent
# ---------------------------------------------------------

def gradient_descent(function, start, step_size=0.1,
                     tolerance=1e-4, max_steps=5000):

    # Starting parameters
    parameters = np.array(start, dtype=float)

    # Store values so they can be plotted later
    values = [function(parameters)]
    path = [parameters.copy()]

    for step in range(max_steps):

        gradient = numerical_gradient(
            function,
            parameters
        )

        gradient_size = np.linalg.norm(gradient)

        # Stop when the gradient is very small
        if gradient_size < tolerance:
            break

        # Direction that moves downhill
        direction = -gradient / gradient_size

        trial_step = step_size
        old_value = values[-1]

        # If the step makes the function larger,
        # reduce the step size and try again
        while trial_step > 1e-10:

            new_parameters = (
                parameters
                + trial_step * direction
            )

            new_value = function(
                new_parameters
            )

            if (
                np.isfinite(new_value)
                and new_value < old_value
            ):
                break

            trial_step = trial_step / 2

        # Stop if no useful step can be found
        if trial_step <= 1e-10:
            break

        parameters = new_parameters

        values.append(new_value)
        path.append(parameters.copy())

    return (
        parameters,
        np.array(values),
        np.array(path)
    )


# =========================================================
# Part 1: Test the gradient-descent method
# =========================================================

def test_function(parameters):
    """
    Simple test function:
    f(x,y) = (x-2)^2 + (y-2)^2
    """

    x = parameters[0]
    y = parameters[1]

    return (x - 2)**2 + (y - 2)**2


# Start at (0,0)
test_best, test_values, test_path = gradient_descent(
    test_function,
    start=[0.0, 0.0],
    step_size=0.2,
    tolerance=1e-8
)


# ---------------------------------------------------------
# Figure 1: Gradient-descent test
# ---------------------------------------------------------

x = np.linspace(-0.5, 3.5, 200)
y = np.linspace(-0.5, 3.5, 200)

X, Y = np.meshgrid(x, y)

Z = (
    (X - 2)**2
    + (Y - 2)**2
)

plt.figure(figsize=(7, 5))

plt.contour(
    X,
    Y,
    Z,
    levels=18
)

plt.plot(
    test_path[:, 0],
    test_path[:, 1],
    "o-",
    markersize=3,
    label="Gradient-descent path"
)

plt.plot(
    2,
    2,
    "x",
    markersize=10,
    label="Exact minimum"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Gradient descent test")

plt.legend()
plt.tight_layout()

plt.savefig(
    "hw2_problem3_test.png",
    dpi=200
)

plt.show()


# =========================================================
# Part 2: Schechter function
# =========================================================

def schechter_model(parameters):
    """
    Calculate the Schechter function.

    The fitted parameters are:
    log10(phi*), log10(M*), alpha
    """

    log_phi = parameters[0]
    log_Mstar = parameters[1]
    alpha = parameters[2]

    phi = 10**log_phi
    Mstar = 10**log_Mstar

    ratio = M_data / Mstar

    model = (
        phi
        * ratio**(alpha + 1)
        * np.exp(-ratio)
        * np.log(10)
    )

    return model


# ---------------------------------------------------------
# Chi-squared
# ---------------------------------------------------------

def chi_squared(parameters):

    model = schechter_model(parameters)

    chi2 = np.sum(
        (
            (n_data - model)
            / error_data
        )**2
    )

    return chi2


# =========================================================
# Part 3: Run from different starting points
# =========================================================

# These are:
# [log10(phi*), log10(M*), alpha]

starting_points = [
    [-2.3, 11.0, -1.2],
    [-3.0, 10.5, -0.5],
    [-2.0, 11.4, -1.5]
]

fit_results = []


# ---------------------------------------------------------
# Figure 2: chi^2 versus step
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

for start in starting_points:

    best, chi_history, path = gradient_descent(
        chi_squared,
        start=start,
        step_size=0.1,
        tolerance=1e-4
    )

    fit_results.append(
        (start, best, chi_history)
    )

    plt.semilogy(
        range(len(chi_history)),
        chi_history,
        label=f"start = {start}"
    )


plt.xlabel("Step i")
plt.ylabel(r"$\chi^2$")
plt.title(r"$\chi^2$ during gradient descent")

plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    "hw2_problem3_chi2.png",
    dpi=200
)

plt.show()


# =========================================================
# Part 4: Best-fit parameters
# =========================================================

# Use the first run for the final plotted fit.
# All three runs give essentially the same answer.

best_parameters = fit_results[0][1]

log_phi_best = best_parameters[0]
log_Mstar_best = best_parameters[1]
alpha_best = best_parameters[2]

phi_best = 10**log_phi_best
Mstar_best = 10**log_Mstar_best

chi2_best = chi_squared(
    best_parameters
)


# ---------------------------------------------------------
# Make a smooth model curve for the plot
# ---------------------------------------------------------

logM_plot = np.linspace(
    logM_data.min(),
    logM_data.max(),
    500
)

M_plot = 10**logM_plot

ratio = M_plot / Mstar_best

model_plot = (
    phi_best
    * ratio**(alpha_best + 1)
    * np.exp(-ratio)
    * np.log(10)
)


# ---------------------------------------------------------
# Figure 3: COSMOS data and best-fit Schechter function
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.errorbar(
    M_data,
    n_data,
    yerr=error_data,
    fmt="o",
    capsize=3,
    label="COSMOS data"
)

plt.plot(
    M_plot,
    model_plot,
    label="Best-fit Schechter function"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$M_{\mathrm{gal}}$")
plt.ylabel(r"$n(M_{\mathrm{gal}})$")

plt.title(
    "Schechter-function fit to COSMOS data"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "hw2_problem3_fit.png",
    dpi=200
)

plt.show()


# =========================================================
# Print the results
# =========================================================

print("Test-function result")
print(
    f"Minimum = "
    f"({test_best[0]:.6f}, "
    f"{test_best[1]:.6f})"
)

print()


# Print all three fits so we can check robustness
for start, best, history in fit_results:

    phi = 10**best[0]
    Mstar = 10**best[1]
    alpha = best[2]

    print("Starting point:", start)

    print(
        f"phi* = {phi:.6e}"
    )

    print(
        f"M* = {Mstar:.6e}"
    )

    print(
        f"alpha = {alpha:.6f}"
    )

    print(
        f"chi^2 = {history[-1]:.6f}"
    )

    print()


# Final values used in the report
print("Best-fit values used in report")

print(
    f"phi* = {phi_best:.4e}"
)

print(
    f"M* = {Mstar_best:.4e}"
)

print(
    f"alpha = {alpha_best:.5f}"
)

print(
    f"chi^2 = {chi2_best:.5f}"
)
