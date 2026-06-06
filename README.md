# Numerical Methods Solver

A Python desktop application for solving numerical methods problems using a graphical user interface. This project includes root-finding methods and an ordinary differential equation solver. The app allows users to enter equations, input starting values, view iteration tables, and see graph visualizations of the results.

## Methods Included

### Root-Finding Methods

- Bisection Method
- Newton's Method
- Secant Method

### Ordinary Differential Equation Method

- Euler's Method

## Features

- Left-side method selector
- Input fields for equations and required values
- Iteration table with aligned values
- Graph visualization
- Status indicator showing convergence or errors
- Separate interface for each numerical method

## Requirements

Install the required Python libraries:

```powershell
py -m pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
numpy
sympy
matplotlib
```

## How To Run

Run the application using:

```powershell
py numerical_methods_solver.py
```

## Equation Format

For Bisection, Newton, and Secant methods, use `x` as the variable.

Examples:

```text
x^3 - x - 2
x^2 - 4
sin(x)
cos(x) - x
```

For Euler's Method, use both `x` and `y`.

Examples:

```text
x + y
x - y
sin(x) + y
```

Powers can be written using `^`.

Example:

```text
x^3 - x - 2
```

## Method Explanations

### Bisection Method

The Bisection Method is a root-finding method that works by repeatedly dividing an interval into two equal parts. It requires two starting values, usually called `a` and `b`.

For this method to work, the function values at the endpoints must have opposite signs:

```text
f(a) * f(b) < 0
```

If the signs are opposite, the root is somewhere between `a` and `b`. The method finds the midpoint:

```text
c = (a + b) / 2
```

Then it checks which side contains the root. The interval is updated again and again until the answer becomes accurate enough.

This method is simple and reliable, but it may take more iterations compared to other methods.

### Newton's Method

Newton's Method is a root-finding method that uses one initial guess, usually called `x0`. It also uses the derivative of the function.

The formula is:

```text
x_next = x - f(x) / f'(x)
```

The method starts from the initial guess and follows the tangent line of the function to estimate the next value. This repeats until the value gets close to the root.

Newton's Method can be very fast when the starting guess is close to the actual root. However, it may fail if the starting guess is poor or if the derivative is too close to zero.

### Secant Method

The Secant Method is similar to Newton's Method, but it does not require the derivative. Instead, it uses two starting values, usually called `x0` and `x1`.

The formula is:

```text
x_next = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
```

The method draws a secant line through two points on the curve and uses that line to estimate the next root value.

This method is useful when finding the derivative is difficult. It is often faster than the Bisection Method, but it may be less stable because it does not always guarantee convergence.

### Euler's Method

Euler's Method is used to approximate solutions of ordinary differential equations written in the form:

```text
dy/dx = f(x, y)
```

It starts with an initial point:

```text
x0, y0
```

Then it moves forward using a step size `h`.

The formula is:

```text
y_next = y + h * f(x, y)
x_next = x + h
```

The value of `f(x, y)` represents the slope at the current point. Euler's Method uses that slope to estimate the next point.

This method is easy to understand and useful for approximations, but smaller step sizes usually give more accurate results.

## Output

Each method displays an iteration table showing the step-by-step process. The table helps users understand how the final answer was obtained.

The graph shows a visual representation of the function or approximation. For root-finding methods, the graph shows the function and root estimate. For Euler's Method, the graph shows the approximate solution curve.