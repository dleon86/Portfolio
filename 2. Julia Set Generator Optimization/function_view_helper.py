    # Use SymPy to construct a function view helper
# Author: DanieL Leon
# Date: 2023-01-13
import sympy
from sympy import *
from sympy.plotting import plot

# Define the function f=1/(1+exp(-x+1))-1/(1+exp(-x-1))-im/(1+exp(-x+im))+im/(1+exp(-x-im))
x = Symbol('x', complex=True)

# f = 5 * exp(x-9) +0.5
f = I/(1+exp(-x+1))-1/(1+exp(-x-1))+1/(1+exp(-x+I))-I/(1+exp(-x-I))
# f = 1/(1+exp(-x+1))-1/(1+exp(-x-1))-0.5/(1+exp(-x+2))+0.5/(1+exp(-x-2))
# f = 1e-12 * x**10

fp = diff(f, x)  # Derivative of f; f'

# display the function and its derivative
print("f(x) = ", f)
print("f'(x) = ", fp)
sympy.pprint(f)
sympy.pprint(fp)

# Plot the real part of the function and the real part of its derivative from -10 to 10
p = plot(re(f), re(fp), (x, -10, 10), show=False)
p[0].line_color = 'b'
p[1].line_color = 'r'
p.show()
# Plot the imaginary part of the function and the imaginary part of its derivative from -10 to 10
p2 = plot(im(f), im(fp), (x, -10, 10), show=False)
p2[0].line_color = 'b'
p2[1].line_color = 'r'
p2.show()










