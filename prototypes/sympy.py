from sympy import linsolve, nonlinsolve, symbols

# Testing that at least one system works


i11, o11, i21, i22, i23, o21, o22 = symbols(
    'i11, o11, i21, i22, i23, o21, o22',
    real=True
)
variables = [
   i11, o11, i21, i22, i23, o21, o22 
]

system = [
    o11 - 50*i11,
    i21 - o11,
    o21 - i21,
    o22 - i21/6,
    i22 - i21/6,
    i23 - i21/54000,
    o21 - 200
]

res = linsolve(system, variables) # Can replace with nonlinsolve
print(res)