import math
a = float(input("Podaj współczynnik a: "))
b = float(input("Podaj współczynnik b: "))
c = float(input("Podaj współczynnik c: "))
delta = b**2 - 4*a*c
if delta < 0:
    print("Brak miejsc zerowych")
elif delta == 0:
    x0 = -b / (2 * a)
    print("Jedno miejsce zerowe:", x0)
else:
    x1 = (-b - math.sqrt(delta)) / (2 * a)
    x2 = (-b + math.sqrt(delta)) / (2 * a)
    print("Dwa miejsca zerowe:", x1, x2)