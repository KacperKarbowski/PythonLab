a, b, c = map(int, input("Podaj trzy liczby oddzielone spacją: ").split())

if a > b:
    if a > c:
        print(a)
    else:
        print(c)
else:
    if b < c:
        print(b)
    else:
        print(c)