## Одномерная оптимизация
Пусть дана функция (оракул) f
Со следующим интерфейсом
```
def f(x):
     return f(x), f'(x)
```
Например, оракул для квадратичной фунции x^2/2
```
def f(x):
    return x * x / 2, x
```

Требуется реализовать метод: который будет находить минимум функции на отрезке [a,b]
```
def optimize(f, a: float, b: float, eps: float = 1e-8):
    pass
```
Задание состоит из 2-х частей— реализовать любой алгоритм оптимизации по выбору
Провести анализ работы алгоритма на нескольких функция, построить графики сходимости вида: кол-во итераций vs log(точность); время работы vs log(точность)
Изучить, как метод будет работать на неунимодальных функций и привести примеры, подтверждающие поведение (например, что будет сходится в ближайший локальный минимум)