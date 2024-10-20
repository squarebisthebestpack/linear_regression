import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as mp


# Чтение файла
def reading(filename):
    try:
        data = pd.read_csv(filename, delimiter=',')
    except:
        try:
            data = pd.read_excel(filename)
        except:
            try:
                data = pd.read_table(filename, delimiter=',')
            except:
                print('ошибка чтения файла')
                sys.exit(1)
    return data

# апроксиммация прямой
def linear(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    # коэффициенты прямой y = kx + b
    k = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    b = (sum_y - k * sum_x) / n
    return k, b

# аппроксимация экспоненты
def exponent(x, y):
    ln_y = np.log(y)
    k, ln_a = linear(x, ln_y)
    return ln_a, k

# апроксимация полинома
def polynomial(x, y, degree):
    
    # создание системы уравнений в матричном виде
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)
    
    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i][j] = np.sum(x ** (i + j))
        B[i] = np.sum(y * (x ** i))

    # решение системы линейных уравнений A * coeffs = B для нахождения коэффициентов полинома
    coeffs = gauss(A, B)
    
    return coeffs

# решение систем линейных уравнений методом Гаусса
def gauss(A, B):
    
    n = len(B)

    # прямой ход
    for i in range(n):
        # приведение главного элемента к единице
        pivot = A[i][i]         # главный элемент
        for j in range(i, n):   
            A[i][j] /= pivot    # нормировка строки 
        B[i] /= pivot           # нормировка вектора

        # обнуление элементов ниже главного элемента
        for k in range(i + 1, n):
            factor = A[k][i]                 # множитель
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]  # вычитание нормированной строки
            B[k] -= factor * B[i]

    # Обратный ход метода Гаусса
    x = np.zeros(n) # вектор решения
    for i in range(n - 1, -1, -1):
        x[i] = B[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
    
    return x

# вычисление предсказанных значений
def coeffs_poly(coeffs, x):
    y_pred = np.zeros(len(x))
    for i in range(len(coeffs)):
        y_pred += coeffs[i] * (x ** i)    # значения полинономов для каждого х
    return y_pred


# написание уравнений
def approx_print(flag, degree, coeffs):
    if flag == None or (flag == '-p' and degree is None):
        k, b = coeffs
        print(f'уравнение y={k:.4f}x+{b:.4f}')
    if flag == '-e':
        a, k = coeffs
        print(f'уравнение y = e^({a:.4f} + {k:.4f}x)')
    if flag == '-p' and degree is not None:
        poly = ' + '.join([f'{coeff:.10f}x^{i}' for i, coeff in enumerate(coeffs)])
        print(f'y={poly}')

#  построение графика
def graph(x, y, y_pred):
    mp.scatter(x, y, color='blue')
    mp.plot(x, y_pred, color='red')
    mp.grid(True)
    mp.show()
    

#основная функция с парсингом аргументов, обработкой данных и вызовом функций
def main():
    
    # парсинг аргументов
    if len(sys.argv) < 2:
        print ('напишите запрос в формате python aproc.py <имя файла> <флаг>')
        sys.exit()
    
    filename = sys.argv[1]
    degree = None
    flag = None
    
    if len(sys.argv) == 3:
        flag = sys.argv[2]
    
    if len(sys.argv) > 3:
        flag = sys.argv[2]
        if flag == '-p':
            try: 
                degree = int(str(sys.argv[3]))               
            except:
                print('неверная степень')
                exit(1)
    
        elif flag != '-e':
            print ('неверная запись флага')
            sys.exit(1)

     # обработка данных
    data = reading(filename)

    x=np.array(data.iloc[:,0])
    y=np.array(data.iloc[:, 1])
    
    if len(x) == 0 or len(y) == 0:
        print('не хватает данных в файле')
        sys.exit()
        
   # аппроксимация (вызов функций)
    if (flag == '-p' and degree is None) or flag is None:
        k, b = linear(x, y)
        y_pred = k * x + b
        approx_print(flag, degree, (k,b))
        graph(x, y, y_pred)
    elif flag == '-e':
        a, k = exponent(x, y)
        y_pred = np.exp(a + k * x)
        approx_print(flag, degree, (a, k))
        graph(x, y, y_pred)
    elif flag == '-p' and degree is not None:
        coeffs = polynomial(x, y, degree)
        y_pred = coeffs_poly(coeffs, x)
        approx_print(flag, degree, coeffs)
        graph(x, y, y_pred)


main()