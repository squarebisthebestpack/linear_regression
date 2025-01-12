import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# чтение файла
def reading(filename):
    try:
        data = pd.read_csv(filename, delimiter=',')
    except Exception:
        try:
            data = pd.read_excel(filename)
        except Exception:
            try:
                data = pd.read_table(filename, delimiter=',')
            except Exception:
                print('Ошибка чтения файла')
                sys.exit(1)
    return data


def linear(X, y):
    ones_likeX = np.ones((X.shape[0],1))
    X_with1 = np.hstack((ones_likeX, X))
    
    X_with1_T = X_with1.T 
    abc = np.linalg.inv(np.dot(X_with1_T, X_with1))
    B = np.dot(np.dot(abc, X_with1_T), y)
    return B

def exponent(X, y):
    if np.any(y <= 0):
        print('нельзя неположительные значения для экспоненциальной регрессии')
        sys.exit(1)
    ln_y = np.log(y)
    
    B = linear(X, ln_y)
    
    return  B

def polynomial(X, y, degree):
    X_temp = X
    for i in range(2, degree + 1):
        X_temp = np.hstack((X_temp, X**i))
        
    ones_likeX = np.ones((X.shape[0],1))
    X_pol = np.hstack((ones_likeX, X_temp))

    abc = np.linalg.inv(np.dot(X_pol.T, X_pol))
    B = np.dot(np.dot(abc, X_pol.T), y)
        
    return B

def predict(X, B, flag, degree):
    
    ones_likeX = np.ones((X.shape[0],1))
    
    if flag == '-p' and degree == None:
        
        X_with1 = np.hstack((ones_likeX, X))
        return np.dot(X_with1, B)
    elif flag == '-e':
        
        X_with1 = np.hstack((ones_likeX, X))
       
        return np.exp(np.dot(X_with1, B))
    elif flag == '-p':
        
        X_temp = X
        for i in range(2, degree + 1):
        
            X_temp = np.hstack((X_temp, X**i))
        
        X_pol = np.hstack((ones_likeX, X_temp))

        return np.dot(X_pol, B)

    else:
        raise ValueError('неверный флаг или степень')


# написание уравнений

def approx_print(flag, degree, coeffs):
    if flag is None or flag == '-p':
        equation = ' + '.join([f'{coeff:.4f}x{i}' for i, coeff in enumerate(coeffs[1:], start=1)])
        print(f'Уравнение: y = {coeffs[0]:.4f} + {equation}')
    elif flag == '-e':
        equation = ' + '.join([f'{coeff:.4f}x{i}' for i, coeff in enumerate(coeffs[1:], start=1)])
        print(f'Уравнение: y = exp({coeffs[0]:.4f} + {equation})')
    elif flag == '-p' and degree is not None:
        equation = ' + '.join([f'{coeff:.4f}x^{i}' for i, coeff in enumerate(coeffs[1:], start=1)])
        print(f'Уравнение: y = {coeffs[0]:.4f} + {equation}')


def graph(X, y, pred):
    if X.shape[1] == 1:
        plt.scatter(X[:,0], y, color = 'red', label = 'исходные данные')
        plt.plot(X[:,0], pred, color = 'blue', label = 'результат')
        plt.grid(True)
        plt.show()
    else: 
        print('график работает со странностями, так как предполагается использование 3д')
        plt.scatter(X[:,0], y, color = 'red', label = 'исходные данные')
        plt.plot(X[:,0], pred, color = 'blue', label = 'результат')
        plt.grid(True)
        plt.show()




def main():
    if len(sys.argv) < 2:
        print('запрос в виде python aproc.py <имя файла> <флаг> [степень для полинома]')
        sys.exit(1)

    filename = sys.argv[1]
    flag = None
    degree = None

    if len(sys.argv) >= 3:
        flag = sys.argv[2]
    if len(sys.argv) == 4 and flag == '-p':
        try:
            degree = int(sys.argv[3])
        except ValueError:
            print('ошибка: степень полинома должна быть числом')
            sys.exit(1)

    data = reading(filename)
    
    X = np.array(data.iloc[ :, :-1])
    y = np.array(data.iloc[ :, -1])
    
    if X.size == 0 or y.size == 0:
        print('ожидаются не пустые данные')
        sys.exit(1)
    
    
    if (flag == '-p' and degree is None) or flag is None:
        
        B = linear(X, y)
        pred = predict(X, B, '-p', None)
        approx_print(flag, degree, B)
        graph(X, y, pred)
        
    elif flag == '-e':
        
        B = exponent(X, y)
        pred = predict(X, B, '-e', None)
        approx_print(flag, degree, B)
        graph(X, y, pred)
        
    elif flag == '-p' and degree is not None:
        
        B = polynomial(X, y, degree)
        pred = predict(X, B, '-p', degree)
        approx_print(flag, degree, B)
        graph(X, y, pred)
    
    
    
if __name__ == "__main__":
    main()