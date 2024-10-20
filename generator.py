import random

arr = ['xy']

# Создаем список из значений
def num ():
    
    for i in range(1, 51):
        coef = i*i + 5
        #print(i, ', ', random.randint(i, coef))
        arr.append(coef)

num()

file_num = '2'
file_form = '.csv'

# Создаем файл со значениями из списка
def file(file_num):
    name =  'data'+ str(file_num) + file_form
    data = open(name, "w+")
    for i in range (len(arr)):
        data.write( ', '.join(map(str, arr[i])) + '\n')
        
    data.close()
    
file(file_num)