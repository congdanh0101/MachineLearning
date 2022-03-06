import numpy as np

# exercise1
def divide5(n=list()):
    array = []
    for i in range(0,n.__len__()):
        if n[i]%5 == 0:
            array.append(n[i])
    return array

a = [1,2,5,10,3,4,55]
print(divide5(a))
print()

# exercise2
b = np.array([-2,6,3,10,15,48])
print(b[2:b.size:2])
print(b[1:b.size:2])
print(b[b.size-3:b.size])
print(b[::-1][:3])
print()

# exercise3
def SapXepMang(arr, sapxeptang):
    arr = sorted(arr)
    if sapxeptang == True:
        return arr
    else: 
        arr = arr[::-1] #reverse
        return arr
    
c = np.array([5,6,7,8,9,1,2,3,4,10])
print(SapXepMang(c,False))