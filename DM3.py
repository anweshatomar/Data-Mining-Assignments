#E.1:Write a script to find duplicates from an array (define an array with some duplicates on it).
arr=[1,2,3,1,2,3,4]
res = [i for n, i in enumerate(arr) if i  in arr[:n]]
print(res)
# E.2:Write a script to find the second largest number in an array (use random number generator) and multiply it by 50.
from random import randrange

arr = []

for i in range(0, 10):
    i = randrange(0, 10)
    arr.append(i)
arr.sort()

print(arr)
print("Second largest value:", arr[-2])
print("Second largest value multiplied with 50:", arr[-2] * 50)
#E.3:Write a script that finds all such numbers which are divisible by 2 and 5, less than 1000.
arr=[]
for i in range (1,1000):
    if (i%2==0 and i%5==0):
        arr.append(i)
print(arr)
"""
I have attempted question 4 twice: the first one accepts all inputs and prints the conversion, 
the second one does not accept values which are not roman numbers,
but it also rejects values like IV which is a roman number.
"""

# E.4:Write a Python class to convert a roman numeral to an integer.
class roman:
    inp = input("Enter roman neumerals:")
    inps = list(inp)
    rl = ['I', 'V', 'X', 'L', 'C', 'D', 'M']

    count = 0
    for i in inps:
        if (i == rl[0]):
            count = count + 1
        elif (i == rl[1]):
            count = count + 5
        elif (i == rl[2]):
            count = count + 10
        elif (i == rl[3]):
            count = count + 50
        elif (i == rl[4]):
            count = count + 100
        elif (i == rl[5]):
            count = count + 500
        elif (i == rl[6]):
            count = count + 1000
    print(count)


# E.4:Write a Python class to convert a roman numeral to an integer.
def roman(inp, rl):
    inps = list(inp)
    count = 0
    for i in inps:
        if (i == rl[6]):
            count = count + 1
        elif (i == rl[5]):
            count = count + 5
        elif (i == rl[4]):
            count = count + 10
        elif (i == rl[3]):
            count = count + 50
        elif (i == rl[2]):
            count = count + 100
        elif (i == rl[1]):
            count = count + 500
        elif (i == rl[0]):
            count = count + 1000
    print("Integer value is:", count)


def prec(inp, rl):
    inp = list(inp)
    nl = len(inp)
    rll = len(rl)
    tt = []
    flag = 0
    for i in range(0, nl):
        for j in range(0, rll):
            if (inp[i] == rl[j]):
                tt.append(j)
    print(tt)
    ttn = len(tt)
    for k in range(0, ttn - 1):
        if (tt[k] > tt[k + 1]):
            flag = flag + 1
    print(flag)
    return flag


def main():
    inp1 = input("Enter roman neumerals:")
    rl1 = ['M', 'D', 'C', 'L', 'X', 'V', 'I']
    c = prec(inp1, rl1)
    if (c > 0):

        print("wrong input")
    else:
        roman(inp1, rl1)


main()


##E5.Write a Python class to find sum the three elements of the given array to zero.
class calc:
    def __init__(self):
        arr=[-20, -10, -6, -4, 3, 4, 7, 10]
        n=len(arr)
        arr.sort()
        for i in range(0, n-2):
              for j in range(i+1, n-1):
                    for k in range(j+1, n):
                        if (arr[i] + arr[j] + arr[k] == 0):
                            print(arr[i], arr[j], arr[k])
calc()
# Class_Ex1:
# Writes a code to simulate a Stopwatch.
# push a button to start the clock (call the start method), push a button
# to stop the clock (call the stop method), and then read the elapsed time
# (use the result of the elapsed method).
import time


def strt():
    st = time.time()
    return st


def stp():
    sp = time.time()
    return sp


def elapsed(st1, sp1):
    epal = st1 - sp1
    return epal


def main():
    st2 = strt()
    print("The elapsed time is:")
    print("\t")
    sp2 = stp()
    eva = elapsed(sp2, st2)
    print(eva)


main()


# Class_Ex2:
# Write a Python program to implement pow(x, n).
def p(x,n):
    count=1
    for i in range(0,n):
        count=count*x
    print(count)
def main():
    v1=int(input("Enter value one:"))
    v2=int(input("Enter value two:"))
    p(v1,v2)
main()


# Class_Ex3:
# Write a Python class to calculate the area of rectangle by length
# and width and a method which will compute the area of a rectangle.
class rec:
    def __init__(self, l, w):
        self.l = l
        self.w = w

    def area(self):
        return (self.l * self.w)


l1 = int(input("Enter  the value of length:"))
w1 = int(input("Enter  the value of width:"))
ara = rec(l1, w1)
print("area is:", ara.area())

# Class_Ex4:
# Write a Python class and name it Circle to calculate the area of circle
# by a radius and two methods which will compute the area and the perimeter
# of a circle.
import math


class circle:

    def __init__(self, r):
        self.r = r

    def area(self):
        return (((self.r) ** 2) * math.pi)

    def peri(self):
        return (2 * self.r * math.pi)


r1 = int(input("Enter  the value of radius:"))
ara = circle(r1)
print("area is:", ara.area())
print("perimeter is:", ara.peri())
