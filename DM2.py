#E.1:
#function to input max, min, array and target value
def finval(arr,mn,mx,val):
    #spliting list into two, looking for value in the middle
    guess=int((mn+mx)/2)
    if (arr[guess]==val):
        print("target found at index", guess)
        # if not found at 'guess', and value at guess is lower than target value increment min
    elif(arr[guess]<val):
        mn=guess+1
        finval(arr,mn,mx,val)
        #if not found at 'guess', and value at guess is higher than target value decrement min
    elif (arr[guess]>val):
        mx=guess-1
        finval(arr,mn,mx,val)
def main():
    num=int(input("Enter length of list:"))
    arr=[]
    for i in range(0,num):
        i=int(input("Enter value to be entered into list:"))
        arr.append(i)
    val1=int(input("Enter the target"))
    mn1=0
    mx1=num-1
    finval(arr,mn1,mx1,val1)
main()
#############
"""CONS:
1. The algorithm works only on ordered lists.
2. The time complexity is high.
"""
##########
#E.2:
str=input("Enter a string:")
count = 0
for i in str:
        count = count + 1

print("The frequency is:", count)

#E.3:
str = input("Enter a string to be tested:")
arr = []
ll = []
t = 0
mx = 0
arr = str.split(' ')
for i in arr:
    ll.append(len(i))
lll = len(ll)
for j in range(0, lll):
    if ll[j] > mx:
        mx = ll[j]
        t = j

print("The longest string is:", arr[t])

#E.4:

ll = [7, 10, 1, 3, 9, 5, 0]
min = 99
listlen = len(ll)
for i in range(0, listlen):
    for j in range(0, listlen):
        if (ll[i] < min):
            min = ll[i]

print(min)
#E.5:

def same(l1, l2):
    i = len(l1)
    j = len(l2)
    c = 0
    for ii in range(0, i):
        for jj in range(0, j):
            if l1[jj] == l2[ii]:
                c = c + 1
            else:
                c = c + 0
    if (c > 0):
        return True
    else:
        return False


def main():
    lst1 = []
    lst2 = []
    n1 = int(input("Enter number of elements for list1 : "))
    for i in range(0, n1):
        val1 = int(input())
        lst1.append(val1)
    n2 = int(input("Enter number of elements for list2 : "))
    for i in range(0, n2):
        val2 = int(input())
        lst2.append(val2)
    cal = same(lst1, lst2)
    print(cal)


main()
#####################

#E.6:

d1={1:"Anne", 2: "Jack"}
d2={4:"Chris", 5:"Rebecca", 6:"Alice",7:"Nathan"}
d1.update(d2)
print(d1)
#####################
#E.7:
keys = [1,2,3]
values = ['Anne','Jack', 'Carrie','Levi']
names = dict(zip(keys, values))
print(names)
#Class_Ex1:
from random import randrange
count = randrange(1,6)
print(' _______\n|'," ",count  ,'\t|\n|_______|')
###############################
#Class_Ex2:
def fun(count):
    print(' _______\n|'," ",count  ,'\t|\n|_______|')
def main():
    count1=int(input("Enter the value"))
    if (count1<=6 and count1>0):
        fun(count1)
    elif(count1>6 or count1<1):
        print("Enter the right value(between 1-6)")
main()
###################
#Class_Ex3:
from random import randrange
num=int(input("Enter length of list:"))
arr=[]
for i in range(0,num):
    i=int(input("Enter value to be entered into list:"))
    arr.append(i)
r1=randrange(0,num)
print(arr[r1])
# Class_Ex4:
def convertTuple(tup):
    str =  ''.join(tup)
    return str
tuple = input("Enter a string:")
str = convertTuple(tuple)
print(str)
#########################
#Class_Ex5:
a=(1,2,3,4,5,6)
print(a[2])
print(a[-3])
####################
#Class_Ex6:
tuplex = ("w", 3, "r", "e", "s", "o", "u", "r", "c", "e")
print("r" in tuplex)
print(5 in tuplex)
####################
#Class_Ex7:
def fun(ll):
    if (len(ll) > 0):
        print("not empty")
    else:
        print("empty")


def main():
    l1 = []
    l2 = [1, 2, 3, 4]
    fun(l1)
    fun(l2)


main()
#Class_Ex8:
a=4
b=5
c=3
lst = [[ ['0' for col in range(a)] for col in range(b)] for row in range(c)]
print(lst)
