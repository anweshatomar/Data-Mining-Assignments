# =================================================================
# Class_Ex1:
# Write python program that converts seconds to
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------
t = int(input("Enter the time in seconds"))
h,s,m=0,0,0
v1=t/60
v2=v1/60
iv2=int(v2)
dv2=v2-iv2
dm=dv2*60
ddm=dm-int(dm)
m=int(dv2*60)
s=int(ddm*60)
h=iv2
print(h , "hours," , m , "min," , s, "sec")

# =================================================================
# Class_Ex2:
# Write a python program to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# ----------------------------------------------------------------
ll=['A','B','C']
for i in range(0,3):
    for j in range (0,3):
        for k in range(0,3):
            if (ll[i]!=ll[j] and ll[j]!=ll[k] and ll[k]!=ll[i]):
                print (ll[i],ll[j],ll[k])


# =================================================================
# Class_Ex3:
# Write a python program to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------
ll=['A','B','C','D']
for i in range(0,4):
    for j in range (0,4):
        for k in range(0,4):
            for m in range(0,4):
                if (ll[i]!=ll[j] and ll[j]!=ll[k] and ll[k]!=ll[i] and ll[m]!=ll[i] and ll[m]!=ll[j] and ll[m]!=ll[k]):
                    print (ll[i],ll[j],ll[k],ll[m])

# =================================================================
# Class_Ex4:
# Suppose we wish to draw a triangular tree, and its height is provided
# by the user.
# ----------------------------------------------------------------

tree=int(input("Enter the depth of the tree:"))
space = 2*tree - 2
for i in range(0, tree):
    for j in range(0, space):
            print(end=" ")
    space = space - 1
    for j in range(0, i+1):
        print("* ", end="")
    print(" ")

# =================================================================
# Class_Ex5:
# Write python program to print prime numbers up to a specified values.
# ----------------------------------------------------------------
limit = int(input("Enter the number of prime numbers:"))

for val in range(0, limit + 1):
    if val > 1:
        for n in range(2, val):
            if (val % n) == 0:
                break
        else:
            print(val)
# =================================================================