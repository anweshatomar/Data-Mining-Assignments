##E5.Write a Python class to find sum the three elements of the given array to zero.
class calc:
    def __init__(self):
        arr = [-20, -10, -6, -4, 3, 4, 7, 10]
        ll = []
        fl = []
        tt = []
        for i in arr:
            for j in arr:
                for k in arr:
                    if ((i + j + k) == 0):
                        ll.append(i)
                        ll.append(j)
                        ll.append(k)
                        fl.append(ll)
                        ll = []
        for i in fl:
            i.sort()
        print(fl)
        print()
        for j in fl:
            for k in fl:
                if j == k:
                    fl.remove(k)

        print(fl)



calc()