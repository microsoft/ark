import ark

ark.init()

a = ark.Dims([1, 2, 3, 4])
print(ark.NO_DIM)
print(a[2])

ark.srand(42)  
  
random_number = ark.rand()  

print(random_number)

print("ark test success")
