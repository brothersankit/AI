import os
import copy
import random
import time
import math
import sys

from pyparsing import Char


def isSolvable(a):
    arr=[]
    for i in range(3):
        for j in range(3):
            arr.append(a[i][j])
    inv_count = 0
    empty_value = -1
    for i in range(0, 9):
        for j in range(i + 1, 9):
            if arr[j] != empty_value and arr[i] != empty_value and arr[i] > arr[j]:
                inv_count += 1
    if(inv_count % 2):
        return False
    else:
        return True

#Functon to print state
def print_state(x):
    for i in x:
        print(i)
    print("\n")

#Functon to print state
def file_print_state(x,fout):
    for i in x:
        fout.write(str(i))
        fout.write("\n")
        
# Function to find possition of blank tile   
def findblank(temp):
    for i in range(3):
        for j in range(3):
            if(temp[i][j] == -1):
                return (i,j)

# We have used a monotonic decreasing cooling function
def linear_coolingFunction(x):
    Temp=10000-x/10
    return Temp

def exponential_coolingFunction(x):
    if x<1000 : 
        Temp=200* math.exp(-0.005* x)
        return Temp
    else:
        return 0
    

def simulated_Annealing(curr,huristic,coolingFuncton,fout):
    
    count=0
    fout.write("\n------Path------- \n")
    for i in range(sys.maxsize):
        print_state(curr.value)
        file_print_state(curr.value,fout)
        fout.write("\n------------------------\n")
        print("depth = ",i)

        if (coolingFuncton == 1) :
            Temp=linear_coolingFunction(i)
        if (coolingFuncton == 2) :
            Temp=exponential_coolingFunction(i)

        print("Temperature value : ",Temp)
        if(Temp==0 or curr.value==target_state):
            return curr,count  

        #Generating and selecting random neighbour
        neighbour=curr.successor()
        rand_ind=int(random.random() * len(neighbour))
        next = neighbour[rand_ind]
     
        #Calculating huristic for the neighbours
        if huristic==1 :
            next.hx = h1x(next.value,target_state)
        elif huristic==2 : 
            next.hx = h2x(next.value,target_state)
        elif huristic==3 :
            next.hx = h1x(next.value,target_state) * h2x(next.value,target_state)

        count=count+1

        #Calculate Energy difference
        E_diff = curr.hx-next.hx
     
        if E_diff > 0:
            curr=next
        else :
            p=random.random()
            #prob=1/(1+math.exp(-(E_diff)/Temp))
            prob=math.exp(E_diff/Temp)
            print("p : ",p)
            print("Probability: ",prob)
            print("delta E : ",E_diff)
            if p<prob :
                curr=next

        
            
#visited={}
class state():
    def __init__(self, value,hx) :
        self.value = value
        self.hx = hx

     # Neighbourhood generating function to generate all the neighbours 
    def successor(self):
        x , y = findblank(self.value)
       # visited.update({self:1})
        children = []
        #right check
        if(y<2):
                temp2 = copy.deepcopy(self.value)
                z = temp2[x][y] 
                temp2[x][y] = temp2[x][y+1] 
                temp2[x][y+1] = z
                child = state(temp2,0)
                children.append(child)
                
        #left check
        if(y>0):
                temp2 = copy.deepcopy(self.value)
                z = temp2[x][y] 
                temp2[x][y] = temp2[x][y-1] 
                temp2[x][y-1] = z
                child = state(temp2,0)
                children.append(child)
                
        #up check
        if(x>0):
                temp2 = copy.deepcopy(self.value)
                z = temp2[x][y] 
                temp2[x][y] = temp2[x-1][y] 
                temp2[x-1][y] = z
                child = state(temp2,0)
                children.append(child)
        #down check
        if(x<2):
                temp2 = copy.deepcopy(self.value)
                z = temp2[x][y] 
                temp2[x][y] = temp2[x+1][y] 
                temp2[x+1][y] = z
                child = state(temp2,0)
                children.append(child)
                
        return children 

# Huristic to find number of missplaced tiles
def h1x(initial_state,target_state):
    h2 = 0
    for i in range(3):
        for j in range(3):
            if(initial_state[i][j] != -1 and initial_state[i][j] != target_state[i][j]) :
                h2 += 1
    return h2

# Huristic to find Manhatten distance
def h2x(initial_state,target_state):
    h3 = 0
    x1,y1=findblank(initial_state)
    x2,y2=findblank(target_state)
    h3= abs(x1-x2) + abs(y1-y2)
    return h3
    


#main/driver function 
initial_s = []
os.chdir("C:\\RC\\Documents\\IIT Patna Remastered 2023-25\\Course\\1st Semester\\CS561 Artificial Intelligence\\Assignments\\Assignment 5 Simulated Annealing\\Misc")
# print( os.getcwd() )
with open("input.txt") as f:
        for line in f:
            line = line.strip()
            initial_s.append(list(map(int, line.split())))

target_state = []            
with open("target.txt") as f:
        for line in f:
            line = line.strip()
            target_state.append(list(map(int, line.split())))


fout = open("Output.txt","w")


print("\n-------Initial State-------\n")
fout.write("\n-------Initial State-------\n")
print_state(initial_s)
file_print_state(initial_s,fout)
print("-------Target State-------\n")
fout.write("-------Target State-------\n")
print_state(target_state)
file_print_state(target_state,fout)

if((isSolvable(initial_s)) == False):
    print("\nproblem cannot be solved")
    fout.write("\nproblem cannot be solved")
    exit(1)


#initialize the start sate 
initial_state = state(initial_s, 0)


print("\n--------Choose huristic : -------\n")
print("1. Missplaced Tiles\n")
print("2. Manhattan Distance\n")
# print("3. h3=h1*h2\n")

choice=int(input())
if choice==1 : 
    initial_state.hx = h1x(initial_state.value,target_state)
elif choice==2 : 
    initial_state.hx = h2x(initial_state.value,target_state)
elif choice==3 :
    initial_state.hx = h1x(initial_state.value,target_state) * h2x(initial_state.value,target_state)

print("\n--------Choose cooling function: -------\n")
print("1. linear\n")
print("2. Exponential\n")

coolingFunction=int(input())



#Call simulated Annealing algo and calculate running time of it
t1 = time.perf_counter() 
curr,count=simulated_Annealing(initial_state,choice,coolingFunction,fout)
t2 = time.perf_counter()

if choice==1 :
        print("Huristic used : Missplaced tiles\n")
        fout.write("Huristic used : Missplaced tiles\n")
if choice==2 : 
        print("Huristic used : Manhatten distance\n")
        fout.write("Huristic used : Manhatten distance\n")
if choice==3 : 
        print("Huristic used : h1*h2 \n") 
        fout.write("Huristic used : h1*h2 \n") 

print("Initial Temperature = 10000\n")
fout.write("Initial Temperature = 10000\n")

if (coolingFunction == 1) :
    print("Cooling Function = Linear ")
    fout.write("Cooling Function = Linear ")
if (coolingFunction == 2) :
    print("Cooling Function = Exponential\n ") 
    fout.write("Cooling Function = Exponential\n ") 


 #Output when target is reached       
if(curr.value==target_state):
    print("Target Reached\n")
    print("\n-------Initial State-------\n")
    print_state(initial_s)
    print("-------Target State-------\n")
    print_state(target_state)
    fout.write("\nTarget Reached\n")
   

    
#Output when target is not reached and T become 0   
else:
    print("Sub optimal goal:\n")
    fout.write("Sub optimal goal:\n")
    print("\n-------Initial State-------\n")
    print_state(initial_s)
    print("-------Sub optimal Goal-------\n")
    print_state(curr.value)
    file_print_state(curr.value,fout)

print("NUmber of State explored: ",count)
print("Total execution time in minute : ",(t2-t1)/360)
fout.write("NUmber of State explored: ")
fout.write(str(count)+"\n")
fout.write("Total execution time in minute : ")
fout.write(str((t2-t1)/360)+"\n")



    
    



