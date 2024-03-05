import copy
from pickle import NONE
import time

t1 = time.perf_counter() 

# Step limit is used when algorithm is stuck in a flat or shoulder to avoid infinite loop
maxstep = 200

# To store neighbouring states, for checking heuristic value
open = []

# To store visited states to avoid loop
visited={}

target_state=[[1, 2, 3],
              [4, 5, 6],
              [7, 8, -1]]

# Checking solvability of problem
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


def print_state(x):
    for i in x:
        print(i)
    print("\n")
    
def findblank(temp):
    for i in range(3):
        for j in range(3):
            if(temp[i][j] == -1):
                return (i,j)

class state():
    def __init__(self, value,hx) :
        self.value = value
        self.hx = hx

        
    def next_state(self):
        x , y = findblank(self.value)
        
        children = []
        
        if(y<2):
                temp2 = copy.deepcopy(self.value)
                z = temp2[x][y] 
                temp2[x][y] = temp2[x][y+1] 
                temp2[x][y+1] = z
                child = state(temp2,0)
                if child not in visited.keys():
                    children.append(child)
                
        #left check
        if(y>0):
            temp2 = copy.deepcopy(self.value)
            z = temp2[x][y] 
            temp2[x][y] = temp2[x][y-1] 
            temp2[x][y-1] = z
            child = state(temp2,0)
            if child not in visited.keys():
                children.append(child)
                
        #up check
        if(x>0):
            temp2 = copy.deepcopy(self.value)
            z = temp2[x][y] 
            temp2[x][y] = temp2[x-1][y] 
            temp2[x-1][y] = z
            child = state(temp2,0)
            if child not in visited.keys():
                    children.append(child)

        #down check
        if(x<2):
            temp2 = copy.deepcopy(self.value)
            z = temp2[x][y] 
            temp2[x][y] = temp2[x+1][y] 
            temp2[x+1][y] = z
            child = state(temp2,0)
            if child not in visited.keys():
                    children.append(child)
                
        return children 

def h2x(initial_state,target_state):
    h2 = 0
    for i in range(3):
        for j in range(3):
            if(initial_state[i][j] != -1 and initial_state[i][j] != target_state[i][j]) :
                h2 += 1
    return h2

def h3x(initial_state,target_state):
    h3 = 0
    for i in range(3):
        for j in range(3):
            if(initial_state[i][j] != -1) :
                v = (initial_state[i][j]-1)
                x,y = v//3,v%3
                h3 += (abs(i-x)+abs(j-y))
    return h3
    

# Main function, Hill Climb Search is implemented 
print("Enter Initial State : ")
initial_s = []

for i in range(1, 4):
    initial_s.append(list(map(int, input().split())))



print("\n-------Initial State-------\n")
print_state(initial_s)
print("-------Target State-------\n")
print_state(target_state)

if((isSolvable(initial_s)) == False):
    print("\nproblem cannot be solved")
    print("No. of steps explored = " + str(0) + "\n")
    exit(1)

initial_state = state(initial_s, 0)
visited[initial_state] = 1

print("\n--------Choose huristic : -------\n")
print("1. Missplaced Tiles\n")
print("2. Manhattan Distance\n")


choice=int(input())
if choice==1 : 
    initial_state.hx = h2x(initial_state.value,target_state)
elif choice==2 : 
    initial_state.hx = h3x(initial_state.value,target_state)


#h = h2x(initial_state.value, target_state)
h = initial_state.hx
count = 0
steps = 0 
cur = initial_state
print_state(initial_s)

while True:
    visited.update({cur: 1})
    if (cur.value == target_state) :
        print("Target Reached\n")
        print("Length of Optimal path : ",count)
        break
    
    count += 1    
    for i in cur.next_state():
        if choice==1:
            i.hx = h2x(i.value, target_state)
            open.append(i)
        if choice==2:
            i.hx = h3x(i.value, target_state)
            open.append(i)
        """i.hx = h2x(i.value, target_state)
        open.append(i)"""
    
    open.sort(key = lambda x:x.hx)
    
    if(open[0].hx > h):
        print("Local maxima reached\n")
        #print_state(open[0].value)
        break
    
    if(open[0].hx < h):
        h = open[0].hx
        cur = open[0]
        steps = 0 
        print_state(open[0].value)
    
    elif(steps > maxstep):
        print("Stuck in Flat or Shoulder\n")
        print_state(open[0].value)
        break
    else :
        h = open[0].hx
        cur = open[0]
        steps += 1
        print_state(open[0].value)
    
    open.clear()
        
    t2=time.perf_counter()
    if (t2-t1)/360>1 :
        print("Exceded time limit")
        break  

print("No. of steps explored = " + str(count))

print("Total execution time in minute : \n ",(t2-t1)/360)
    
    



