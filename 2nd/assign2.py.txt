import copy
from pickle import NONE
import time
open = []
close = []

t1=time.perf_counter()   
target_state=[[1, 2, 3],
              [4, 5, 6],
              [7, 8, -1]]


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
def element(ele,x):
    for i in range(3):
        for j in range(3):
            if(ele[i][j] == x):
                return (i,j)  

visited={}
class state():
    def __init__(self, value, lvl, fx, parent) :
        self.value = value
        self.lvl = lvl
        self.fx = fx
        self.parent=parent

        
    def next_state(self):
        x , y = findblank(self.value)
        children = []
        if(y<2):
                temp2 = copy.deepcopy(self.value)
                z = temp2[x][y] 
                temp2[x][y] = temp2[x][y+1] 
                temp2[x][y+1] = z
                child = state(temp2, self.lvl+1, 0,self)
                if child not in visited.keys():
                    visited.update({child: 1})
                    children.append(child)
                
        #left check
        if(y>0):
            temp2 = copy.deepcopy(self.value)
            z = temp2[x][y] 
            temp2[x][y] = temp2[x][y-1] 
            temp2[x][y-1] = z
            child = state(temp2, self.lvl+1, 0,self)
            if child not in visited.keys():
                visited.update({child: 1})
                children.append(child)
                
        #up check
        if(x>0):
            temp2 = copy.deepcopy(self.value)
            z = temp2[x][y] 
            temp2[x][y] = temp2[x-1][y] 
            temp2[x-1][y] = z
            child = state(temp2, self.lvl+1, 0,self)
            if child not in visited.keys():
                    visited.update({child: 1})
                    children.append(child)
        #down check
        if(x<2):
            temp2 = copy.deepcopy(self.value)
            z = temp2[x][y] 
            temp2[x][y] = temp2[x+1][y] 
            temp2[x+1][y] = z
            child = state(temp2, self.lvl+1, 0,self)
            if child not in visited.keys():
                    visited.update({child: 1})
                    children.append(child)
                
        return children

def f1x(initial_state,target_state):
    return initial_state.lvl

def f2x(initial_state,target_state):
    return h2x(initial_state.value,target_state) + initial_state.lvl
    
def f3x(initial_state,target_state):
    return h3x(initial_state.value,target_state) + initial_state.lvl   
  
def f4x(initial_state,target_state):
    return h4x(initial_state.value,target_state) + initial_state.lvl

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
    
 
def h4x(initial_state,target_state):
    order=len(initial_state)
    l=[[0,1,2],[7,8,3],[6,5,4]]
    score=0
    for i in range(0, order):
            for j in range(0, order):
                if(initial_state[i][j]!=-1):
                    squareNo = l[i][j]
                    m,n=element(target_state,initial_state[i][j])
                    score= score + abs(i-m) + abs(j-n)
                    num = initial_state[i][j]+1
                    m=0
                    o=0
                    for m in range(0, order):
                        for o in range(0, order):
                            if initial_state[m][o]==num:
                                break
                    if(squareNo==8):
                        score+=3*1
                    elif(l[m][o] != ((squareNo+1)%8)):
                        score+=3*2
    return score

 

# driver code
print("Enter Initial State : ")
initial_s = []

for i in range(1, 4):
    initial_s.append(list(map(int, input().split())))



print("\n-------Initial State-------\n")
print_state(initial_s)
print("-------Target State-------\n")
print_state(target_state)
#print(h3x(initial_state,target_state))

if((isSolvable(initial_s)) == False):
    print("\nproblem cannot be solved")
    print("Number of steps before termination = 362880")
    exit(1)

initial_state = state(initial_s,0,0,None)
visited[initial_state]=1
print("\n--------Choose heuristic : -------\n")
print("1. Depth of node\n")
print("2. Missplaced Tiles\n")
print("3. Manhattan Distance\n")
print("4. Nilsson's sequence score\n ")

choice=int(input())
if choice==1 : 
    initial_state.fx = f1x(initial_state,target_state)
elif choice==2 : 
    initial_state.fx = f2x(initial_state,target_state)
elif choice==3 :
    initial_state.fx = f3x(initial_state,target_state)
elif choice==4 :
    initial_state.fx = f4x(initial_state,target_state)

""" Putting the initial_state in  open"""
open.append(initial_state)

print("\n\n")

count = 0
while True:
    cur = open[0]
    if(len(open)==0):break
    
    if cur.value not in close:

        if cur.value==target_state :
            print("Length of Optimal path : ",cur.lvl)
            l=[]
            while(cur.parent!=None):
                l.append(cur.value)
                cur=cur.parent
            l.append(cur.value)
            l.reverse()
            for x in l:
                    for y in x:
                        print("")
                        print(y,end=" ")
                    print("")
            break
        
        for i in cur.next_state():
            if choice==1:
                i.fx = f1x(i, target_state)
                open.append(i)
            if choice==2:
                i.fx = f2x(i, target_state)
                open.append(i)
            if choice==3:
                i.fx = f3x(i, target_state)
                open.append(i)
            if choice==4:
                i.fx = f4x(i, target_state)
                open.append(i)
            
            
        close.append(cur.value)
        count += 1
    del open[0]
    #print(count)
    """ sort the open list based on fx """
    open.sort(key = lambda x:x.fx)
    t2=time.perf_counter()
    if (t2-t1)/360>1 :
        print("Excided time limit")
        break          
print("No. of steps explored = " + str(count))

print("Total execution time in minute : \n ",(t2-t1)/360)            