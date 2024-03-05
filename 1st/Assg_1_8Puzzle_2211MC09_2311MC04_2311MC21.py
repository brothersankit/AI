from queue import Queue, LifoQueue
import copy

# Check if the number of inversion counts in the given grid is even (solvable)
def isSolvable(grid):
    arr = [num for row in grid for num in row if num != -1]
    inv_count = 0
    
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[i]:
                inv_count += 1
    
    return inv_count % 2 == 0

# Print the state of the grid
def print_state(state):
    for row in state:
        print(row)
    print()

# Find the coordinates of the blank space in the puzzle
def find_blank(temp):
    for i in range(3):
        for j in range(3):
            if temp[i][j] == -1:
                return i, j

target_state=[[1, 2, 3],
              [4, 5, 6],
              [7, 8, -1]]

# Breadth-first search implemented using a FIFO queue data structure
def bfs(initial_state, target_state):
    q = Queue()
    q.put(initial_state)
    step = 0
    visited = {}
    
    while not q.empty():
        temp = q.get()
        
        if temp == target_state:
            return step
        
        k = tuple(map(tuple, temp))
        if k not in visited.keys():
            visited[k] = 1
            step += 1
            x, y = find_blank(temp)
            
            # Right check
            if y < 2:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x][y+1] = temp2[x][y+1], temp2[x][y]
                q.put(temp2)
            
            # Left check
            if y > 0:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x][y-1] = temp2[x][y-1], temp2[x][y]
                q.put(temp2)
            
            # Up check
            if x > 0:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x-1][y] = temp2[x-1][y], temp2[x][y]
                q.put(temp2)
            
            # Down check
            if x < 2:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x+1][y] = temp2[x+1][y], temp2[x][y]
                q.put(temp2)
    
    return -1

# Depth-first search implemented using a stack data structure
def dfs(initial_state, target_state):
    q = LifoQueue()
    q.put(initial_state)
    step = 0
    visited = {}
    
    while not q.empty():
        temp = q.get()
        
        if temp == target_state:
            return step
        
        k = tuple(map(tuple, temp))
        if k not in visited.keys():
            visited[k] = 1
            step += 1
            x, y = find_blank(temp)
            
            # Right check
            if y < 2:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x][y+1] = temp2[x][y+1], temp2[x][y]
                q.put(temp2)
            
            # Left check
            if y > 0:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x][y-1] = temp2[x][y-1], temp2[x][y]
                q.put(temp2)
            
            # Up check
            if x > 0:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x-1][y] = temp2[x-1][y], temp2[x][y]
                q.put(temp2)
            
            # Down check
            if x < 2:
                temp2 = copy.deepcopy(temp)
                temp2[x][y], temp2[x+1][y] = temp2[x+1][y], temp2[x][y]
                q.put(temp2)
    
    return -1

# Main program
print("Enter Initial State:")
initial_state = []
print("Enter 1 for the sample grid.\nEnter 2 for user input.")
choice = int(input("Enter your choice: "))

if choice == 2:
    for i in range(1, 4):
        print("Enter row", i, ":", end=" ")
        initial_state.append(list(map(int, input().split())))
else:
    initial_state = [[3, 2, 1],
                    [4, 5, 6],
                    [8, 7, -1]]

print("\n-------Initial State-------\n")
print_state(initial_state)
print("-------Target State-------\n")
print_state(target_state)

if isSolvable(initial_state):
    bfscount = bfs(initial_state, target_state)
    
    if bfscount != -1:
        print("Problem solved with BFS in", bfscount, "steps")
    else:
        print("\nProblem cannot be solved")
        exit(1)
        
    dfscount = dfs(initial_state, target_state)
    print("\nProblem solved with DFS in", dfscount, "steps")
    
    if bfscount > dfscount:
        print("\nDFS took fewer steps than BFS\n")
    elif dfscount > bfscount:
        print("\nBFS took fewer steps than DFS\n")
    else:
        print("\nBFS and DFS took the same number of steps\n")
else:
    print("\nProblem cannot be solved\n")
exit(1)
