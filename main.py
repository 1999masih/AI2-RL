import math
import sys
import operator
import numpy as np


# pygame.init()


# screem = pygame.dispay

ACTIONS = np.array(['N', 'E', 'S', 'W'])
# ACTION_PROBS = {
#     ACTIONS[0]: [0.0, 0.7, 0.1, 0.1, 0.1],
#     ACTIONS[1]: [0.0, 0.1, 0.7, 0.1, 0.1],
#     ACTIONS[2]: [0.0, 0.1, 0.1, 0.7, 0.1],
#     ACTIONS[3]: [0.0, 0.1, 0.1, 0.1, 0.7]
# }


class Cell:
    def __init__(self, north, east, south, west):
        self.north = north
        self.west = west
        self.east = east
        self.south = south
        self.source = False
        self.goal = False
        self.col = None
        self.row = None
        self.best_action = ''
        self.value = 0
    
    
    def __repr__(self):
        return "{0} {1} {2} {3}, issource:{4}, isgoal:{5}, value:{6}, bestAct:{7}, loc:{8}{9} ".format(self.north, self.east, self.south, self.west, self.source, self.goal, self.value, self.best_action, self.row, self.col)

    #setters
    def set_source(self):
        self.source = True

    def set_goal(self):
        self.goal = True

    def set_value(self, value):
        self.value = value
    
    def set_col(self, col):
        self.col = col
    
    def set_row(self, row):
        self.row = row

    def set_action(self, action):
        self.best_action = action
    
    #getters
    def get_col_row(self):
        return self.row, self.col
    
    def get_action(self):
        return self.best_action
    
    def get_north(self):
        return self.north

    def get_south(self):
        return self.south
    
    def get_east(self):
        return self.east
    
    def get_west(self):
        return self.west

    def is_goal(self):
        return self.goal

    def is_source(self):
        if self.source == True:
            return True

    def is_blocked(self, action):

        if action == 'N':
            if self.north == '0':
                return True
        elif action == 'E':
            if self.east == '0':
                return True
        elif action == 'S':
            if self.south == '0':
                return True
        elif action == 'W':
            if self.west == '0':
                return True
        
        return False
        

    
    
    

def print_better(sth):
    print(repr(sth))


def reward(state):
    
    if state.is_goal():
       return 5
    else:
        return 0
    
    

def get_neighbor(state, maze):
    
    row, col = state.get_col_row()

    return [
        maze[row][col], #self
        maze[row - 1][col], #N
        maze[row][col + 1], #E
        maze[row + 1][col], #S
        maze[row][col - 1] #W
    ]


def is_in_maze(state, maze):

    R, C = maze.shape
    row, col = state.get_col_row()
    return (0<= row < R) and (0<= col < C) and state.is_blocked() == False
    
def find_source_cell(maze):
    
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i][j].is_source():
                return maze[i][j]


def next_position(state ,action, maze):
        
    row, col = state.get_col_row()
    is_locked = state.is_blocked(action)
    
    if action == 'N':
        next_state = (row - 1, col)
    elif action == 'E':
        next_state = (row, col + 1)
    elif action == 'S':
        next_state = (row + 1, col)
    elif action == 'W':
        next_state = (row, col - 1)
    
    if (next_state[0] >= 0) and (next_state[0] < maze.shape[0]):
        if(next_state[1] >= 0) and (next_state[1] < maze.shape[1]):
            if not is_locked:
                # print('hmm', maze[next_state[0]][next_state[1]].is_blocked(action))
                return (next_state[0], next_state[1])
    return ('x', 'x')

def policy_evaluation(maze, epsilon = 0.00001  ,discount_factor = 0.5):
    
    value = np.zeros((maze.shape[0], maze.shape[1]), dtype=np.uint64)

    while True:
        old_value = np.zeros((maze.shape[0], maze.shape[1]), dtype =np.uint64)

        delta = 0
        for state in maze:
            for x in range(len(state)):
                bellman = 0
                i, j = state[x].get_col_row()
                for action in ACTIONS:
                    next_i, next_j = next_position(maze[i][j], action, maze)
                    if next_i == 'x' and next_j == 'x':
                        continue
                    bellman += reward(maze[next_i][next_j]) + discount_factor * value[next_i][next_j]
                
                delta = max(delta, abs(bellman - value[i][j]))

                old_value[i][j] = bellman

        value = old_value

        if(delta < epsilon):
            break
    
    return value


def policy_iteration(maze, epsilon = 0.00001, discount_factor = 0.5):
    

    
    while True:
        
        value = policy_evaluation(maze)
        is_stable = True
        for state in maze:
            for x in range(len(state)):

                action_values = {'N': 0, 'E': 0, 'S': 0, 'W':0}
                i, j = state[x].get_col_row()

                for action in ACTIONS:
                    next_i, next_j = next_position(maze[i][j], action, maze)
                    if next_i == 'x' and next_j == 'x':
                        continue
                    action_values[action] = reward(maze[next_i][next_j]) + discount_factor * value[next_i][next_j]
                
                best_action_value = max(action_values.items(), key=operator.itemgetter(1))

                chosen_action = maze[i][j].get_action()

                best_action = best_action_value[0]

                if(chosen_action != best_action):
                    is_stable = False
                
                maze[i][j].set_action(best_action)
        
        if is_stable:
            return maze, value
                
                


def value_itertation(maze, epsilon = 0.0001, discount_factor = 0.9):
    
    next_state_values = np.zeros((maze.shape[0], maze.shape[1]))
    iteration = 1
    while True:
        delta = 0
        for state in maze:
            for x in range(len(state)):
                action_values = {'N': 0, 'E': 0, 'S': 0, 'W':0}
                i, j = state[x].get_col_row()
                for action in ACTIONS:
                    next_i, next_j = next_position(maze[i][j], action, maze)
                    if next_i == 'x' and next_j == 'x':
                        continue
                    # print('action in maze[{0}][{1}]'.format(i, j) + action)
                    action_values[action] = reward(maze[next_i][next_j]) + discount_factor * next_state_values[next_i][next_j]
                
                best_action_value = max(action_values.items(), key=operator.itemgetter(1))

                delta = max(delta, abs(best_action_value[1] - next_state_values[i][j]))

                next_state_values[i][j] = best_action_value[1]

                best_action = best_action_value[0]
                # print('bestAct',i,j , best_action)
                maze[i][j].set_action(best_action)
             

        if delta < epsilon:
            break

    return maze, next_state_values




def main():

    with open('samples/sample_5.txt') as sample:
        sampleList = list(sample)

    #number of columns and rows
    row_col_num = sampleList[0].split()
    row_num = int(row_col_num[0])
    col_num = int(row_col_num[1])

    cells_num = sampleList[1:-1]
    print('num', cells_num)
    
    #source and goal
    goal_and_source = sampleList[-1].split()
    
    source = int(goal_and_source[0])
    goals = list(map(int, goal_and_source[1:]))
    
    print('source', source)
    print('goals', goals)

    cells_obejct = []
    for index, node in enumerate(cells_num):
        x = node.split()
        cell = Cell(x[0], x[1], x[2], x[3])
        
        if index == source - 1:
            cell.set_source()

        for i in range(len(goals)):
            if index == goals[i] - 1:
                cell.set_goal()
        
        cells_obejct.append(cell)  
    temp = cells_obejct[22]
    print_better(temp)
    print('obj', cells_obejct)
    maze = np.ndarray(shape = (row_num, col_num), dtype = Cell)

    # print(maze)
    z = 0
    # for i in range(row_num):
    #     for j in range(col_num):
    #         cells_obejct[z].set_row(i)
    #         cells_obejct[z].set_col(j)
    #         maze[i][j] = cells_obejct[z]
    #         z = z + 1
    # print('before',maze)
    # maze = np.transpose(maze)
    z = 0
    for i in range(col_num):
        for j in range(row_num):
            cells_obejct[z].set_row(j)
            cells_obejct[z].set_col(i)
            maze[i][j] = cells_obejct[z]
            z = z + 1
    # print('before',maze)
    # print('after', np.transpose(maze) )
    maze = np.transpose(maze)
    print('after', maze)    
    print('source cell',find_source_cell(maze))
    # print_better(maze)
    print('fuck',maze[1][0],maze[1][0].is_blocked('W'))


    # print('next', maze[next_position(maze[0][0], 'E', maze)[0]][next_position(maze[0][0], 'E', maze)[1]])
    print('next', maze[0][3].is_blocked('S') , next_position(maze[0][3], 'S', maze))


    print('--------------------------------------')

    # for action in ACTIONS:
    #     next_i, next_j = next_position(maze[0][3], action, maze)
    #     print('next_i, next_j', next_i, next_j)
    #     if next_i == 'x' and next_j == 'x':
    #         continue
    #     print('Action', action)
    newmaze, vall = policy_iteration(maze)
    print(vall)
    print(newmaze)
    # print(next_position(maze[0][0], 'N', maze))
    # for state in maze:
    #     print(state.)


main()

