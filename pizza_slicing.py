import torch as tc
import argparse
import numpy as np


class Data:
    """
    0 represents mashroom and 1 represents tomato.
    """

    def __init__(self, fp):
        with open(fp, 'r') as data:
            info = list(map(int, data.readline().split()))
            print('Pizza info:', info)
            self.size = tuple(info[:2])
            self.R = info[0]
            self.C = info[1]
            self.L = info[2]
            self.H = info[3]
            self.nb = self.R * self.C
            self.cells = []
            self.offset = tc.IntTensor([1,1])
            pizza = []
            for l in data:
                row = []
                for v in l.strip():
                    if v == 'T': row.append(1)
                    else: row.append(0) 
                pizza.append(row)
            self.pizza = tc.IntTensor(pizza)


    def cuda(self):
        self.pizza = self.pizza.cuda()

def possibleShapes(d):
    listWL = []
    max_S = d.H
    min_S = d.L *2
    for w in range(1, max_S+1):
        for l in range(1, max_S+1):
            if w*l >= min_S and w*l <= max_S:
                listWL.append((w,l))
    print('Nb possible slice shapes:', len(listWL))
    return tc.IntTensor(list(reversed(listWL)))  
    # return tc.IntTensor(listWL)  

def nextCell(d, tmp_cell):
    
    while True:
        if tmp_cell[0] >= d.R:            
            tmp_cell[0] = 0
            tmp_cell[1] += 1
            if tmp_cell[1] >= d.C:
                return
        if d.pizza[tmp_cell[0], tmp_cell[1]] <= 1:
            return tc.IntTensor(tmp_cell)
        tmp_cell[0] += 1

def cutPizza(d, listWL, pices, tmp_cell):
    # 1 find current cell - upper left one
    current_pos = nextCell(d, tmp_cell)
    tmp_cell[0] += 1

    if current_pos is None: 
        print('Finished slicing')
        return True

    next_cell = [0,0]
    #tmp_states = []
    for shape in listWL:

         next_cell = current_pos + shape
         if next_cell[0] <= d.R and next_cell[1] <= d.C: # in the range of pizza
            temp_slice = d.pizza[int(current_pos[0]):int(next_cell[0]), 
                int(current_pos[1]):int(next_cell[1])]
            nb_T = tc.sum(temp_slice) #  nb_T = tf.reduce_sum(temp_slice)
            nb_M = shape[0]*shape[1] - nb_T
            if nb_T >= d.L and nb_M >= d.L and nb_M+nb_T <= d.H: # valid slice
                print('From cell ', list(current_pos), '\t\tslicing ', list(shape))
                # pizza_copy = d.pizza.clone()
                d.pizza[int(current_pos[0]):int(next_cell[0]),
                    int(current_pos[1]):int(next_cell[1])] =\
                    tc.mul(tc.ones(*shape), d.nb)
                #tmp_states.append(pizza_copy)
                tmp_cell = [next_cell[0], current_pos[1]]
                pices.append(list(current_pos)+list(next_cell-d.offset))

                return False


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='small',
        help='dataset - medium | big | small | example')  
parser.add_argument('-c', action='store_true', 
        help='use CUDA')

args = parser.parse_args()

tmp_cell = tc.IntTensor([0,0])
# cells_visited = set()
pices = []
finished = False
d = Data('./data/%s.in' % args.d)

# cuda 
if args.c:
    d.cuda()

listWL = possibleShapes(d)
print(listWL)
while not finished:
    finished = cutPizza(d, listWL, pices, tmp_cell)

scores = tc.sum(d.pizza.ge_(5))
result = '%d/%d'%(scores, d.nb)
print(result)

np.savetxt('result_%s.txt'%args.d[:-3], np.asarray(pices), fmt='%d',
    header='%s\t%s' % (len(pices), result))
