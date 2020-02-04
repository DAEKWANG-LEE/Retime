from numpy import array

def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):#  10개
            end_ix = i + n_steps  #  0 + 4 = 4
            if end_ix > len(sequence) -1: # 멈춰라 
                #4 > 10-1 
                break
            seq_x, seq_y = sequence[ i:end_ix], sequence[end_ix]  # 0,1,2,3 / 4,
            x.append(seq_x)
            y.append(seq_y)
    return array(x), array(y)

dataset = [0,1,2,3,4,5,6,7,8,9]
n_steps = 3
x, y = split_sequence(dataset, n_steps)

print(x)
print(y)

'''
0,1,2,3 /4
2,3,3,4 /5

5,6,7,8 /9

[[0 1 2]
 [1 2 3]
 [2 3 4]
 [3 4 5]
 [4 5 6]
 [5 6 7]
 [6 7 8]]
[3 4 5 6 7 8 9]

'''