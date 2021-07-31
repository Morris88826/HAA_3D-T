import numpy as np

def calculate_cost(target1, target2):
    return np.linalg.norm(target1 - target2)


def dynamic_time_warp(sequence1, sequence2):
    N = sequence1.shape[0]
    M = sequence2.shape[0]

    D = np.zeros((N+1, M+1))
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0,0] = 0

    trace_matrix = np.zeros((N, M))

    for i in range(1, N+1):
        for j in range(1, M+1):

            previous_info = np.array([D[i-1, j-1], D[i-1, j], D[i, j-1]])
            D[i,j] = calculate_cost(sequence1[i-1], sequence2[j-1]) + np.amin(previous_info)
            
            trace_matrix[i-1, j-1] = np.argmin((previous_info))


    backtrack = np.array([N, M])
    
    trace = []
    trace.append(np.copy(backtrack))
    while(backtrack[0]!=1 or backtrack[1]!=1):
        if trace_matrix[backtrack[0]-1, backtrack[1]-1] == 0:
            backtrack[0] -= 1
            backtrack[1] -= 1
        elif trace_matrix[backtrack[0]-1, backtrack[1]-1] == 1:
            backtrack[0] -= 1
        else:
            backtrack[1] -= 1
        trace.append(np.copy(backtrack))

    trace = np.array(trace)[::-1]
    trace = trace-1
    cost = D[-1, -1]
    return D, trace, cost

if __name__ == '__main__':
    np.random.seed(1)
    # skeleton_3d_1 = np.random.rand(6, 17, 3)
    # skeleton_3d_2 = np.random.rand(7, 17, 3)

    skeleton_3d_1 = np.array([0,2,0,1,0,0])
    skeleton_3d_2 = np.array([0,0,0.5,2,0,1,0])
    dynamic_time_warp(skeleton_3d_1, skeleton_3d_2) 

    