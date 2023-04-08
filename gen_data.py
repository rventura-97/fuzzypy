import numpy as np



def siso_reg_1():
    step = 0.00025
    x = np.arange(0,1.0,step)
    y = 0.5*np.ones(x.size)
    return x, y

def siso_reg_2():
    step = 0.00025
    x = np.arange(0,1.0,step)
    y = x
    return x, y

def siso_reg_3():
    x = [[]]*8
    y = [[]]*8
    step = 0.00025
    
    # Segment 1: y=0.5, x=[0,0.125[
    x[0] = np.arange(0,0.125,step)
    y[0] = 0.5*np.ones(x[0].size)
    
    # Segment 2: y=-4x+1 x=]0.125,0.25]
    x[1] = np.arange(0.125,0.25,step)
    y[1] = -2*x[1]+0.75
    
    # Segment 3: y=8x-2
    x[2] = np.arange(0.25,0.375,step)
    y[2] = 6*x[2]-1.25
        
    # Segment 4:
    x[3] = np.arange(0.375,0.5,step)
    y[3] = -8*x[3]+4
        
    # Segment 5:
    x[4] = np.arange(0.5,0.625,step)
    y[4] = 8*x[4]-4    
    
    # Segment 6:
    x[5] = np.arange(0.625,0.75,step)
    y[5] = -6*x[5]+4.75
        
    # Segment 7:
    x[6] = np.arange(0.75,0.875,step)
    y[6] = 2*x[6]-1.25
        
    # Segment 8:
    x[7] = np.arange(0.875,1+step,step)
    y[7] = 0.5*np.ones(x[7].size)
    
    x = np.concatenate(x)
    y = np.concatenate(y)
    
    return x, y