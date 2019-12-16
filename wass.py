'''
Created on Jan 19, 2013

This code computes the Wasserstein distance between two diagrams,
as well as running the gradient descent algorithm from:

Frechet Means for Distributions of Persistence diagrams
Katharine Turner, Yuriy Mileyko, Sayan Mukherjee, John Harer
http://arxiv.org/abs/1206.2790



'''

import numpy as np
#import munkres as munk
# import _hungarian as h
import hungarian

import random
import pickle
#import matplotlib.pyplot as plt

   
## EDIT DISTANCE TO BE L_\INFTY

def dist( a, b, insideDistance = np.linalg.norm, bottleneck=False ):
    """
    This is a modified Wasserstein, using 'internal' norm that is
    L2, not L_\infty.

    Returns L2 or L_\infty distance between diagram points.
    """
    if a == 'Diag':
        if b == 'Diag':
            dist =  0
        else:
            if bottleneck:
                dist = b[1] - b[0]
            else:
                dist = (b[1]-b[0]) / np.sqrt(2)
    else:
        if b == 'Diag':
            if bottleneck:
                dist = a[1] - a[0] 
            else:
                dist = np.sqrt( (a[1]-a[0])**2/2. )
        else:
            if bottleneck:
                dist = max( [ b[i] - a[i] for i in range(2) ] )
            else:
                dist = insideDistance([b[i]-a[i] for i in range(2)])
    return dist


def WassDistDiagram(D,E,p = 2, bottleneck=False, returnPairing=False):
    '''
    Determines pairwise distances between all pairs of points in diagrams D and E
    Diagrams should be entered as lists of off-diagonal points, but could have 'Diag.
    
    EDIT: Diagrams could potential have None value as an off-diagonal point.
    We need to keep track of these so that the pairings line up for mean distribution 
    necessity, but this should return the same answer up to renumber if there are no
    None values.
    
    
    If returnPairing == True, this wil also return the actual pairing used for the 
    Wass distance
    '''
    #===========================================================================
    # while len(D)<len(E):
    #    D.append('Diag')
    # while len(E)<len(D):
    #    D.append('Diag')
    #===========================================================================
    
    k = len(D)
    l = len(E)
    D = D[:]    #Necessary to not edit the diagram at the higher level
    E = E[:]
    D.extend(['Diag' for i in range(l)])
    E.extend(['Diag' for i in range(k)])

    M = np.zeros([k+l,k+l]) #Note: should be square!

    for i in range(len(D)):
        for j in range(len(E)):
            if bottleneck:
                M[i,j] = dist( D[i],E[j], bottleneck=True )
            else:
                M[i,j] = dist(D[i],E[j])**p
   
    N = M.tolist()
    pairs = zip(range(len(D)), hungarian.lap(N))
    
    answer = 0
    
    if not bottleneck:
        # sum over the 
        for pair in pairs:
            answer += M[pair[0],pair[1]]
        answer = (answer)**(1./p)
    # bottleneck
    else:
        pair_values = [ M[pair[0],pair[1]] for pair in pairs ]
        answer = max( pair_values )
    
    if returnPairing:
        '''
        Returns pairs used for distance.
        If one of the entries in the diagram is 'Diag', it does not return 
        a specific pair for that one: All pairs are of the form:
        A) [i,j]
        B) [i, 'Diag']
        C) ['Diag', j]
        '''
        PairsEdit = []
        for p in pairs:
            p = list(p)
            
            if D[p[0]] == 'Diag':
                p[0] = 'Diag'
            
            if E[p[1]] == 'Diag':
                p[1] = 'Diag'
            
            if not p[0] == 'Diag' or not p[1] == 'Diag':
                PairsEdit.append(p)
            
            #===================================================================
            # if p[0]< k:
            #    if p[1]<l:
            #        PairsEdit.append(p)
            #    else:
            #        PairsEdit.append([p[0],'Diag'])
            # else:   #p[0] must be diagonal
            #    if p[1] < l:
            #        PairsEdit.append(['Diag',p[1]])
            #    else:
            #        pass    #both are beyond marked points, so don't add
            #===================================================================

        
        return answer, PairsEdit
    else:
        return answer
    
    

def pointsBarycenter(pointsList, numberAdditionalDiagonals=0):
    '''
    Computes bary center of a list of points, where some could be "Diag"
    
    '''    
    m = numberAdditionalDiagonals   # m is number of diagonals
    X = []
    Y = []
    
    for p in pointsList:
        if p == 'Diag' or p == None:
            m+=1
        else:
            X.append(p[0])
            Y.append(p[1])
    
    k = len(X)  #k is number of off-diagonal points
    X = sum(X)
    Y = sum(Y)
    
    if not k == 0:
        denom = 2*k**2 + 2*k*m
        denom = float(denom)
        xNumerator = (2*k+m)*X + m*Y
        yNumerator = m*X + (2*k+m)*Y
        
        x = xNumerator / denom
        y = yNumerator / denom
        barycenter = [x,y]
    else:
        barycenter = 'Diag'     #Do I want this??????
    
    return barycenter
    
 
 
 
 
 
 
 
 
 
 
 
 
#@profile    
def WassBarycenter(diagramList, returnGrouping = False, printLoopCount = False):
    '''
    Runs Kate's barycenter mean algorithm.
    '''
    numDiagrams = len(diagramList)
    stop = False
    lastGrouping = None
    #pick random diagram to initialize
    Y = diagramList[  random.randint(0,numDiagrams-1)]
    
    #Remove 'Diag's from Y if there are any
    Y = [p for p in Y if not p == 'Diag']
    
    whileLoopCount = 0
    
    while not stop and whileLoopCount<100:
        whileLoopCount += 1
        
    
        grouping = [  [] for i in range(len(Y))]
        # grouping stores the multi-pairing info
        # grouping[i] is list of points matched to point i in Y, 
        # grouping[i][j] is the point coming from the j-th diagram in the list
        
        for i in range(len(diagramList)):
            d = diagramList[i]
            dist,pairing = WassDistDiagram(Y,d,returnPairing = True)
            for p in pairing:
                
                #off-diagonal in Y paired with something in d
                if not p[0] == 'Diag':
                    grouping[p[0]].append(p[1])
    
                
                
                #Diagonal in Y paired with off-diagonal in d
                else:
                    if not p[1] == 'Diag':
                        match = ['Diag' for k in range(len(diagramList))]
                        match[i] = p[1]
                        grouping.append(match)
                
        newY = []
        for i in range(len(grouping)):
            # Here, the entries in the grouping are indices in the diagrams
            match = grouping[i]
            
            
            P = []
            m = 0
            for j in range(len(match)):
                if match[j] == 'Diag':
                    m += 1
                else:
                    P.append(diagramList[j][match[j]])
            if not len(P) == 0:              
                bary = pointsBarycenter(P,m)
                if not bary == 'Diag':
                    newY.append(bary)
            
        
        ### Check to see if we can stop
        if Y == newY:
            stop = True
        else:
            Y = newY[:]
         
    if printLoopCount:
        print("The code looped through Kate's algorithm ", whileLoopCount, " times.")
    
                 
    if returnGrouping:
        return Y, grouping
    else:
        return Y
    



#===============================================================================
# 
# 
# def drawDiagram(d, color = 'purple',size = 10):
#    x = [round(p[0],2) for p in d if not p == 'Diag']
#    y = [round(p[1],2) for p in d if not p == 'Diag']
# 
#    plt.scatter(x, y, color = color, s = size)
# 
# 
# 
#===============================================================================





    

if __name__ == '__main__':
    

    # JJB - 10/28/13
    d1 = [[1,7],[1,1.1],[5,5.5]]
    d2 = [[2,6],[3,3.5],[5,6]]
    dist,pairing = WassDistDiagram( d1, d2, bottleneck=False, returnPairing=True )

    # from cyWasserstein import WassDistDiagram
    # dist,pairing = WassDistDiagram( d1, d2, bottleneck=False, returnPairing=True )
    
    # f = open('AnnulusMeanDistribution-30Draws-100Pts-0.3epsilon.pkl')
    # Distr = pickle.load(f)
    # Diags = []
    # for i in range(5):
    #     Diags.append(Distr[i][0][:50])
    # Y  = WassBarycenter(Diags)
    
    # x = []
    # y = []
    # for p in Y:
    #     x.append(p[0])
    #     y.append(p[1])
    #plt.scatter(x,y)
    #plt.show()
    
    #===========================================================================
    # d1 = [[1,7],[1,1.1],[5,5.5]]
    # d2 = [[2,6],[3,3.5],[5,6]]
    # d3 = [[1.5,7.5],[8,9]]
    #===========================================================================
    
    #===========================================================================
    # d2 = [[1,7],'Diag',[2,6]]
    # d1 = [[2,7],[1,6],[8,8.5]]
    # d3 = [[1,8],[2,9],'Diag',[3,3.5]]
    # Y =  WassBarycenter([d1,d2,d3])
    #===========================================================================
    #===========================================================================
    # 
    # drawDiagram(d1,'red',50)
    # drawDiagram(d2,'orange',50)
    # drawDiagram(d3,'green',50)
    # drawDiagram(Y,'purple',25)
    # plt.plot([0,10],[0,10])
    # plt.axes([0,10,0,10])
    # plt.show()
    #===========================================================================