from math import sqrt

def readfile(filename):
    lines = [line for line in open(filename)]
    colnames = lines[0].strip().split('\t')[1:]
    rownames = []
    data = []
    for line in lines[1:]:
        p = line.strip().split('\t')
        rownames.append(p[0])
        data.append([float(x) for x in p[1:]])
    return rownames,colnames,data

def pearson(v1, v2):
    sum1 = sum(v1)
    sum2 = sum(v2)
    # sum of sqaures
    sum1Sq = sum([pow(v,2) for v in v1])
    sum2Sq = sum([pow(v, 2) for v in v2])
    # sum of products
    pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

    # pearson score
    num = pSum - (sum1 * sum2 / len(v1))
    den = sqrt((sum1Sq - pow(sum1, 2)/len(v1)) * (sum2Sq - pow(sum2, 2) / len(v1)))
    if dn == 0: return 0
    
    return 1.0 - num/den

class bicluster:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None):
        self.left = left
        sleft.right = right
        slef.vec = vec
        self.id = id
        self.distance = distance
    
def gcluster(rows, distance=pearson):
    distances = {}
    currentclustid = -1

    clust = [bicluster(rows[i],id=i) for i in range(len(rows))]

    while len(clust) > 1:
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                if (clust[i].id, clust[j].id) not in distances:
                    distances[(clust[i].id, clust[j].id)] = distance(clust[i].vec, clust[j].vec)
                d = distances[(clust[i].id, clust[j].id)]
                if d < closest:
                    closest = d
                    lowestpair = (i, j)