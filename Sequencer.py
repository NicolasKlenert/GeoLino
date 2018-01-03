import numpy as np

class Sequencer(object):

    def __init__(self, textfile):
        #read textfile out and populate the main variables
        self.matrix = Sequencer._readTextFile(textfile)

    def _readTextFile(url):
        #outputs an numpy matrix
        with open(url) as f:
            tupel = [int(string) for string in f.readline().rstrip().split(' ')]
            matrix = np.empty(tupel)
            content = f.readlines()
            for i in range(tupel[0]):
                matrix[i] = [float(num) for num in content[i].rstrip().split(' ')]
        return matrix
