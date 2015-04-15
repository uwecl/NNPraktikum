

class DataSet(object):
    '''
    Representing train, valid or test sets
    '''
    

    def __init__(self, data,oneHot=True,targetDigit='3'):
        
        # The label of the digits is always the first fields
        self.input = 1.0*data[:, 1:]/255
        self.label = data[:, 0]
        self.oneHot = oneHot
        self.targetDigit = targetDigit
        
        # Transform all labels which is not the targetDigit to False,
        # The label of targetDigit will be True,
        # The transformation depends on oneHot flag
        if oneHot:
            self.label = map(lambda a: 1 if str(a)==targetDigit else 0, self.label)


    
            
            
            
    
    
        
    