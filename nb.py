import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################
    # the rows of matrix are numbers of emails
    # the columns of matrix is numbers of tokens
    spam_token = matrix[category == 1,:]
    news_token = matrix[category == 0,:]
    a = news_token.shape[0]+spam_token.shape[0]
    b = spam_token.shape[0]
   
    spam_num = np.sum(spam_token,axis = 1)
    news_num = np.sum(news_token,axis = 1)
    state['spam'] = (1+np.sum(spam_token,axis = 0) )/(N + np.sum(spam_num))
    state['news'] = (1+np.sum(news_token,axis = 0) )/(N + np.sum(news_num))
    state['spam_prob'] = np.float(b)/np.float(a)
    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    prob_test_spam = np.log(np.dot(matrix,state['spam']))
    prob_test_news = np.log(np.dot(matrix,state['news']))
    score = np.exp(prob_test_news+np.log(1-state['spam_prob'])-prob_test_spam-np.log(state['spam_prob']))
    prob = 1/(1+score)
    output[prob>0.5] = 1
    
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
