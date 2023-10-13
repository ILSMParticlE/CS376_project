import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from Dataset import jacobkie_transform

FIRST_LINE = 'img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n'
RESULT_PATH = './result/'
SUBMISSION_PATH = './submission.csv'

def predict_test(model, test_dataloader, result_path=SUBMISSION_PATH):
    test_len = len(test_dataloader)
    print(f'Test: {test_len} batches') 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    softmax = nn.Softmax(dim=1)
    model.freeze()
    model.cuda()
    model.eval()
    result_file = open(result_path, 'w')
    result_file.write(FIRST_LINE)
    result_strs = ''
    print('Predict the test datas')

    i = 0
    for X_test, fname in test_dataloader:
        if i % 100 == 0:
            print(f'{i+1}/{test_len}')
        with torch.no_grad():
            output = model(X_test.to(device))
        output = softmax(output)
        
        for j in range(X_test.size(0)):
            result = ','.join(list(map(lambda x: str(x), output[j].tolist())))
            # result_file.write(f'{fname[j]},{result}\n')
            result_strs += f'{fname[j]},{result}\n'
        i += 1
        del X_test
    
    result_file.write(result_strs)
    result_file.close()
    