from data import torch, line_to_tensor
from train import category_from_output


rnn = torch.load('char-rnn-classification.pt')

def predict(input_line):
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        hidden = rnn.init_hidden()
    
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        return guess


if __name__ == '__main__':
    while True:
        sentence = input("Input:")
        if sentence == "quit":
            break
        prediction = predict(sentence)
        print(prediction)