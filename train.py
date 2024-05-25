import torch
import random
from data import *
from model import *

category_lines, all_categories = load_data()
n_categories = len(all_categories)

def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

n_hidden = 128
rnn = RNN(N_LETTERS, n_hidden, n_categories)

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        
    loss = criterion(output, category_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return output, loss.item()

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]


if __name__ == '__main__':
    
    current_loss = 0
    all_losses = []
    print_steps = 5000
    n_iters = 100000

    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        
        output, loss = train(line_tensor, category_tensor)
        current_loss += loss 
            
        if (i+1) % print_steps == 0:
            guess = category_from_output(output)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f"{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}")

    torch.save(rnn, 'char-rnn-classification.pt')