import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

#hyperparameters
batch_size = 32 # No of independent sequences processed in parallel (B)
block_size = 16 # sequence lenght/time (T)
learning_rate = 1e-2
max_iters = 5000
eval_interval = 500
eval_iters = 200
n_embd = 32 # number of embedding dimension
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_head = 4
n_layers = 6
dropout = 0.2

torch.manual_seed(1337)

#reading the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#extracting the unique characters from the file
chars = sorted(list(set(text)))
vocab_size = len(chars) #channel_size

#tokenization

#1. mapping the string to int and vice versa
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

#2. encode and decode
encode = lambda e:[stoi[s] for s in e]
decode = lambda d:''.join([itos[i] for i in d])

data = torch.tensor(encode(text), dtype = torch.long)

#spliting the data into train and test
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

#get diff chunks of data of particular sequence length
def get_batch(split):
    data = train_data if split == 'train' else test_data
    randx = torch.randint(len(data)- block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in randx]) #(B,T)
    y = torch.stack([data[i+1:i+block_size+1] for i in randx]) #(B,T)
    x,y = x.to(device), y.to(device)
    return x,y

#estimate loss
@torch.no_grad() #No backpropagation (no calc of grad desc required)
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros((eval_iters))
        for k in range(eval_iters): #checking loss for eval_iters number of batches
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#single-head self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias = False) 
        self.key = nn.Linear(n_embd, head_size, bias = False) 
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #since tril is not an existing parameter

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x) #(B,T,H)
        k = self.query(x) #(B,T,H)
        #computing attention scores (affinities)
        weight = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,H) * (B,H,T) => (B,T,T)
        #masking :(makes sure the tokens dont interact with future tokens) / (decoder attention blocks)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        #softmax
        weight = F.softmax(weight, dim=-1)
        #dropout
        weight = self.dropout(weight)
        #value
        v = self.value(x)
        #weighted aggregation of values
        out = weight @ v

        return out


#Multihead attention
class MultiHeadAttention(nn.Module):
    #multiple self attention heads in parallel, each head of size (B, T, n_embd/num of heads)
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) #each head focuses on diff aspects of input (captures diverse patterns)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        #concatenate the weighted aggregation from each head back to the original dimension
        out = torch.cat([h(x) for h in self.heads], dim = - 1) 
        out = self.dropout(self.proj(out)) #projects the output back to the residual path(i.e matching dimensionality as input, input features are refined) and adds a dropout
        return out
    

class feedforward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        #feedforward layer with non-linear activation function
        self.network = nn.Sequential(
            #linear layer
            nn.Linear(n_embd, 4 * n_embd),
            
            #Rectified Linear Unit (Adds Non_linearity in the model:: Captures complex patterns)
            nn.ReLU(),

            #project back to residual path (for residual connections)
            nn.Linear(4 * n_embd, n_embd),

            #adding dropout before connecting back to residual path
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)       
     

class Block(nn.Module):
    '''transformer block: communication followed by computation'''
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        #communication
        self.sa = MultiHeadAttention(n_head, head_size)
        #computation
        self.ffwd = feedforward(n_embd)
        #layernorms #pre-layernormalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x




#creating a simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layers)]) # here * unpacks the list of blocks to be passed individually as separate arguments which is what we d
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, target = None):
        B,T = idx.shape
        tok_embd = self.token_embedding_table(idx) #(B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_embd + pos_embd #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,C) => raw, unnormalized scores 
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            #reshaping logits as cross entropy is expected N*C (N= number of prob distribution, C = Channel size)
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target) # -ve log likelihood (-log(P))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for steps in range(max_new_tokens):

            idx_cond = idx[:,-block_size:] #takes only the last block_size length of input tokens(prevents out of range error due to positional encoding)

            #get predictions/logits
            logits, loss = self(idx_cond)
            
            #focus only the last time step/ last block of each batch
            logits = logits[:,-1,:]

            #softmax
            probs = F.softmax(logits, dim=-1)

            #sampling new token
            idx_next = torch.multinomial(probs, num_samples=1)

            #appending the generated token
            idx = torch.cat((idx, idx_next), dim = 1)
        
        return idx


model = BigramLanguageModel()
m = model.to(device)

#training the model
optimizer = optim.AdamW(model.parameters(), lr = learning_rate) #AdamW adds weight decay (adds penalty to the weight) which generalizes the patterns (discourages larger weights) avoiding overfitting

for iters in range(max_iters):

    #computing loss every once in a while
    if iters % eval_interval == 0:
        loss = estimate_loss()
        print(f"for step {iters} the train loss = {loss['train']:.4f} and test loss = {loss['test']:.4f}")

    #sampling batches of data
    xa, ya = get_batch('train')

    #get loss
    logits, loss = model(xa, ya)

    #reset the gradient to zero (so prev grad doesnt append in the new iteration)
    optimizer.zero_grad(set_to_none= True)

    #calc grad descent
    loss.backward()

    #update parameters
    optimizer.step()
 

#generating
context = torch.zeros((1,1), dtype= torch.long) #initial token
print(decode(m.generate(context, max_new_tokens=800)[0].tolist()))
