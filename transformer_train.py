import torch
import torch.nn as nn
import torch.nn.functional as F
#设备处理
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


#数据预处理
chars=sorted(list(set(open('input.txt','r').read())))
stoi={j:i for i,j in enumerate(chars)}
itos={i:j for i,j in enumerate(chars)}
input_text=open('input.txt','r').read()

batch_size=64


def makeegs(inputtext,T):#inputtext的格式应该是：把全文达成字符的大列表
    Xs=[]
    Ys=[]
    for _ in range (batch_size):
        i=torch.randint(0,(len(inputtext)-T),(1,)).item()
        charlist=inputtext[i:i+T]
        answerlist=inputtext[i+1:i+T+1]
        charnum=[]
        answernum=[]
        for char,answer in zip(charlist,answerlist):
            charnum.append(stoi[char])
            answernum.append(stoi[answer])
        X=torch.tensor(charnum).view(1,T)#1*T
        Y=torch.tensor(answernum).view(1,T)
        Xs.append(X)
        Ys.append(Y)
    Xfinal=torch.cat(Xs,dim=0)
    Yfinal=torch.cat(Ys,dim=0)
    return Xfinal,Yfinal
    
    
    

    #这个X final和Y final的形状都是64*T，(T是看到的上下文长度)



#我会把char_num作为总类数
char_num=len(chars)
block_size=256

class pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(char_num,384)#每个token变成了384维的vec
        self.position=nn.Parameter(torch.zeros(1,256,384))
    def forward(self,input):#我希望是B（64）*T的input
        x_embedded=self.embedding(input)
        T=input.shape[1]
        output=x_embedded+self.position[:,:T,:]
        return output

def causalmusk(B,T,device):
    a=torch.zeros(B,6,T,T,device=device)
    for i in range(T):
        for j in range(i+1,T):
            a[:,:,i,j]=float('-inf')
    return a


class BLOCK(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(384)
        self.multiQ=nn.Parameter(torch.randn(384,384)/(384**0.5))
        self.multiK=nn.Parameter(torch.randn(384,384)/(384**0.5))
        self.multiV=nn.Parameter(torch.randn(384,384)/(384**0.5))
        self.comb12=nn.Linear(384,384)
        self.ln2=nn.LayerNorm(384)
        self.mlp=nn.Sequential(nn.Linear(384,1536),nn.GELU(),nn.Linear(1536,384))


#这个时候x的默认输入就是B（定为64），256，384
    def forward(self,x):
        T=x.shape[1]
        B=x.shape[0]
        #第一轮残差

        input1=self.ln1(x)#B 256(人定) 384
        mulQ=(input1@self.multiQ).permute(0,2,1).view(B,6,64,T).permute(0,1,3,2)#注意这东西不用nn.para，因为nn.para和require grad是两码事。
        mulK=(input1@self.multiK).permute(0,2,1).view(B,6,64,T)#不调轴相当于直接转置
        mulV=(input1@self.multiV).permute(0,2,1).view(B,6,64,T).permute(0,1,3,2)
        mulA=torch.softmax((mulQ@mulK+causalmusk(B,T,x.device))/8,dim=3)
        mul=mulA@mulV
        Attentionfied=torch.cat([mul[:,i,:,:] for i in range(6)],dim=2)
        #下面是多头的最后一步让12也混合，虽然12本身的每一个就在结构上已经具有混合所有的能力了
        final_0=self.comb12(Attentionfied)#多头结束

        #残差接回
        final=final_0+x#同时作为新的残差

        #进mlp
        beforemlp=self.ln2(final)
        after=self.mlp(beforemlp)
        FINAL=after+final#这个FINAL将作为下一层的x（就是最开始那个x）
        return FINAL

class transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks=nn.ModuleList([pre()]+[BLOCK() for _ in range(6)])
        self.lnfinal=nn.LayerNorm(384)
        self.finalLinear=nn.Linear(384,char_num)
    def forward(self,input):
        for block in self.blocks:
            input=block(input)#这个时候完成了6次trans的迭代，出来还是B 256 384
        input=self.lnfinal(input)
        logits=self.finalLinear(input)
        return logits

#主训练函数：
model=transformer().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
for _ in range(1000):
        egs,ans=makeegs(input_text,256)
        egs=egs.to(device)
        ans=ans.to(device)
        logits=model(egs)#egs是pre的输入
        loss=F.cross_entropy(logits.view(-1,char_num),ans.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

#存参数
torch.save(model.state_dict(),"model.pth")