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



#我会把char_num作为总类数
char_num=len(chars)
block_size=256

class pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(char_num,384)#每个token变成了384维的vec
        self.position=nn.Parameter(torch.zeros(1,256,384))
    def forward(self,input):#我现在希望是B（1）*T_user的input
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


#这个时候x的默认输入就是B（目前是1了），T_user，384
    def forward(self,x):
        T=x.shape[1]
        B=x.shape[0]
        #第一轮残差

        input1=self.ln1(x)#B T_user 384
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

#用户输入：
length=input("您要生成多长的内容？")
usertxt=input("请输入文本(最多输入256个字母):")
usernum=[]
for i in usertxt:#splitlines是按照换行符切字符串。而字符串本身就可以遍历
    usernum.append(stoi[i])
T_user=len(usertxt)
usernum=torch.tensor(usernum,device=device).view(1,T_user)
#usernum此时已经是数字序列

model=transformer().to(device)
model.load_state_dict(torch.load("model4000.pth",map_location=device))
model.eval()
T=T_user
while T <256:
    logits=model(usernum)#出来还是1*T_user*char_num
    probs=F.softmax(logits,dim=2)
    ix_next=torch.multinomial(probs[0,-1,:],num_samples=1).view(1,1)#multinomial就是按index的概率抽index,并装到tensor里
    usernum=torch.cat([usernum,ix_next],dim=1)
    T=T+1
result1=usernum
for _ in range(int(length)+T_user-256):
    logits=model(usernum)
    probs=F.softmax(logits,dim=2)
    ix_next=torch.multinomial(probs[0,-1,:],num_samples=1).view(1,1)
    usernum=torch.cat([usernum[:,1:],ix_next],dim=1)
    result1=torch.cat([result1,ix_next],dim=1)
finalresult_num=result1[0].tolist()
finalresult_txtt=[]
for i in finalresult_num:
    finalresult_txtt.append(itos[i])
finalresult_txt=''.join(finalresult_txtt)
print(finalresult_txt)

#4000的loss约为2.6