#tokenizer已经在a1下面了，叫做vocab.json和merges.json
import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import json
import time
from collections import defaultdict
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


#下面是这一文件的主要目标
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):#vocab and merges are from the function above.
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        #vocab: dict[int, bytes]
        #merges: list[tuple[bytes, bytes]]
        #special_tokens: list[str] | None = None
        i=len(vocab)
        if special_tokens:
            for special in special_tokens:
                special=special.encode("utf-8")
                if special in vocab.values():
                    continue
                else:
                    vocab[i]=special
                    i+=1
        self.vocabinverse={v:k for k,v in self.vocab.items()}
        self.vocabnew=defaultdict(list)#这个的key是某一个byte而value是合成自己的俩东西里有一个是key的byte的int的list。
        for i in self.vocab.keys():
            if 256<=i<256+len(self.merges):
                left,right=self.merges[i-256]
                self.vocabnew[left].append(i)
                self.vocabnew[right].append(i)
        self.mergesdict={self.merges[i]:i for i in range(len(self.merges))}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        #vocab_filepath: str
        #merges_filepath: str
        #special_tokens: list[str] | None = None

#这个对应上面我让ai写的存储方式。
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)
    def encode(self, text: str):
        totlist=[]
        def encodeword(x):#x应该是tuple(bytes)
            #新版代码达到了旧版8倍速度
            xlisted=list(x)    
            i=0
            templist=[]
            templistfinal=[]
            for l in xlisted:
                templist.extend(self.vocabnew[l])
            templist=list(sorted(set(templist)))
            for m in templist:
                templistfinal.append(self.merges[m-256])
#templistfinal里面就是，有我这个x的这些byte们的那些tuple(注意和merges的value一样)
#templist里面则是有我这个x的这些byte们的那些int（int代表了vocab的key）
            length=len(templistfinal)
            while True:
                j=0
                xnew=[]
                k=[]
                rpin=-1
                judge=0
                for r in range(len(templistfinal)):
                    j=0
                    while j<len(xlisted)-1:
                        if (xlisted[j],xlisted[j+1])==templistfinal[r]:
                            k.append(j)
                            rpin=r
                            j+=2
                        else:
                            j+=1
                    if k!=[]:
                        break
                if k==[]:
                     break;
                else:
                    pointer=0
                    while pointer<len(xlisted):
                        if pointer not in k:
                            xnew.append(xlisted[pointer])
                            pointer+=1 
                        else:
                            xnew.append(self.vocab[self.mergesdict[templistfinal[rpin]]+256])
                            pointer+=2
                                                
                xlisted=xnew
                templist=[]
                templistfinal=[]
                for l in xlisted:
                    templist.extend(self.vocabnew[l])
                templist=list(sorted(set(templist)))
                for m in templist:
                    templistfinal.append(self.merges[m-256])
                length=len(templistfinal)
            xindex=[self.vocabinverse[bts] for bts in xlisted]
            return xindex
            
            
        if self.special_tokens:
            #在训练tokenizer的时候，并行是只能用一个special切的。但这里不一定只用一个切
            sortlist=sorted(self.special_tokens,key=len,reverse=True)
            escaped=[re.escape(x) for x in sortlist]
            pattern="("+"|".join(escaped)+")"
            parts_uncoded=re.split(pattern,text)
            parts=[txt.encode("utf-8") for txt in parts_uncoded]
            special_encoded=[sp.encode("utf-8") for sp in self.special_tokens]
        else:
            parts=[text.encode("utf-8")]#这一大段都是照搬原来worker的函数而当初worker接受的chunk是由rb读出来的也就是是bytes。
            special_encoded=[]
        for part in parts:
            if part in special_encoded:
                totlist.extend([self.vocabinverse[part]])
            else:
                for match in re.finditer(PAT, part.decode("utf-8")):
                    x=tuple(bytes([b]) for b in match.group().encode("utf-8"))
                    totlist.extend(encodeword(x))
        return totlist
    def encode_iterable(self, iterable):
        for txt in iterable:#for循环相当于自动一次一次调用next
            yield from self.encode(txt)#yield from list指的是从这个list里一个一个拿
            #然后就是，内层迭代完后，外面的for才会继续next一次，然后再内层迭代完后外面的for才会再推一次。
#总之iterable的输入代表一个迭代器，里面是文本。注意with open...as f的那个f就是迭代器，也即可以print(next(f))然后就
#可以把open里的东西一行一行打出来。至于如果把f交到上面的函数，那么过程是先处理一行，输出效果是把encode好的token一个一个打出来；然后再处理下一行这样
    def decode(self, ids: list[int]):
        result_binary=[]
        for id in ids:
            this=self.vocab[id]
            result_binary.append(this)
        result=b"".join(result_binary)
        resultfinal=result.decode("utf-8",errors="replace")
        return resultfinal



