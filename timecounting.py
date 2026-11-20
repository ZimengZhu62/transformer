import os
import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
#注意这里下面是我自己在调整接口所以输出会和官方输出有一个特殊符长度的位移。
    length=len(split_special_token)
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at+length
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))
#你注意我上面写过注释这个api是被我改过的现在只会返回那个特殊符号结束后的下一个的索引。


def worker(task:tuple):#说白了下面的tasks构造成了（special，chunk）的形式所以你先接进来再拆
    special_tokens,chunk=task

    if len(special_tokens)>0:
        bspecial_tokens=special_tokens[0].encode("utf-8")
        lenspecial=len(bspecial_tokens)
        parts=chunk.split(bspecial_tokens)
    else:
        parts=[chunk]

    counts=defaultdict(int)
    include_dict=defaultdict(set)
    word_freq=defaultdict(int)

    for part in parts:
        for match in re.finditer(PAT, part.decode("utf-8")):
            x=tuple(match.group().encode("utf-8"))
            word_freq[x]+=1
            pairs=list(zip(x[:-1],x[1:]))

            for pair in pairs:
                counts[pair]+=1

            for pair in set(pairs):
                include_dict[pair].add(x)

    return counts,include_dict,word_freq



def train_bpe(input_path: str,vocab_size: int ,special_tokens: list[str]):
    
    token_id_pairs={}
    token_bytes={}

    t0=time.perf_counter()

    for i in range(256):
        token_id_pairs[i]=i
        token_bytes[i]=bytes([i])


    num_processes=8
    chunks=[]

    if len(special_tokens)>0:
        bspecial_tokens=special_tokens[0].encode("utf-8")

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, bspecial_tokens)
            for start,end in zip(boundaries[:-1],boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end-start))
    else:
        with open(input_path,"rb") as f:
            chunks.append(f.read())


    tasks=[(special_tokens,chunk) for chunk in chunks]

    t1=time.perf_counter()
    print("制造任务耗时：",t1-t0)


    #并行
    t5=time.perf_counter()
    from multiprocessing import Pool
    with Pool(processes=num_processes) as pool:
        results=pool.map(worker,tasks)
    t6=time.perf_counter()
    print("第一轮读取总计耗时：",t6-t5)



    countstot=defaultdict(int)
    includetot=defaultdict(set)
    word_freq_tot=defaultdict(int)

    for result in results:
        result1,result2,result3=result

        for key,value in result1.items():
            countstot[key]+=value

        for key,value in result2.items():
            includetot[key].update(value)

        for key,value in result3.items():
            word_freq_tot[key]+=value

    t3tot=0
    t4tot=0

    while len(token_id_pairs)<vocab_size-len(special_tokens):

        if len(countstot)==0:
            break

        t3=time.perf_counter()

        maxnum=max(countstot.values())

        max_pairs=[]
        for key in countstot.keys():
            if countstot[key]==maxnum:
                max_pairs.append(key)

        #频率相同时按照实际bytes做lexicographic tie-break
        to_be_combined=max(
            max_pairs,
            key=lambda pair:(token_bytes[pair[0]],token_bytes[pair[1]])
        )

        p1,p2=to_be_combined
        tokennum_now=len(token_id_pairs)

        token_id_pairs[tokennum_now]=to_be_combined
        token_bytes[tokennum_now]=token_bytes[p1]+token_bytes[p2]

        t4=time.perf_counter()

        #开始更新唯一的表
        affected_words=list(includetot[to_be_combined])

        for word in affected_words:

            freq=word_freq_tot[word]

            list_tuples_before=list(zip(word[:-1],word[1:]))
            list_tuples_before_setted=list(set(list_tuples_before))

            list_after_iteration=[]
            length_before=len(word)
            inx=0

            while inx <length_before-1:

                if (word[inx],word[inx+1])==to_be_combined:
                    list_after_iteration.append(tokennum_now)
                    inx=inx+2

                else:
                    list_after_iteration.append(word[inx])
                    inx=inx+1

            if inx==length_before-1:
                list_after_iteration.append(word[inx])


            list_tuples_after=[
                (list_after_iteration[i],list_after_iteration[i+1])
                for i in range(len(list_after_iteration)-1)
            ]

            list_tuples_after_setted=list(set(list_tuples_after))

            wordnew=tuple(list_after_iteration)


            for tuple1 in list_tuples_after_setted:
                includetot[tuple1].add(wordnew)

            for tuple2 in list_tuples_before_setted:
                includetot[tuple2].discard(word)


            for tuple3 in list_tuples_after:
                countstot[tuple3]+=freq

            for tuple4 in list_tuples_before:
                countstot[tuple4]-=freq


            word_freq_tot[wordnew]+=freq
            del word_freq_tot[word]



        del includetot[to_be_combined]
        del countstot[to_be_combined]
        t5=time.perf_counter()
        t3tot+=(t4-t3)
        t4tot+=(t5-t4)
    print("合并总耗时：",t3tot)
    print("更新表总耗时：",t4tot)



    vocab={}
    merges=[]

    for i in range(256):
        vocab[i]=bytes([i])

    for token_id in range(256,len(token_id_pairs)):
        p1,p2=token_id_pairs[token_id]

        left=vocab[p1]
        right=vocab[p2]

        vocab[token_id]=left+right
        merges.append((left,right))


    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode("utf-8")

    return vocab,merges

print("FILE:", __file__)
print("NAME:", __name__)
print("PID:", os.getpid())


if __name__ == "__main__":
    print("进入主进程")
    a, b = train_bpe(
        "TinyStoriesV2-GPT4-train_副本.txt",
        10000,
        ["<|endoftext|>"]
    )
    print("训练完成")
    print(a[5000])
    print(a[6001])
    print(a[800])
    print(a[410])
    print(a[408])
    print(a[308])

#以下是mac m5的数据：
#制造任务耗时： 0.21414699999877485
#第一轮读取统计耗时： 109.82128008399741
#合并总耗时： 16.433052003063494
#更新表总耗时： 0.8367872523813276
#可以看出就算更新表的算法比较复杂也可以大大节省时间。接近10000轮的合并更新总共才用了17.3s但读取一轮就要了109s。