# BPE Tokenizer

# Requirements:
# 1. Creates Token Mapper
# 2. Able to Train the Mapper
# 3. Encode / Decode
# 4. Can handle simple English Text

# TODO:
# - [X] Extract and Corpus from true and fake news
# - [X] Convert to Unicode : utf-8 list
# - [X] Train the Mapper 
#   - [X] Find Most Frequent Pairs
#   - [X] Create a replace function for the pairs
#   - [X] Store the Genearted Pairs in a Dictionary
# - [ ] Write the Save and Load Function for the Mapper
# - [X] Write the Encode Function
# - [X] Write the Decode Function
# - [ ] Benchmark the Encode and Decode Function

# - [X] Compare with andrejs.. tokenizer
# - [ ] Test if everything even works

# TODO LATER:
# - [ ] Make it efficient --- very slow as fuck
# - [ ] Sanitize and partition the input corpus during training the same way GPT does [watch anderj's video for context]
# - [ ] Extract and Corpus from multiple datasets and texts

import unicodedata


# ********************************************************
# COPIED FROM ANDJREJ's MINBPE
# ********************************************************

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# ********************************************************

class MyBPETokenizer():
    def __init__(self,):
        self.mapper = {} # tuple(int, int):int
        self.tokens_to_values = {} # int:str [stores the token and their correspoding val]
        
    def _replace(self, replace_pair:tuple[int, int], replace_with_token:int, byte_list:list[int]) -> list:
        le_blist = len(byte_list)
        new_bytelist = []
        idx = 0
        while idx < le_blist:
            if idx + 1 < le_blist and byte_list[idx] == replace_pair[0] and byte_list[idx+1] == replace_pair[1]:
                new_bytelist.append(replace_with_token)
                idx += 2
            else:
                new_bytelist.append(byte_list[idx])
                idx += 1
        return new_bytelist
        
    
    def train(self, corpus:str, max_tokens:int = 1000):
        assert max_tokens > 256
        self.max_tokens = max_tokens
        self.available_tokens = max_tokens - 256
        
        for i in range(256):
            # Examine why the utf-8 values for 1-31 are like that [`\x00`, `x\01` ...]
            self.tokens_to_values[i] = bytes([i]).decode("utf-8", errors="replace")
        
        byte_list = corpus.encode("utf-8", errors="replace")
        new_token_id = 256
        while len(byte_list) > 1:
            pair_list = {}
            for pair in zip(byte_list[:], byte_list[1:]):
                pair_list[pair] = pair_list.get(pair, 0) + 1

            replace_pair, _ = max(pair_list.items(), key= lambda x: x[1]) # no use of actual occurence count

            self.mapper[replace_pair] = new_token_id
            newbyte_list = self._replace(
                byte_list=byte_list, 
                replace_pair = replace_pair, 
                replace_with_token= new_token_id
            )
            self.tokens_to_values[new_token_id] = self.tokens_to_values[replace_pair[0]] + self.tokens_to_values[replace_pair[1]]
            new_token_id += 1
            if new_token_id >= max_tokens:
                break
            byte_list = newbyte_list
    
    def save(self, filepath:str):
        """
        Generates 2 Files by names : filepath.map, filepath.values
        """
        with open(filepath + ".map", mode="w", encoding="utf-8") as map_file:
            map_file.write(f"{self.available_tokens}\n")
            for map, value in self.mapper.items():
                map_file.write(f"({map[0]})({map[0]}) -> {value}\n")
        
        with open(filepath + ".values", mode="w", encoding="utf-8") as value_file:
            value_file.write(f"{len(self.tokens_to_values)}\n")
            for k,v in self.tokens_to_values.items():
                print(f"{int(k)} -|> {v}\n")
                value_file.write(f"{int(k)} -|> {render_token(v.encode("utf-8"))}\n")
    
    def load(self, filepath:str):
        """
        Loads the files filepath.map and filepath.values into mapper and tokens_to_values respectively
        """
        with open(filepath + ".map", mode="r", encoding="utf-8") as map_file:
            total_tokens = int(map_file.readline())
            for line in map_file.readlines():
                k, v = line.split(" -> ")
                k1, k2 = k.split(")(")
                self.mapper[(int(k1[1:]), int(k2[:-1]))] = int(v)
        
        for i in range(256):
            # Examine why the utf-8 values for 1-31 are like that [`\x00`, `x\01` ...]
            self.tokens_to_values[i] = bytes([i]).decode("utf-8", errors="replace")
            
        with open(filepath + ".values", mode="r", encoding="utf-8") as value_file:
            le = int(value_file.readline())                
            for line in value_file.readlines():
                k,v = line.split(" -|> ")
                self.tokens_to_values[int(k)] = str(v).strip("\n")
        
    
    def encode(self, text:str) -> list[int]:
        byte_list = list(text.encode("utf-8", errors="replace"))
        for pair in self.mapper.keys():
            byte_list = self._replace(byte_list=byte_list, replace_pair=pair, replace_with_token=self.mapper[pair])
        return byte_list
        
    def decode(self, tokens:list[int]) -> str:
        out = ""
        for t in tokens:
            out += self.tokens_to_values[t]
        return out
    

if __name__ == "__main__":
    tokenizer = MyBPETokenizer()
    
    corpus = ""
    with open("./data/#synthetic/tokenizer_corpus.txt", mode="r", encoding="utf-8") as file:
        corpus = '\n'.join(file.readlines())
    
    tokenizer.train(corpus=corpus[:100], max_tokens= 256 + 10)
    tokenizer.save("./data/#synthetic/tokenizer_v0")
    
    new_tokenizer = MyBPETokenizer()
    new_tokenizer.load("./data/#synthetic/tokenizer_v0")
    
    print(f"Old: {tokenizer.encode("hello ")} New: {new_tokenizer.encode("hello ")}")
    print(f"Old: {tokenizer.decode(tokenizer.encode("hello "))} New: {new_tokenizer.decode(new_tokenizer.encode("hello "))}")
    print(f"Old: {tokenizer.decode(tokenizer.encode("damn sone"))} New: {new_tokenizer.decode(new_tokenizer.encode("damn sone"))}")