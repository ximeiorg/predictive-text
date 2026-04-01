"""Mixed vocabulary builder: characters + high-frequency words."""

import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json
from tqdm import tqdm


class MixedVocabularyBuilder:
    """Build vocabulary with characters + high-frequency words."""
    
    def __init__(self, vocab_size: int = 5120):
        self.vocab_size = vocab_size
        self.char_vocab: Set[str] = set()
        self.word_vocab: Set[str] = set()
        self.word_freq: Counter = Counter()
        
        # Special tokens
        self.special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]']
        
        # Common Chinese punctuation
        self.punctuation = [
            '。', '，', '！', '？', '；', '：', '"', '"', '、',
            '(', ')', '《', '》', '【', '】', '…', '—', 
            '.', ',', '!', '?', ';', ':', '"', "'", '-',
            ' ', '\n', '\t'
        ]
        
    def extract_high_freq_words(self, 
                                texts: List[str], 
                                min_freq: int = 10,
                                min_length: int = 2,
                                max_length: int = 4) -> List[Tuple[str, int]]:
        """Extract high-frequency words from texts."""
        
        print(f"Extracting high-frequency words from {len(texts)} texts...")
        
        word_pattern = re.compile(r'[\u4e00-\u9fff]{2,4}')
        
        for text in tqdm(texts, desc="Extracting words"):
            words = word_pattern.findall(text)
            for word in words:
                if min_length <= len(word) <= max_length:
                    self.word_freq[word] += 1
        
        # Filter by frequency and quality
        high_freq_words = []
        
        # Common complete words (whitelist)
        common_words = {
            '我们', '你们', '他们', '大家', '自己', '什么', '这个', '那个',
            '时候', '今天', '明天', '昨天', '现在', '以后', '之前', '正在',
            '知道', '看到', '觉得', '想到', '发现', '开始', '结果', '觉得',
            '一起', '一定', '一样', '可能', '应该', '必须', '已经', '还是',
            '不是', '没有', '可以', '就是', '这样', '那样', '怎么', '为什么',
            '因为', '所以', '如果', '但是', '然而', '虽然', '不过', '而且',
            '或者', '还是', '以及', '对于', '关于', '通过', '根据', '按照',
            '时候', '地方', '方面', '问题', '事情', '情况', '状态', '结果',
            '起来', '出来', '进来', '回来', '下去', '上去', '过来', '过去',
            '说道', '问道', '笑道', '喊道', '叫道', '想道', '写道', '回答',
            '当然', '其实', '竟然', '忽然', '终于', '果然', '居然', '显然',
            '仍然', '总是', '经常', '常常', '往往', '有时', '偶尔', '暂时',
            '吕树', '小鱼', '聂廷', '天罗', '地网', '祖安', '李一笑',
            '情绪', '负面', '灵石', '修行', '觉醒', '遗迹', '洛城', '道观',
        }
        
        # Blacklist: incomplete fragments
        blacklist_patterns = [
            r'^的',      # "的..."
            r'^了',      # "了..."
            r'^在',      # "在..."
            r'^有',      # "有..."
            r'^是',      # "是..."
            r'^和',      # "和..."
            r'^与',      # "与..."
            r'^或',      # "或..."
            r'^但',      # "但..."
            r'^而',      # "而..."
            r'^所',      # "所..."
            r'^以',      # "以..."
            r'^为',      # "为..."
            r'^被',      # "被..."
            r'^把',      # "把..."
            r'^让',      # "让..."
            r'^给',      # "给..."
            r'^从',      # "从..."
            r'^到',      # "到..."
            r'^对',      # "对..."
            r'^这',      # "这..."
            r'^那',      # "那..."
            r'^如',      # "如..."
            r'^果',      # "果..."
            r'^因',      # "因..."
            r'^虽',      # "虽..."
            r'^但',      # "但..."
            r'^不',      # "不..."
            r'^也',      # "也..."
            r'^都',      # "都..."
            r'^就',      # "就..."
            r'^还',      # "还..."
            r'^只',      # "只..."
            r'^很',      # "很..."
            r'^太',      # "太..."
            r'^更',      # "更..."
            r'^最',      # "最..."
            r'^已',      # "已..."
            r'^经',      # "经..."
            r'^正',      # "正..."
            r'^将',      # "将..."
            r'^要',      # "要..."
            r'^能',      # "能..."
            r'^会',      # "会..."
            r'^可',      # "可..."
            r'^应',      # "应..."
            r'^该',      # "该..."
            r'的$',      # "...的"
            r'了$',      # "...了"
            r'在$',      # "...在"
            r'有$',      # "...有"
            r'是$',      # "...是"
            r'和$',      # "...和"
            r'与$',      # "...与"
            r'但$',      # "...但"
            r'而$',      # "...而"
            r'所$',      # "...所"
            r'以$',      # "...以"
            r'为$',      # "...为"
            r'被$',      # "...被"
            r'把$',      # "...把"
            r'让$',      # "...让"
            r'给$',      # "...给"
            r'从$',      # "...从"
            r'到$',      # "...到"
            r'对$',      # "...对"
            r'这$',      # "...这"
            r'那$',      # "...那"
            r'如$',      # "...如"
            r'果$',      # "...果"
            r'因$',      # "...因"
            r'虽$',      # "...虽"
            r'但$',      # "...但"
            r'不$',      # "...不"
            r'也$',      # "...也"
            r'都$',      # "...都"
            r'就$',      # "...就"
            r'还$',      # "...还"
            r'只$',      # "...只"
            r'很$',      # "...很"
            r'太$',      # "...太"
            r'更$',      # "...更"
            r'最$',      # "...最"
            r'已$',      # "...已"
            r'经$',      # "...经"
            r'正$',      # "...正"
            r'将$',      # "...将"
            r'要$',      # "...要"
            r'能$',      # "...能"
            r'会$',      # "...会"
            r'可$',      # "...可"
            r'应$',      # "...应"
            r'该$',      # "...该"
            r'着$',      # "...着"
            r'过$',      # "...过"
            r'地$',      # "...地"
            r'得$',      # "...得"
            r'啊$',      # "...啊"
            r'呢$',      # "...呢"
            r'吧$',      # "...吧"
            r'吗$',      # "...吗"
            r'呀$',      # "...呀"
            r'哦$',      # "...哦"
            r'额$',      # "...额"
            r'嗯$',      # "...嗯"
            r'哈$',      # "...哈"
            r'呵$',      # "...呵"
            r'嘿$',      # "...嘿"
            r'哎$',      # "...哎"
            r'唉$',      # "...唉"
            r'哇$',      # "...哇"
            r'喔$',      # "...喔"
            r'嘘$',      # "...嘘"
            r'咦$',      # "...咦"
            r'噢$',      # "...噢"
            r'人$',      # "...人"
            r'个$',      # "...个"
            r'些$',      # "...些"
            r'次$',      # "...次"
            r'点$',      # "...点"
            r'里$',      # "...里"
            r'外$',      # "...外"
            r'上$',      # "...上"
            r'下$',      # "...下"
            r'前$',      # "...前"
            r'后$',      # "...后"
            r'左$',      # "...左"
            r'右$',      # "...右"
            r'中$',      # "...中"
            r'内$',      # "...内"
            r'间$',      # "...间"
            r'时$',      # "...时"
            r'年$',      # "...年"
            r'月$',      # "...月"
            r'日$',      # "...日"
            r'天$',      # "...天"
            r'号$',      # "...号"
            r'者$',      # "...者"
            r'们$',      # "...们"
        ]
        
        blacklist_re = re.compile('|'.join(blacklist_patterns))
        
        for word, freq in self.word_freq.items():
            if freq < min_freq:
                continue
            
            # Whitelist check
            if word in common_words:
                high_freq_words.append((word, freq))
                continue
            
            # Blacklist check
            if blacklist_re.search(word):
                continue
            
            # Additional quality checks
            # 1. Should not have repeating characters
            if len(set(word)) < len(word) * 0.5:
                continue
            
            # 2. Should be meaningful (not too short or too long)
            if len(word) < min_length or len(word) > max_length:
                continue
            
            # 3. Frequency threshold based on length
            length_freq_threshold = {
                2: min_freq * 2,
                3: min_freq * 1.5,
                4: min_freq,
            }
            
            if freq >= length_freq_threshold.get(len(word), min_freq):
                high_freq_words.append((word, freq))
        
        # Sort by frequency
        high_freq_words.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(high_freq_words)} high-frequency words (filtered)")
        
        return high_freq_words
    
    def build_char_vocab(self, texts: List[str], min_freq: int = 1) -> List[str]:
        """Build character vocabulary from texts."""
        
        print(f"Building character vocabulary from {len(texts)} texts...")
        
        char_freq = Counter()
        
        for text in tqdm(texts, desc="Counting characters"):
            for char in text:
                # Only Chinese characters
                if '\u4e00' <= char <= '\u9fff':
                    char_freq[char] += 1
        
        # Filter by frequency
        chars = [char for char, freq in char_freq.items() if freq >= min_freq]
        
        # Sort by frequency (high freq first)
        char_freq_list = [(char, freq) for char, freq in char_freq.items() if freq >= min_freq]
        char_freq_list.sort(key=lambda x: x[1], reverse=True)
        chars = [char for char, freq in char_freq_list]
        
        print(f"Found {len(chars)} unique characters")
        
        return chars
    
    def build_vocab(self, 
                    texts: List[str],
                    char_ratio: float = 0.7,
                    word_min_freq: int = 10) -> Dict[str, int]:
        """Build mixed vocabulary: chars + high-frequency words."""
        
        print(f"\n{'='*50}")
        print("Building Mixed Vocabulary")
        print(f"{'='*50}")
        print(f"Target vocab size: {self.vocab_size}")
        print(f"Character ratio: {char_ratio}")
        
        # Step 1: Extract characters
        chars = self.build_char_vocab(texts)
        
        # Step 2: Extract high-frequency words
        high_freq_words = self.extract_high_freq_words(
            texts, 
            min_freq=word_min_freq,
            min_length=2,
            max_length=4
        )
        
        # Step 3: Allocate vocab space
        special_size = len(self.special_tokens)
        punctuation_size = len(self.punctuation)
        
        remaining_size = self.vocab_size - special_size - punctuation_size
        
        # Characters get 70% of remaining space
        char_vocab_size = int(remaining_size * char_ratio)
        # Words get 30% of remaining space
        word_vocab_size = remaining_size - char_vocab_size
        
        print(f"\nVocabulary allocation:")
        print(f"  Special tokens: {special_size}")
        print(f"  Punctuation: {punctuation_size}")
        print(f"  Characters: {char_vocab_size} (max available: {len(chars)})")
        print(f"  Words: {word_vocab_size} (max available: {len(high_freq_words)})")
        
        # Step 4: Build final vocabulary
        vocab = {}
        idx = 0
        
        # Add special tokens
        for token in self.special_tokens:
            vocab[token] = idx
            idx += 1
        
        # Add punctuation
        for punct in self.punctuation:
            if punct not in vocab:
                vocab[punct] = idx
                idx += 1
        
        # Add characters (top N by frequency)
        selected_chars = chars[:min(char_vocab_size, len(chars))]
        for char in selected_chars:
            if char not in vocab:
                vocab[char] = idx
                idx += 1
        
        # Add high-frequency words (top N)
        selected_words = [word for word, freq in high_freq_words[:word_vocab_size]]
        for word in selected_words:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        
        print(f"\nFinal vocabulary size: {len(vocab)}")
        print(f"  Total characters: {sum(1 for k in vocab if len(k) == 1 and '\u4e00' <= k <= '\u9fff')}")
        print(f"  Total words: {sum(1 for k in vocab if len(k) >= 2)}")
        
        self.vocab = vocab
        
        return vocab
    
    def save_vocab(self, vocab_path: str):
        """Save vocabulary to file."""
        
        vocab_file = Path(vocab_path)
        vocab_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in format: token\tindex
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for token, idx in self.vocab.items():
                f.write(f"{token}\t{idx}\n")
        
        print(f"\nVocabulary saved to {vocab_file}")
        
        # Save statistics
        stats = {
            'vocab_size': len(self.vocab),
            'special_tokens': len(self.special_tokens),
            'punctuation': len(self.punctuation),
            'characters': sum(1 for k in self.vocab if len(k) == 1 and '\u4e00' <= k <= '\u9fff'),
            'words': sum(1 for k in self.vocab if len(k) >= 2),
            'vocab_file': str(vocab_file)
        }
        
        stats_file = vocab_file.parent / f"{vocab_file.stem}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to {stats_file}")
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using mixed vocabulary."""
        
        tokens = []
        
        # Try to match words first (greedy matching)
        i = 0
        while i < len(text):
            # Try to match longest word first (4 chars)
            matched = False
            
            for length in range(4, 1, -1):  # Try 4, 3, 2
                if i + length <= len(text):
                    candidate = text[i:i+length]
                    if candidate in self.vocab:
                        tokens.append(self.vocab[candidate])
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Fall back to character
                char = text[i]
                if char in self.vocab:
                    tokens.append(self.vocab[char])
                else:
                    tokens.append(self.vocab['[UNK]'])
                i += 1
        
        return tokens
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        
        reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        text = []
        for token_id in tokens:
            if token_id in reverse_vocab:
                text.append(reverse_vocab[token_id])
            else:
                text.append('[UNK]')
        
        return ''.join(text)


def build_vocab_from_file(file_path: str, 
                         vocab_size: int = 5120,
                         output_path: str = "data/mixed_vocab.txt",
                         char_ratio: float = 0.7,
                         word_min_freq: int = 10):
    """Build vocabulary from a single file."""
    
    print(f"Loading texts from {file_path}...")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        cache_path = Path("data/cache/data_cache.txt")
        if cache_path.exists():
            print(f"File not found, using cache: {cache_path}")
            file_path = cache_path
        else:
            raise FileNotFoundError(f"Neither {file_path} nor cache {cache_path} exists")
    
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and len(line) >= 5:
                texts.append(line)
    
    print(f"Loaded {len(texts)} lines")
    
    builder = MixedVocabularyBuilder(vocab_size=vocab_size)
    vocab = builder.build_vocab(
        texts,
        char_ratio=char_ratio,
        word_min_freq=word_min_freq
    )
    
    builder.save_vocab(output_path)
    
    return builder


def test_tokenization(builder: MixedVocabularyBuilder, test_texts: List[str]):
    """Test tokenization quality."""
    
    print(f"\n{'='*50}")
    print("Testing Tokenization")
    print(f"{'='*50}")
    
    for text in test_texts:
        tokens = builder.tokenize(text)
        decoded = builder.detokenize(tokens)
        
        print(f"\nOriginal:  {text}")
        print(f"Tokens:    {tokens[:20]}...")
        print(f"Decoded:   {decoded}")
        print(f"Match:     {text == decoded}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build mixed vocabulary")
    parser.add_argument("--input", type=str, default="大王饶命.txt", help="Input text file")
    parser.add_argument("--vocab-size", type=int, default=5120, help="Vocabulary size")
    parser.add_argument("--output", type=str, default="data/mixed_vocab.txt", help="Output vocab file")
    parser.add_argument("--char-ratio", type=float, default=0.7, help="Character ratio (0.0-1.0)")
    parser.add_argument("--word-min-freq", type=int, default=10, help="Minimum word frequency")
    parser.add_argument("--test", action="store_true", help="Test tokenization")
    
    args = parser.parse_args()
    
    builder = build_vocab_from_file(
        args.input,
        vocab_size=args.vocab_size,
        output_path=args.output,
        char_ratio=args.char_ratio,
        word_min_freq=args.word_min_freq
    )
    
    if args.test:
        test_texts = [
            "今天天气很好",
            "我们一起去吃饭",
            "吕树看着吕小鱼",
            "我觉得这个事情有点奇怪",
        ]
        test_tokenization(builder, test_texts)