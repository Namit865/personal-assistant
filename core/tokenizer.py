def get_pair_counts(sentence):
    counts = {}

    for idx in range(len(sentence) - 1):

        current_pair = (sentence[idx], sentence[idx + 1])

        if current_pair in counts:
            counts[current_pair] += 1
        else:
            counts[current_pair] = 1

    return counts


def merge(sentence, best_pair, new_token="X"):
    sentence = list(sentence)
    skip = False
    result = []

    for i in range(len(sentence)):
        if skip:
            skip = False
            continue

        if (
            i < len(sentence) - 1
            and (sentence[i], sentence[i + 1]) == best_pair
        ):
            result.append(new_token)
            skip = True
        else:
            result.append(sentence[i])

    return result


def train_bpe(sentence, num_merges):
    sentence = list(sentence)
    result = []

    for i in range(num_merges):
        get_counts = get_pair_counts(sentence)

        if len(get_counts) > 0:
            best_pair = max(get_counts, key=get_counts.get)

            new_token = f"X_{i + 1}"

            sentence = merge(sentence, best_pair, new_token)

            result.append((new_token, best_pair))

        else:
            print("Maximum len reached for merge")
            break

    return sentence, result


def encode(sentence, merges):
    sentence = list(sentence)

    for tokens, pair in merges:
        sentence = merge(sentence, pair, tokens)

    return sentence


def decode(token, merges):
    token = list(token)

    for tok, pair in reversed(merges):
        result = []
        for toks in token:
            if toks == tok:
                result.append(pair[0])
                result.append(pair[1])
            else:
                result.append(toks)
        token = result
    return token


def build_token_ranges(text, merges):
    ranges = []
    pos = 0
    text = list(text)

    tokens = encode(text, merges)

    for tok in tokens:
        decode_tokens = decode([tok], merges)
        length = len(decode_tokens)
        ranges.append((tok, pos, pos + length))

        pos += length

    return ranges


def find_token_for_char(ranges, ans_idx):
    for tok, start, end in ranges:
        if ans_idx >= start and ans_idx < end:
            return tok, start, end
