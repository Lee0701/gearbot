import numpy as np
from konlpy.tag import Mecab

from one_hot_encode import encode, encode_vals

hn = Mecab()

fill_char = ';'
end_char = 'END'


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def get_key(content):
    updated = []
    for line in content:
        updated += hn.morphs(line)
    return [fill_char, end_char] + unique(updated)


def load_data(path):
    with open(path) as file:
        content = file.readlines()
        updated = []
        for line in content:
            if line != '\n':
                line = line.strip()
                updated.append(line)
        key = get_key(updated)
        updated = [encode_vals(hn.morphs(line) + [end_char], key)
                   for line in updated]
        max_len = len(max(updated, key=len))

        updated2 = []
        for line in updated:
            rem = max_len - len(line)
            if rem > 0:
                line.extend(np.tile(encode(fill_char, key), (rem, 1)))
            updated2.append(line)

        questions = []
        answers = []
        for i in range(0, len(updated2), 2):
            questions.append(updated2[i][::-1])
            answers.append(updated2[i+1])
        return (questions, answers, max_len, key)
