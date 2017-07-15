from konlpy.tag import Mecab

def fix_spacing(sentence):
    hn = Mecab()
    poses = hn.pos(sentence)

    fixed_sentence = ''

    for pos in poses:
        splited = pos[1].split('+')

        to_attach = lambda p : p.startswith(('J', 'E')) or p == 'SF'

        if any(to_attach(p) for p in splited):
            fixed_sentence += pos[0]
        else:
            fixed_sentence += ' ' + pos[0]

    return fixed_sentence
