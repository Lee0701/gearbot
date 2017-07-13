import re
import operator

import numpy as np

import pickle

from konlpy.tag import Mecab

from seq2seq.models import Seq2Seq

from one_hot_encode import encode_vals, decode_vals
import train_data

from telegram.ext import Updater
from telegram.ext import CommandHandler
import logging

hn = Mecab()

updater = Updater(token='380828376:AAGyyC1xPGhRAI2o0EV0sOTtHC-dOAxILys')
dispatcher = updater.dispatcher

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s'
                           '- %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

questions, answers, max_len, key = train_data.load_data('./data.txt')

input_length = max_len
input_dim = len(key)

output_length = max_len
output_dim = len(key)

hidden_dim = output_dim * 2

model = Seq2Seq(input_length=input_length,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_length=output_length,
                output_dim=output_dim,
                depth=1)
model.load_weights('trained_weights')

f = open('cont_ranking.pkl', 'rb')
ranking = pickle.load(f)

def chat(bot, update, args):
    user = update.message.from_user
    q = ' '.join(args)
    if q != '':
        print('Got question from ' + user.username)
        print(q, end='')
        q_tok = hn.morphs(q)
        q_tok += [';'] * (max_len - len(q_tok))
        q_res = np.array([encode_vals(q_tok, key)[::-1]])
        result = model.predict(q_res)
        dec_result = decode_vals(result[0], key)
        ans = ' '.join(list(filter(lambda w: w != ';', dec_result)))
        print(' => ' + ans)
        bot.send_message(chat_id=update.message.chat_id,
                         text=ans,
                         reply_to_message_id=update.message.message_id)
    else:
        bot.send_message(chat_id=update.message.chat_id,
                         text="/chat <말> 의 형식으로 기어봇에게 말을 걸어보세요!",
                         reply_to_message_id=update.message.message_id)


def teach(bot, update, args):
    com_args = ' '.join(args)
    qna = re.findall('"([^"]*)"', com_args)
    if len(qna) != 2:
        bot.send_message(chat_id=update.message.chat_id,
                         text="/teach \"질문\" \"질문에 대한 대답\" 의 형식으로 기어봇에게"
                              " 학습 데이터를 기여하세요!\n"
                              "(큰 따옴표는 필수입니다.)",
                         reply_to_message_id=update.message.message_id)
    else:
        user = update.message.from_user
        print('Got train data from ' + user.username)

        if user.username in ranking:
            ranking[user.username] = ranking[user.username] + 1
        else:
            ranking[user.username] = 1
        
        with open('cont_ranking.pkl', 'wb') as f2:
            pickle.dump(ranking, f2)

        print(qna[0] + ' => ' + qna[1])
        with open('cont_data.txt', 'a') as file:
            file.write(qna[0] + '\n')
            file.write(qna[1] + '\n\n')
        bot.send_message(chat_id=update.message.chat_id,
                         text=user.username + "님! 기여해 주셔서 정말 감사합니다.\n",
                         reply_to_message_id=update.message.message_id)


def rank(bot, update, args):
    txt = '기여 랭킹입니다.\n'

    sorted_rank = sorted(ranking.items(), key=operator.itemgetter(1))[::-1]
    
    i = 1
    for username, count in sorted_rank:
        txt += str(i) + '위 ' + username + ': ' + str(count) + '회\n'
        i = i + 1
    
    bot.send_message(chat_id=update.message.chat_id,
                     text=txt)
                     


chat_handler = CommandHandler('chat', chat, pass_args=True)
teach_handler = CommandHandler('teach', teach, pass_args=True)
rank_handler = CommandHandler('rank', rank, pass_args=True)
dispatcher.add_handler(chat_handler)
dispatcher.add_handler(teach_handler)
dispatcher.add_handler(rank_handler)

print('Start pooling')
updater.start_polling()
updater.idle()
