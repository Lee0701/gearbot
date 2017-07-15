import sys
sys.path.insert(0, './chatbot')

import logging
import operator
import pickle
import re
import time

from telegram.ext import CommandHandler
from telegram.ext import Updater

import chatbot
import spacing


class Gearbot:
    def __init__(self, path):
        token_file = open(path, 'r')
        self.api_token = token_file.readline().strip()
        print('Bot API Token : ' + self.api_token)
        self.updater = Updater(token=self.api_token)
        token_file.close()

        self.dispatcher = self.updater.dispatcher

        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s'
                                   '- %(message)s',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.timeout = 10

    def init_chatbot(self, data_path, weights_path):
        self.chatbot = chatbot.Chatbot()
        self.chatbot.load_data(data_path)
        self.chatbot.build_model()
        self.chatbot.load_weights(weights_path)

    def load_ranking(self, path):
        f = open(path, 'rb')
        self.ranking = pickle.load(f)

    def chat(self, bot, update, args):
        user = update.message.from_user

        if time.time() - time.mktime(update.message.date.timetuple()) > self.timeout:
            print('Message from ' + user.username + ' has discarded, TIMEOUT')
            return

        q = ' '.join(args)
        if q != '':
            print('Got question from ' + user.username)
            print(q, end='')
            ans = self.chatbot.predict(q)
            ans = spacing.fix_spacing(ans)
            print(' => ' + ans)
            bot.send_message(chat_id=update.message.chat_id,
                             text=ans,
                             reply_to_message_id=update.message.message_id)
        else:
            bot.send_message(chat_id=update.message.chat_id,
                             text="/chat <말> 의 형식으로 기어봇에게 말을 걸어보세요!",
                             reply_to_message_id=update.message.message_id)

    def teach(self, bot, update, args):
        user = update.message.from_user

        if time.time() - time.mktime(update.message.date.timetuple()) > self.timeout:
            print('Message from ' + user.username + ' has discarded, TIMEOUT')
            return

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

            if user.username in self.ranking:
                self.ranking[user.username] = self.ranking[user.username] + 1
            else:
                self.ranking[user.username] = 1

            with open('./chatbot/etc/cont_ranking.pkl', 'wb') as f2:
                pickle.dump(self.ranking, f2)

            print(qna[0] + ' => ' + qna[1])
            with open('./chatbot/traindata/cont_data.txt', 'a') as file:
                file.write(qna[0] + '\n')
                file.write(qna[1] + '\n\n')
            bot.send_message(chat_id=update.message.chat_id,
                             text=user.username + "님! 기여해 주셔서 정말 감사합니다.\n",
                             reply_to_message_id=update.message.message_id)

    def rank(self, bot, update):
        user = update.message.from_user

        if time.time() - time.mktime(update.message.date.timetuple()) > self.timeout:
            print('Message from ' + user.username + ' has discarded, TIMEOUT')
            return

        txt = '기여 랭킹입니다.\n'

        sorted_rank = sorted(self.ranking.items(), key=operator.itemgetter(1))[::-1]

        i = 1
        for username, count in sorted_rank:
            txt += str(i) + '위 ' + username + ': ' + str(count) + '회\n'
            i = i + 1

        bot.send_message(chat_id=update.message.chat_id,
                         text=txt)

    def add_handlers(self):
        chat_handler = CommandHandler('chat', self.chat, pass_args=True)
        teach_handler = CommandHandler('teach', self.teach, pass_args=True)
        rank_handler = CommandHandler('rank', self.rank)
        self.dispatcher.add_handler(chat_handler)
        self.dispatcher.add_handler(teach_handler)
        self.dispatcher.add_handler(rank_handler)

    def start_pool(self):
        print('Start pooling')
        self.updater.start_polling()
        self.updater.idle()