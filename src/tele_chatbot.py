import gearbot

bot = gearbot.Gearbot('../etc/api_token.txt')
bot.init_chatbot(data_path='./chatbot/traindata/data.txt',
                 weights_path='./chatbot/traindata/trained_weights')
bot.load_ranking('./chatbot/etc/cont_ranking.pkl')
bot.add_handlers()
bot.start_pool()
