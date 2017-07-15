import sys
import chatbot

bot = chatbot.Chatbot()
bot.load_data('./traindata/data.txt')
bot.build_model()
bot.compile_model()

epochs = int(sys.argv[1])
batch_size = 20

bot.train_model(batch_size, epochs, './traindata/trained_weights')

while True:
    q = input('> ')
    print(bot.predict(q))
