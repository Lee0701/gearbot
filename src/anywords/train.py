import sys
import anywords

bot = anywords.Anywords()
bot.load_data('./traindata/data.txt', 50)
bot.build_model()
bot.compile_model()

epochs = int(sys.argv[1])
batch_size = 20

bot.train_model(batch_size, epochs, './traindata/trained_weights')

while True:
    q = input('> ')[0]
    print(bot.predict(q))
