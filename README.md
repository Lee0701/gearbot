# gearbot
Machine learning chatting bot which runs on Telegram.
머신러닝 하는 텔레그램 채팅봇.

https://t.me/g34rbot

# 개요

Keras 에서 seq2seq 모델을 기반으로 말 -> 말 형식의 데이터를 학습하는 봇입니다.

# 명령어

`/chat <말>` : 기어봇에게 <말>을 입력하여 출력을 전송하게 합니다.
`/teach "<말1>" "<말2>"` : 기어봇 서버에 <말1> -> <말2> 형식의 데이터를 전송합니다. 명령어 실행 즉시 반영되지 않습니다.
`/rank` : teach 명령어 횟수에 따른 순위를 전송하게 합니다.
