https://deepinsider.jp/issue/deeplearningnext/textgeneration#%E8%A8%93%E7%B7%B4%E3%83%87%E3%83%BC%E3%82%BF%E3%81%AE%E5%85%A5%E6%89%8B を試してみる

- sentence_generator00.py : 一番最初のやつ
- sentence_generator01.py : validation splitを0にして全てのデータを学習に利用するもの  
  これだとearly stoppingが効かなくなるし過学習しまくるのでだめじゃないか?
- sentence_generator02.py : 文章生成するやつ。00をベースに再現