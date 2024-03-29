﻿# coding: UTF-8
###### configuration.py #####
#                                           Last Update:  2020/4/13
#
# 各種設定用ファイル
# インスタンスはcnfとして生成

import numpy as np

# configurationクラス
class Configuration:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self):
        # GAの設定
        self.max_pop    = 50                    # 個体数
        self.h          = 50                             # Hパラメータ

        # 問題設定
        self.prob_dim  = 20                  # 問題の次元数
        self.prob_name = ["F1","F5"]            # 解く問題

        # 実験環境
        self.max_trial = 30              # 試行回数
        self.max_evals = 100000                 # 評価回数(max_pop × max_gen)

        # 入出力設定
        self.path_out   = "./"                  # 出力先フォルダ
        self.log_name   = "_result_" + "GA" + str(self.h)    # ログの出力先フォルダ(path_outの直下)
        self.log_out    = True                  # ログ出力の有無

        

    """ インスタンスメソッド """
    # ランダムシード値設定
    def setRandomSeed(self, seed=1):
        # シード値の固定
        self.seed = seed
        self.rd = np.random
        self.rd.seed(self.seed)
