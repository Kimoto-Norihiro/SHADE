# coding: UTF-8
###### optimizer.py #####
#                                           Last Update:  2020/4/13
#
# 遺伝的アルゴリズムの詳細アルゴリズムファイル
# インスタンスはoptとして生成

# 他ファイル,モジュールのインポート
import function as fc
import numpy as np
import copy

#個体のクラス
class Solution:
    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc, parent=None):
        self.cnf, self.fnc, self.x, self.f = cnf, fnc, [], 0.
        # 個体の初期化
        if parent == None:
            self.x = [self.cnf.rd.uniform(self.fnc.axis_range[0], self.fnc.axis_range[1]) for i in range(self.cnf.prob_dim)]
        # 親個体のコピー
        else:
            self.x = [parent.x[i] for i in range(self.cnf.prob_dim)]
        # リスト -> ndarray
        self.x = np.array(self.x)

# SHADE
class SHADE:

    """ コンストラクタ """
    # 初期化メソッド
    def __init__(self, cnf, fnc):
        self.cnf = cnf      # 設定
        self.fnc = fnc      # 関数
        self.pop = []       # 個体群
        self.nextpop = []   # 次世代個体
        self.mcr = []
        self.mf = []
        self.scr = []
        self.sf = []
        self.delta_f = []
        self.k = 0

    """ インスタンスメソッド """
    # 初期化
    def initializeSolutions(self):
        for i in range(self.cnf.max_pop):
            self.pop.append(Solution(self.cnf, self.fnc))
            self.getFitness(self.pop[i])

        for i in range(self.cnf.h):
            self.mcr.append(0.5)
            self.mf.append(0.5)


    # 次世代個体群生成
    def getNextPopulation(self):
        for i in range(self.cnf.max_pop):
            #パラメータ設定
            r = self.cnf.rd.randint(0,self.cnf.h-1)

            cr = self.cnf.rd.normal(loc = self.mcr[r],scale = 0.1)
            if cr<0:
                cr = 0
            if cr>1:
                cr = 1

            f_param = self.cnf.rd.cauchy(self.mf[r],0.1)
            if f_param<0:
                f_param = 0
            if f_param>1:
                f_param = 1

            p = 2/self.cnf.max_pop + self.cnf.rd.rand() * (0.2-2/self.cnf.max_pop)

            parent = self.pop[i]
            v = self.current_to_best_1(parent,p,f_param)
            o = self.generateOffspring(parent, v, cr)
            self.getFitness(o)
            if parent.f > o.f:
                self.nextpop.append(o)
            else:
                self.nextpop.append(parent)

            if parent.f > o.f:
                #self.A.append(parent)
                self.sf.append(f_param)
                self.scr.append(cr)
                self.delta_f.append(abs(parent.f-o.f))

        if self.sf: 
            if self.scr:
                sum_f = sum(self.delta_f)

                
                mf_deno = np.dot(np.array(self.delta_f)/sum_f , np.power(np.array(self.sf),2))
                mf_frac = np.dot(np.array(self.delta_f)/sum_f , np.array(self.sf))
                new_mf = mf_deno/mf_frac

                new_mcr = np.dot(np.array(self.delta_f)/sum_f , np.array(self.scr))

                self.mf[self.k] = new_mf
                self.mcr[self.k] = new_mcr

                if self.k == self.cnf.h-1:
                    self.k = 0
                else:
                    self.k +=1

             
        self.pop = [self.nextpop[i] for i in range(self.cnf.max_pop)]
        self.nextpop = []
        self.scr = []
        self.sf = []
        self.delta_f = []


    # 子個体の生成
    def generateOffspring(self, p1, p2, cr):
        o1 = self.binominalXover(Solution(self.cnf, self.fnc, parent=p1), Solution(self.cnf, self.fnc, parent=p2), cr)
        return o1
    
    # binominal交叉
    def binominalXover(self, o1, o2, cr):
        xpoint = self.cnf.rd.randint(0, self.cnf.prob_dim)

        for i in range(self.cnf.prob_dim):
            if (self.cnf.rd.rand() <= cr) or i==xpoint:
                tmp = o1.x[i]
                o1.x[i] = o2.x[i]
                o2.x[i] = tmp

        return o1
    
    # 評価値fの計算
    def getFitness(self, solution):
        solution.f = self.fnc.doEvaluate(solution.x)

    #変異戦略
    def current_to_best_1(self, p1, p, f_param):
        k = self.cnf.rd.randint(0,int((self.cnf.max_pop)*p) - 1)
        v = Solution(self.cnf, self.fnc, parent=p1)
        pbest = sorted(self.pop, key=lambda t:t.f)[k]

        r1, r2 = self.rand_ints_nodup(0,self.cnf.max_pop,2,pbest)
        v.x += f_param * (pbest.x - p1.x)
        v.x += f_param * (self.pop[r1].x - self.pop[r2].x)

        return v

    #重複なし組み合わせ
    def rand_ints_nodup(self,a, b, k, pbest):
        ns = []
        while len(ns) < k:
            n = self.cnf.rd.randint(a, b)
            if not n in ns:
                if self.pop[n]!=pbest:

                    ns.append(n)
        return ns