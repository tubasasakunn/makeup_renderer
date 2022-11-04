import torch

import utils.utils as utils
import torchvision.utils


class  cubicSpline:
    def __init__(self, points):
        # Xが昇順になるようにソートする
        index=torch.argsort(points,dim=0)[:,0]
        #3point
        sorted_points=torch.stack((points[index[0]],points[index[1]],points[index[2]]))
        self.x, self.y = sorted_points[:,0],sorted_points[:,1]
        # フィッティングのために各係数を求めておく
        self.__initialize(self.x, self.y)

    def __initialize(self,x,y):
        xlen = len(x) # 点の数
        N = xlen - 1 # 求めるべき変数の数（＝方程式の数）

        # Xが一致する値を持つ場合例外を発生させる
        if xlen != len(set(x)): raise ValueError("x must be different values")

        matrix = torch.zeros([4*N, 4*N])
        Y = torch.zeros([4*N])

        equation = 0
        for i in range(N):
            for j in range(4):
                matrix[equation, 4*i+j] = torch.pow(x[i]+0.0001, j)
            Y[equation] = y[i]
            equation += 1
        for i in range(N):
            for j in range(4):
                matrix[equation, 4*i+j] = torch.pow(x[i+1]+0.0001, j)
            Y[equation] = y[i+1]
            equation += 1
        for i in range(N-1):
            for j in range(4):
                matrix[equation, 4*i+j] = j*torch.pow(x[i+1]+0.0001, j-1)
                matrix[equation, 4*(i+1)+j] = -j*torch.pow(x[i+1]+0.0001, j-1)
            equation += 1
        for i in range(N-1):
            matrix[equation, 4*i+3] = 3*x[i+1]
            matrix[equation, 4*i+2] = 1
            matrix[equation, 4*(i+1)+3] = -3*x[i+1]
            matrix[equation, 4*(i+1)+2] = -1
            equation += 1
        matrix[equation,3] = 3*x[0]
        matrix[equation,2] = 1
        equation += 1
        matrix[4*N-1,4*N-1] = 3*x[N]
        matrix[4*N-1,4*N-2] = 1

        # Wa=Y => a=W^(-1)Yとして変数の行列を求める
        # その際、逆行列を求めるのにtorch.inversを使う
        eye=torch.eye(matrix.shape[0])
        self.variables = torch.matmul(torch.inverse(matrix+0.0001*eye),Y)

    def fit(self, x):
        """
        引数xが該当する区間を見つけてきて補間後の値を返す
        """
        xlen = len(self.x)
        for index,j in enumerate(self.x):
            if x < j:
                index -= 1
                break
        if index == -1:
            index += 1
        elif index == xlen-1:
            index -= 1
        a3 = self.variables[4*index + 3]
        a2 =  self.variables[4*index + 2]
        a1 = self.variables[4*index + 1]
        a0 = self.variables[4*index + 0]

        result = a3*torch.pow(x+0.00001,3) + a2*pow(x+0.000001,2) + a1*x + a0
        return result