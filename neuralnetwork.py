import numpy as np  

class Add:
    def __init__(self):
        pass 

    def forward(self,a,b):
        out = a + b 
        return out  

    def backward(self,dout):
        dx = dout * 1  
        dy = dout * 1  
        return dx,dy  

class Mul:
    def __init__(self):
        self.x = None 
        self.y = None 

    def forward(self,a,b):
        self.x = a  
        self.y = b  
        out = a * b  
        return out  

    def backward(self,dout):
        dx = dout * self.y 
        dy = dout * self.x  
        return dx,dy  

class Div:
    def __init__(self):
        self.y = None  

    def forward(self,x):
        self.y = 1/x
        return self.y

    def backward(self,dout):
        a = self.y**2
        dx = -1 * a
        dx = dout * dx 
        return dx   

class Exp:
    def __init__(self):
        self.out = None  

    def forward(self,a):
        self.out = np.exp(a)  
        return self.out

    def backward(self,dout):
        return dout * self.out

class Log:
    def __init__(self):
        self.x = None 

    def forward(self,x):
        self.x = x
        return np.log(x)

    def backward(self,dout):
        dx = dout * (1/self.x) 
        return dx  

class Sigmoid:
    def __init__(self):
        pass 

    def forward(self,x):
        self.mul = Mul()
        self.exp = Exp()
        self.add = Add()
        self.div = Div()

        a = self.mul.forward(x,-1)
        b = self.exp.forward(a)
        c = self.add.forward(b,1)
        d = self.div.forward(c)  

        return d  

    def backward(self,dout):
        ddx     =   self.div.backward(dout)
        dcx,dcy =   self.add.backward(ddx)
        dbx     =   self.exp.backward(dcx)
        dax,day =   self.mul.backward(dbx)

        return dax  

class Softmax:
    def __init__(self):
        pass 

    def forward(self,x):
        exp_a = [
            [] for i in x  
        ]
        self.exps = [
            [] for i in x
        ]
        self.adds = [
            [] for i in x 
        ]
        self.div = Div()
        self.muls = [
            [] for i in x 
        ]
        self.y = [
            [] for i in x 
        ]
        for i,_ in enumerate(x):
            self.exps[i] = Exp()
            exp_a[i] = self.exps[i].forward(x[i])
        # print(exp_a)
        
        exp_sum = 0
        for i,_ in enumerate(x):
            self.adds[i] = Add()
            exp_sum = self.adds[i].forward(exp_sum,exp_a[i])
        # print(exp_sum)
        
        expDiv = self.div.forward(exp_sum)
        # print(expDiv)

        for i,_ in enumerate(x):
            self.muls[i] = Mul()
            self.y[i] = self.muls[i].forward(exp_a[i],expDiv)
        # print(self.y)

        return self.y   

    def backward(self,dout):
        dexp_a_list = [
            [] for i in dout
        ]
        result_list = [
            [] for i in dout 
        ]
        dexp_sum = 0
        for i,_ in enumerate(self.muls):
            dexp_a,dexp_s = self.muls[i].backward(dout[i])    
            dexp_a_list[i]= dexp_a
            dexp_sum += dexp_s   
        dexpDiv = self.div.backward(dexp_sum)  
        for i,_ in enumerate(self.adds):
            _,dexp_aFrom_dexpDiv = self.adds[i].backward(dexpDiv) 
            #   2つの入力の和にexp()をかけた値が逆伝播
            #   dout = dexp_aFrom_dexpDiv + dexp_a_list[i]
            #   dy   = self.exps[i].backward(dout)
            dout    = dexp_aFrom_dexpDiv + dexp_a_list[i]
            # print(dout)
            dy      = self.exps[i].backward(dout)
            # print("逆伝播{}:{}".format(i,dy))
            result_list[i] = dy
        # print(self.y)
        return result_list

class CrossEntropyError:
    def __init__(self):
        pass 

    def forward(self,x,t):
        self.x = x
        self.t = t

        delta = 1e-7 
        self.log_class = [
            [] for i in range(len(x))
        ]
        self.mul_class = [
            [] for i in range(len(x))
        ]
        self.add_class = [
            [] for i in range(len(x))
        ]
        E = 0
        for i,_ in enumerate(x):
            self.log_class[i] = Log()
            self.mul_class[i] = Mul()
            self.add_class[i] = Add()
            log_x   = self.log_class[i].forward(x[i])
            m       = self.mul_class[i].forward(log_x,t[i])
            E       = self.add_class[i].forward(E,m)  
        self.mul2 = Mul()
        E = self.mul2.forward(-1,E)
        
        return E

    def backward(self,dout):
        result = [
            [] for i in range(len(self.x))
        ]
        _,dE = self.mul2.backward(dout)
        for i,_ in enumerate(self.x):
            _,dm    = self.add_class[i].backward(dE)
            da,_    = self.mul_class[i].backward(dm)
            # print(da)
            dlog_x  = self.log_class[i].backward(da)  
            # print(dlog_x)
            # print(-(self.t[i]/self.x[i]))
            result[i] = dlog_x  

        return result 

class SimpleNet:
    def __init__(self,input_size,hidden_size,output_size):
        #   重みリスト作成(入力と隠れ層の間)
        self.weight1 = [
            [
                [] for i in range(input_size)
            ] for i in range(hidden_size)
        ]
        #   重みリスト作成(隠れ層と出力層の間)
        self.weight2 = [
            [
                [] for i in range(hidden_size)
            ] for i in range(output_size)
        ]
        #   隠れ層の入力ノードに対するバイアス
        self.bias1 = [
            [] for i in range(hidden_size)
        ]
        #   出力層の入力ノードに対するバイアス
        self.bias2 = [
            [] for i in range(output_size)
        ]
        #   隠れ層の出力ノード
        self.hidden_out = [
            [] for i in range(hidden_size)
        ]
        #   出力層の出力ノード
        self.output_out = [
            [] for i in range(output_size)
        ]

        #   初期化(重みとバイアス)
        for i,_ in enumerate(self.weight1):
            for j,_ in enumerate(self.weight1[i]):
                self.weight1[i][j] = np.random.randn()
                pass 
            pass 
        for i,_ in enumerate(self.weight2):
            for j,_ in enumerate(self.weight2[i]):
                self.weight2[i][j] = np.random.randn()
                pass
            pass
        for i,_ in enumerate(self.bias1):
            self.bias1[i] = 0.0  
            pass 
        for i,_ in enumerate(self.bias2):
            self.bias2[i] = 0.0  
            pass
        pass 

    def fc1(self,x):
        #   入力×重み用
        self.mul_list = [
            [
                [] for i in x
            ] for i in self.weight1  
        ]
        #   入力×重みの数列の和(更新後の変数、入力×重み)
        self.add_list0 = [
            [
                [] for i in x 
            ] for i in self.weight1
        ]
        #   数列の和＋バイアス
        self.add_list1 = [
            [] for i in self.weight1
        ]
        #   xを活性化関数に通す用
        self.sigmoid_list = [
            [] for i in self.weight1
        ]

        for i,_ in enumerate(self.weight1):
            tmp = 0
            for j,_ in enumerate(self.weight1[i]):
                self.mul_list[i][j] = Mul()
                self.add_list0[i][j]= Add()

                a   = self.mul_list[i][j].forward(x[j],self.weight1[i][j])
                tmp = self.add_list0[i][j].forward(tmp,a)    
                pass 
            self.add_list1[i]   = Add()
            self.sigmoid_list[i]= Sigmoid()    
            self.hidden_out[i]  = self.add_list1[i].forward(tmp,self.bias1[i])
            self.hidden_out[i]  = self.sigmoid_list[i].forward(self.hidden_out[i])
            pass 
        pass 

    def fc2(self):
        self.mul_list_fc2 = [
            [
                [] for i in self.weight2[0]
            ] for i in self.weight2
        ]
        self.add_list0_fc2 = [
            [
                [] for i in self.weight2[0]
            ] for i in self.weight2  
        ]
        self.add_list1_fc2 = [
            [] for i in self.weight2
        ]
        for i,_ in enumerate(self.weight2):
            tmp = 0
            for j,_ in enumerate(self.weight2[i]):
                self.mul_list_fc2[i][j] = Mul()
                self.add_list0_fc2[i][j]= Add()

                a   = self.mul_list_fc2[i][j].forward(
                    self.hidden_out[j],
                    self.weight2[i][j]
                )
                tmp = self.add_list0_fc2[i][j].forward(
                    tmp,
                    a  
                )
                pass 
            self.add_list1_fc2[i]   = Add()
            self.output_out[i]      = self.add_list1_fc2[i].forward(
                tmp,
                self.bias2[i]
            )
        pass 

    def dfc1(self,dout):
        dbias1_list = [
            [] for i in self.weight1  
        ]
        dweight1_list = [
            [
                [] for i in self.weight1[0]
            ] for i in self.weight1
        ]
        for i,_ in enumerate(self.weight1):
            dhidden_out = self.sigmoid_list[i].backward(dout[i])
            dtmp,dbias1 = self.add_list1[i].backward(dhidden_out)
            dbias1_list[i] = dbias1
            for j,_ in enumerate(self.weight1[i]):
                dtmp,da = self.add_list0[i][j].backward(dtmp)
                dx,dweight1 = self.mul_list[i][j].backward(da)
                dweight1_list[i][j] = dweight1
                pass 
            pass  
        return dweight1_list,dbias1_list

    def dfc2(self,dout):
        dbias2_list = [
            [] for i in self.weight2 
        ]
        dweight2_list = [
            [
                [] for i in self.weight2[0]
            ] for i in self.weight2
        ]
        dhidden_out_list = [
            [] for i in self.weight2[0]  
        ]
        for i,_ in enumerate(dhidden_out_list):
            dhidden_out_list[i] = 0.0   
        for i,_ in enumerate(dout):
            dtmp,dbias2 = self.add_list1_fc2[i].backward(dout[i])
            dbias2_list[i] = dbias2
            for j,_ in enumerate(self.weight2[i]):
                dtmp,da = self.add_list0_fc2[i][j].backward(dtmp)
                dhidden_out,dweight2 = self.mul_list_fc2[i][j].backward(da)
                dweight2_list[i][j] = dweight2
                dhidden_out_list[j] += dhidden_out
                pass
            pass
        return dweight2_list,dbias2_list,dhidden_out_list

    def forward(self,x):
        self.softmax = Softmax()
        
        self.fc1(x)
        self.fc2()

        y = self.softmax.forward(self.output_out)
        
        return y  

    def loss(self,x,t): 
        self.cross_entropy_error = CrossEntropyError()

        y   = self.forward(x)  
        # print("ソフトマックス：{}".format(y))
        loss= self.cross_entropy_error.forward(y,t)

        return loss 

    def backward(self,dout):
        result_list = self.cross_entropy_error.backward(dout)
        result_list = self.softmax.backward(result_list)
        dweight2_list,dbias2_list,dhidden_out_list = self.dfc2(result_list)
        dweight1_list,dbias1_list = self.dfc1(dhidden_out_list)
        
        return dweight1_list,dbias1_list,dweight2_list,dbias2_list

if __name__ == "__main__":
    net = SimpleNet(
        input_size=2,
        hidden_size=3,
        output_size=2
    )
    
    epoch   = 10000
    lr      = 0.01 
    for e in range(epoch):
        loss = net.loss(
            x=[0.4,0.1],
            t=[1,0]
        )
        dW1,db1,dW2,db2 = net.backward(1)
        for i,_ in enumerate(dW1):
            for j,_ in enumerate(dW1[i]):
                net.weight1 -= lr * dW1[i][j]
                pass
            pass
        for i,_ in enumerate(db1):
            net.bias1 -= lr * db1[i]
            pass
        for i,_ in enumerate(dW2):
            for j,_ in enumerate(dW2[i]):
                net.weight2 -= lr * dW2[i][j]
                pass
            pass
        for i,_ in enumerate(db2):
            net.bias2 -= lr * db2[i]
            pass
        
        print(loss)
        pass

    




















