import numpy as np

class LBFGS:
    def __init__(self, m=10):
        self.m = m
        self.s_list = []
        self.y_list = []
        self.rho_list = []
    def two_loop(self, q):
        if not self.s_list:
            return q
        alphas = []
        q_work = q.copy()
        for s,y,rho in zip(reversed(self.s_list), reversed(self.y_list), reversed(self.rho_list)):
            alpha = rho * np.dot(s,q_work)
            alphas.append(alpha)
            q_work -= alpha*y
        y_last = self.y_list[-1]
        s_last = self.s_list[-1]
        gamma = np.dot(s_last,y_last)/(np.dot(y_last,y_last)+1e-16)
        r = gamma*q_work
        for s,y,rho,alpha in zip(self.s_list,self.y_list,self.rho_list,reversed(alphas)):
            beta = rho * np.dot(y,r)
            r += s*(alpha - beta)
        return r
    
    def update(self, s, y):
        rho = 1.0/(np.dot(y,s)+1e-16)
        if len(self.s_list) == self.m:
            self.s_list.pop(0)
            self.y_list.pop(0)
            self.rho_list.pop(0)
        self.s_list.append(s)
        self.y_list.append(y)
        self.rho_list.append(rho)

# ==============================
# Line search
# ==============================
def line_search(x,f,g,p,loss_func,c1=1e-2,tau=0.5,max_iters=4):
    alpha = 1.0
    f0 = f
    gTp = np.dot(g,p)
    
    for _ in range(max_iters):
        x_new = x + alpha*p
        f_new,_ = loss_func(x_new)
        if f_new <= f0 + c1*alpha*gTp:
            return alpha, f_new
        alpha *= tau
    return alpha, f_new