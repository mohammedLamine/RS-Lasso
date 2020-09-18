import numpy as np
import pandas as pd
from networkx import nx
import numpy.linalg as la


class DataSimulation:
    def __init__(self,p,n_days,t=None,road_props=None, noise_scale=1,t_switch=0,test_frac=0.8):
        self.p=p
        self.n_days=n_days
        if not t is None:
            self.t=t
        else : 
            self.t = np.arange(14.75,20,0.25)
        if not road_props is None:
            self.road_props=road_props
        else : 
            self.road_props= dict(zip([30, 50, 80, 130],np.random.multinomial(self.p,[0,0.25,0.5,0.25])))
        self.noise_scale=noise_scale
        self.t_switch=t_switch
        self.tau = len(self.t)
        self.test_frac = test_frac


    
    def rearange_data(data,p):
        data=data.swapaxes(0,2)
        data=data.swapaxes(1,2)
        data=data.reshape(p,-1)
        return data

    def generate_date_index(n_days):
        start_date = pd.datetime(2020,1,1,15,0,0)
        dinx=pd.date_range('2020-1-1 15:00:00+01:00',periods=4*24*n_days,freq='15min')
        return dinx[(dinx.time>= pd.datetime(2020,1,1,15,0,0).time()) & (dinx.time< pd.datetime(2020,1,1,20,0,0).time()) ]

    def gen_one_instant_speed(max_speed,normal_center=0.9,size=1):
        normal_values = np.random.normal(0,(max_speed-normal_center*max_speed)/2,size)
        return normal_center*max_speed+normal_values

    def run_generation_formula(A,b,w,tau,p,A_R=None,t_switch=None,noise_scale=1):
        data=[]
        cur_A=A
        for i in range(tau-1):
            if t_switch is not None and i>t_switch :
                cur_A=A_R
            x= w-b[i][:]
            noise = np.random.normal(0,noise_scale,size=p)        
            w= b[i+1][:] + cur_A.dot(x) + noise
            data.append(w)
        return np.array(data)       
    def generate_intercept(t,road_props,tau):
        b_t= (2.5**2-(t-17.5)**2)
        b_t=np.reshape(b_t,(1,-1))
        b_t_difference = b_t[0][1:]-b_t[0][0:-1]
        b_p = np.concatenate([DataSimulation.gen_one_instant_speed(max_speed,normal_center=0.9,size=prop) for max_speed,prop in road_props.items()])
        b_p=np.reshape(b_p,(-1,1))
        b_p=b_p.repeat(tau,axis=1)
        b=b_p.T-b_t.T
        return b

    def generate_graph(p):
        g = nx.gnm_random_graph(p, 8*p,directed=True)
        return g
    def generate_A_matrix(g,p):
        A=np.random.uniform(-1,1,size=(p,p))*(np.array([[1  if i in g.adj[j] else 0 for i in range(p)] for j in g.adj])+np.diag([1]*p))
        A=(A.T/la.norm(A,axis=1)).T
        return A

    def generate_data(self):
        self.g = DataSimulation.generate_graph(self.p)
        self.b = DataSimulation.generate_intercept(self.t,self.road_props,self.tau)
        self.A_L = DataSimulation.generate_A_matrix(self.g,self.p)
        self.A_R = DataSimulation.generate_A_matrix(self.g,self.p)
        full_days_data = []
        for i in range(self.n_days):
            w0 = np.concatenate([DataSimulation.gen_one_instant_speed(max_speed,normal_center=0.9,size=prop) for max_speed,prop in self.road_props.items()])
            data= DataSimulation.run_generation_formula(self.A_L,self.b,w0,self.tau,self.p,A_R = self.A_R,t_switch= self.t_switch+1)
            full_days_data.append(data)
        self.full_days_data=np.array(full_days_data)
        return self.full_days_data



    def split_center_data(self):
        full_days_data_train=self.full_days_data[:int(self.test_frac*self.n_days)]
        full_days_data_test=self.full_days_data[int(self.test_frac*self.n_days):]

        full_days_data_train = DataSimulation.rearange_data(full_days_data_train,self.p)
        full_days_data_test  = DataSimulation.rearange_data(full_days_data_test,self.p)

        sim_train_df = pd.DataFrame(data= full_days_data_train,columns=DataSimulation.generate_date_index(self.n_days)[:int(self.test_frac*self.n_days*(self.tau-1))])
        sim_test_df = pd.DataFrame(data= full_days_data_test,columns=DataSimulation.generate_date_index(self.n_days)[int(self.test_frac*self.n_days*(self.tau-1)):])

        intercept = pd.concat([sim_train_df.groupby(pd.to_datetime(sim_train_df.columns).time,axis=1).mean()
        ]*self.n_days,axis=1)
        intercept.columns=DataSimulation.generate_date_index(self.n_days)

        sim_train_intercept = intercept[intercept.columns[:int(self.test_frac*self.n_days)*(self.tau-1)]]
        sim_test_intercept = intercept[intercept.columns[int(self.test_frac*self.n_days)*(self.tau-1):]]

        sim_train_df=sim_train_df-sim_train_intercept
        sim_test_df=sim_test_df-sim_test_intercept
    
        return sim_train_df,sim_train_intercept,sim_test_df,sim_test_intercept


