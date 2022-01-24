from ray import tune,init
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import jetnet 
from markdown import markdown
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import hist
import vector
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
from hist import Hist
import ctypes
from optparse import OptionParser
import inspect
plt.style.use(hep.style.ROOT)
import mplhep as hep
plt.style.use(hep.style.ROOT)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import sys
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch import autograd
from torch.autograd import grad
from  torch.cuda.amp import autocast
import time
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.schedulers import HyperBandScheduler,AsyncHyperBandScheduler
import nevergrad as ng
import sys
import json

plt.style.use(hep.style.ROOT)
#from functools import partial
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class NF():
    def __init__(self,data,config,metrics={"w1p":[],"w1m":[],"w1efp":[],"step":[],"loss":[]},hyperopt=False):
        self.config=config  
        self.data=data
        self.hyperopt=hyperopt  
        self.flow=self.build_model()
        self.set_training()
        self.metrics=metrics 
        self.train() 
        self.plot()
        print(markdown(pd.DataFrame(self.metrics)))

    def subnet(self,dims_in, dims_out):
        network=[]
        nodes=self.config["network_nodes"]
        network.append(nn.Linear(dims_in,nodes))
        if self.config["activation"]=='relu' :
            act=nn.ReLU()
        elif self.config["activation"]=='lrelu':
            act=nn.LeakyReLU()
        elif self.config["activation"]=='tanh':
            act=nn.Tanh()
        elif self.config["activation"]<1:
                act=nn.ReLU()
        elif self.config["activation"]<2:
                act=nn.LeakyReLU()
        ##Commented out because performs badly
        # elif self.config["activation"]<3:
        #         act=nn.Sigmoid() 
        elif self.config["activation"]<3:
                act=nn.Tanh()
        
        else:
            print(int(self.config["activation"]))
            raise CustomError("Unknown activation function")
        network.append(act)

        for k in range(int(self.config["network_layers"]-1)):
            layer=nn.Linear(nodes,nodes)
            torch.nn.init.xavier_uniform_(layer.weight)
            network.append(layer)
            network.append(act)
        layer=nn.Linear(nodes,dims_out)
        torch.nn.init.zeros_(layer.weight)
        network.append(layer)
        return nn.Sequential(*network)    
        
    def build_model(self):
        inn = Ff.SequenceINN(90).to(device)
        for k in range(int(self.config["coupling_layers"])):       
                inn.append(Fm.AllInOneBlock,  subnet_constructor=self.subnet,permute_soft=self.config["permute_soft"])
        return inn
    
    def set_training(self):  
        if  self.config["opt"]=="adam":
            self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.config["lr"])
        else:#elif  self.config["opt"]=="adam":
            self.optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.config["lr"],weight_decay=self.config["wdecay"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config["lrdecay"])

    def train(self,scheduler=None,patience=20):
        #trains and evaluates the model
        torch.manual_seed(0)
        self.dataloader = DataLoader(self.data,int(self.config["batch_size"]),shuffle=True)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.min_loss=np.inf
        self.losses=[]
        self.flow.to(device)
        for step in range(self.config["max_steps"]): 
            self.losses.append(0)
            for id_batch, (x_batch) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                x_batch=x_batch.to(device)
                # calculate the negative log-likelihood of the model with a standard normal prior
                z, log_jac_det = self.flow(x_batch.float())
                
                loss = 0.5*torch.sum(z**2, 1) - log_jac_det
                #divide by amount dimensions
                loss = loss.mean() / x_batch.shape[1] 
                # backpropagate and update the weights
                loss.backward()
                self.optimizer.step()
                self.losses[-1] +=float(loss.cpu().detach().numpy())
            if self.scheduler:         
                self.scheduler.step()
            self.losses[-1]=(self.losses[-1]/(id_batch+1))
            if step%100==99:
                print('step: {}, loss: {}, '.format(step, self.losses[-1]))
                self.eval(step)
            if self.losses[-1] > self.min_loss:
                trigger_times += 1
                if trigger_times >= patience:
                    break    
            else:
                
                trigger_times = 0
                self.min_loss = self.losses[-1]
            
    
    def eval(self,step,N=1000):
        #Generates a sample and calculates wasserstein metrics from truth to gen
        #The tensors are reshaped in a wierd way because jetnet library wants this
        if len(self.metrics.keys())>0:
            self.gen=self.flow(torch.randn([N,90]).to(self.device),rev=True)[0].cpu().detach().numpy()

            self.true=torch.tensor(self.data)[:np.random.choice(N)]
            self.metrics["loss"].append(self.losses[-1])
            self.metrics["step"].append(step)
            start=time.time()
            if "w1p" in self.metrics.keys():
                w1p=np.round(jetnet.evaluation.w1p(self.gen[:N].reshape((len(self.gen[:N]),30,3)),
                self.true[:N].reshape((len(self.true[:N]),30,3)),num_batches=20),5)[0]
                self.metrics["w1p"].append(w1p)                
                print("w1p: {} s".format(time.time()-start))
            start=time.time()
            if "w1m" in self.metrics.keys():
                w1m=np.round(jetnet.evaluation.w1m(self.gen[:N].reshape((len(self.gen[:N]),30,3)),
                self.true[:N].reshape((len(self.true[:N]),30,3)),num_batches=20),5)[0]
                self.metrics["w1m"].append(w1m)
                print("w1m: {} s".format(time.time()-start))
            start=time.time()
            if "w1efp" in self.metrics.keys():
                w1efp=np.round(jetnet.evaluation.w1efp(self.gen[:N].reshape((len(self.gen[:N]),30,3)),
                self.true[:N].reshape((len(self.true[:N]),30,3)),num_batches=20),5)[0]
                self.metrics["w1efp"].append(w1efp)
                print("w1efp: {} s".format(time.time()-start))
            if self.hyperopt:
                tune.report(w1p=w1p,w1m=w1m,w1efp=w1efp,loss=self.losses[-1])


    def plot(self):
        path=tune.get_trial_dir()+"/plots"
        os.mkdir(path)
        for v,name in zip(["eta","pt","m"],[r"$\eta^{rel}$",r"$p_T^{rel}$",r"$m_T^{rel}$"]):
            v_sum=vector.array({"pt":np.zeros(len(self.gen)),"phi":np.zeros(len(self.gen)),"eta":np.zeros(len(self.gen)),"M":np.zeros(len(self.gen))})
            v2_sum=vector.array({"pt":np.zeros(len(self.gen)),"phi":np.zeros(len(self.gen)),"eta":np.zeros(len(self.gen)),"M":np.zeros(len(self.gen))})

            if v=="pt":
                a=0
                b=0.1
            if v=="eta":
                a=-0.4
                b=0.4
            if  v=="m":
                a=0
                b=0.25
            
            h=hist.Hist(hist.axis.Regular(100,a,b))
            h2=hist.Hist(hist.axis.Regular(100,a,b))
        
            if v=="m":
                m=np.zeros_like(self.gen[:,0])
                m_t=np.zeros_like(self.gen[:,0])
                for i in range(30):
                    m+=(np.cos(self.gen[:,3*i])*self.gen[:,3*i+2])**2+(np.sin(self.gen[:,3*i])*self.gen[:,3*i+2])**2
                    m_t+=(np.cos(self.true[:,3*i])*self.true[:,3*i+2])**2+(np.sin(self.true[:,3*i])*self.true[:,3*i+2])**2
                h.fill(m)
                h2.fill(m_t)
            if v=="eta": 

                h.fill(torch.tensor(self.gen).reshape(len(self.gen)*30,3).numpy()[:,1])
                h2.fill(torch.tensor(self.true).reshape(len(self.gen)*30,3).numpy()[:,1])
            if v=="pt":    
                
                h.fill(torch.tensor(self.gen).reshape(len(self.gen)*30,3).numpy()[:,2])
                h2.fill(torch.tensor(self.true).reshape(len(self.true)*30,3).numpy()[:,2])
                
        

            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            main_ax_artists, sublot_ax_arists = h.plot_ratio(
                h2,
                ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                rp_ylabel=r"Ratio",
                rp_num_label="Generated",
                rp_denom_label="Data",
                rp_uncert_draw_type="line",  # line or bar
            )
            ax[0].set_xlabel("")
            ax[1].set_ylim(0.25,2)
            ax[0].set_xlim(a,b)
            ax[1].set_xlim(a,b)
            plt.xlabel(name)
            plt.savefig("{}/{}.png".format(path,name))
            plt.show()
    
    # plt.ylim(0.5,2)
    # plt.tight_layout()

    def grad_penalty(real_data):
        batch_size = real_data.size(0)
        fake_data,_ = inn(torch.randn(batch_size,90).to(device))
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 90).to(device)
        # eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
 
        # get logits for interpolated images
        interp_logits,_ = inn(interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(  batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
    
    
        

    
    # cov,mmd=np.round(jetnet.evaluation.cov_mmd(gen[:N].reshape((len(gen[:N]),30,3)),
    #                  true[:N].reshape((len(true[:N]),30,3)),num_batches=20),5)

     #                 #    ,"mmd":mmds,"cov":covs                   
    #                   

# # bla
    #,"w1m","w1p","w1efp","cov","mmd"
  
if __name__=='__main__':
    start=time.time()
    best_config = {
        "network_layers": 5,
        "network_nodes":374,
        "lr": 0.004437,
        "activation":"lrelu",
        "permute_soft":True,
        "coupling_layers":14,
         "batch_size":4000,
         "lrdecay":0.99,
         "wdecay":0.01,
         "opt":"adamw",
         "lambda":0,
         "affine_clamping":2.0,
         "max_steps":1000
        }
    bayes_config={
            "network_layers": tune.randint(2,8),
            "network_nodes":tune.randint(200,600),
            "lrdecay":0.99,
            "wdecay":0.01,
            "opt":tune.choice(["adam","adamw"]),
            "permute_soft":False,
            "batch_size":6000,
            "lr": tune.uniform(0.0001,0.001),# tune.sample_from(lambda _: 1**(int(-np.random.randint(1, 4))),
            "activation": tune.uniform(0,3),
            "coupling_layers":tune.randint(6,80),
            "max_steps":100            
            }
    

    resources={"cpu":8 , "gpu": 0.5}

    num_samples=1000
    use_scheduler=False
    data_dir="/home/kaechben/JetNet_NF/train_{}_jets.csv".format(sys.argv[1])
    configs=[]
    configs=['/home/kaechben/ray_results/q/train_544ae_00013_13_activation=2.5991,coupling_layers=75,lr=0.00075166,network_layers=6,network_nodes=554_2022-01-23_19-50-31',
       '/home/kaechben/ray_results/q/train_544ae_00336_336_activation=2.0892,coupling_layers=43,lr=0.00097805,network_layers=6,network_nodes=593_2022-01-24_05-57-52',
       '/home/kaechben/ray_results/q/train_544ae_00606_606_activation=2.8892,coupling_layers=69,lr=0.00091325,network_layers=4,network_nodes=582_2022-01-24_13-58-13',
       '/home/kaechben/ray_results/q/train_544ae_00450_450_activation=2.6224,coupling_layers=57,lr=0.00098424,network_layers=6,network_nodes=459_2022-01-24_09-15-59',
       '/home/kaechben/ray_results/q/train_544ae_00116_116_activation=2.1155,coupling_layers=53,lr=0.00083447,network_layers=5,network_nodes=594_2022-01-23_22-57-06',
       '/home/kaechben/ray_results/q/train_544ae_00470_470_activation=2.4121,coupling_layers=50,lr=0.00091957,network_layers=6,network_nodes=476_2022-01-24_10-01-48',
       '/home/kaechben/ray_results/q/train_544ae_00690_690_activation=2.3466,coupling_layers=51,lr=0.00098684,network_layers=7,network_nodes=454_2022-01-24_16-38-22',
       '/home/kaechben/ray_results/q/train_544ae_00307_307_activation=2.6789,coupling_layers=68,lr=0.00074161,network_layers=7,network_nodes=434_2022-01-24_04-52-06',
       '/home/kaechben/ray_results/q/train_544ae_00424_424_activation=2.5037,coupling_layers=44,lr=0.00073179,network_layers=7,network_nodes=556_2022-01-24_08-34-09',
       '/home/kaechben/ray_results/q/train_544ae_00278_278_activation=2.485,coupling_layers=43,lr=0.0009597,network_layers=7,network_nodes=442_2022-01-24_03-53-49']
    
    hyperopt=""
    if hyperopt!="":
        init("auto",_redis_password='5241590000000000')
    # Create HyperBand scheduler 
    scheduler = HyperBandScheduler(metric="loss", mode="min")

    limit=100

    data=pd.read_csv(data_dir,sep=" ",header=None)
    jets=[]
    for njets in range(30,31):
        masks=np.sum(data.values[:,np.arange(3,120,4)],axis=1)
        df=data.loc[masks==njets,:]
        df=df.drop(np.arange(3,120,4),axis=1)
        df=df.iloc[:,:3*njets]
    
        
        #print("the subsample with {} particles in a jet has {} entries".format(njets,df.shape[0]))
        if len(df)>0:
                jets.append(df.values)
        ###Standard Scaling
    for i in [-1]:
        scaler=StandardScaler().fit(jets[i])
        jets[i]=scaler.transform(jets[i][:limit])
    for config in configs:
                with open("{}/params.json".format(config)) as json_file:

                    config=json.load(json_file)
                config["max_steps"]=1000
                NF(jets[-1],config)
    

    if False:
        if hyperopt=="random":
            config=bayes_config
        
            result = tune.run(
                tune.with_parameters(train,jets=jets,scaler=scaler),   
                resources_per_trial=resources,
                config=config,
                num_samples=num_samples,
                scheduler=scheduler if use_scheduler else None,
                name=sys.argv[1]+"test"
            

            )
        elif hyperopt=="dragonfly":
            df_search = DragonflySearch(
            optimizer="bandit",
            domain="euclidean",
            metric="loss",
            mode="min"
            # space=space,  # If you want to set the space manually
        )
            df_search = ConcurrencyLimiter(df_search, max_concurrent=5)
            analysis = tune.run(
            tune.with_parameters(train,jets=jets,scalar=scaler),
            # metric="loss",
            # mode="min",
            #name="second_dragonfly_search",
            search_alg=df_search,
            scheduler=scheduler,
            num_samples= num_samples,
            config=bayes_config,
            resources_per_trial=resources
        )
            print("Best hyperparameters found were: ", analysis.best_config)
        ##############################################
        #     Best Run                               #
        ##############################################
        else:
          pass   
        if hyperopt!="":
            print(result)
            best_trial = result.get_best_trial("w1p", "min", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final loss: {}".format(
                best_trial.last_result["loss"]))
            print("Best trial final w1: {}".format(
                best_trial.last_result["w1p"]))

            
    print("finished after {} s".format(time.time()-start))
#cur_reserved = torch.cuda.memory_reserved(device)
#cur_alloc = torch.cuda.memory_allocated(device) 
#https://pypi.org/project/nvidia-htop/