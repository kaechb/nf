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
import traceback
plt.style.use(hep.style.ROOT)
#from functools import partial
device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class NF():
    def __init__(self,data=None,config={},metrics={"w1p":[],"w1m":[],"w1efp":[],"step":[],"loss":[]},hyperopt=False):
        '''
        :param data: the data to be used for training
        :param config: dictionary of parameters
        :param metrics: a dictionary of lists, where each list is a list of metrics for each epoch
        :param hyperopt: If True, the model will be trained using ray. If False, the model will be
        trained using the config parameters, defaults to False (optional)
        '''
        if hyperopt==False:
            self.config=config  
        self.data=data
        self.metrics=metrics
        self.hyperopt=hyperopt  
        
        

    def subnet(self,dims_in, dims_out):
        # The code below is a class that inherits from the class SequenceINN. 
        # It has a constructor that takes the number of input channels as an argument. 
        # It then creates an instance of the class AllInOneBlock, which is a class that inherits from
        # the class Fm.Sequential. 
        # The constructor of the class AllInOneBlock takes the subnet_constructor as an argument. 
        # The subnet_constructor is a function that returns a subnet. 
        # The network is a sequence of linear layers, with an activation function between each layer. 
        # The activation function is either ReLU, LeakyReLU, or Tanh. 
        # The last layer is initialized with a weight matrix of zeros. -> output of trafo is ID
        # The network is initialized with Xavier initialization. 
        # The network is a sequence of layers, with an activation function between each layer. 
        
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
            print(self.config["activation"])
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
        print(self.config["coupling_layers"])
        for k in range(int(self.config["coupling_layers"])):       
                inn.append(Fm.AllInOneBlock,  subnet_constructor=self.subnet,permute_soft=self.config["permute_soft"])
        return inn
    
    def set_training(self):  
       
        if  self.config["opt"]=="adam":
            self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.config["lr"])
        else:#elif  self.config["opt"]=="adam":
            self.optimizer = torch.optim.AdamW(self.flow.parameters(), lr=self.config["lr"],weight_decay=self.config["wdecay"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.config["lrdecay"])

    def train(self,config,scaler,data=None,scheduler=None,patience=50):
        '''
        :param config: dictionary of parameters
        :param scaler: scaler object
        :param data: the data to be used for training
        :param scheduler: 
        :param patience: how many times the loss has to be the same before stopping, defaults to 50
        (optional)
        
        '''
        
        #print(markdown(pd.DataFrame(self.config,index=[1])))
        self.config=config

        n_eval=max(int(config["max_steps"]/10),10)
        print("n_eval: {}".format(n_eval))
        self.data=data
        self.scaler=scaler
        self.flow=self.build_model()        
        self.set_training()
        
        #trains and evaluates the model

        torch.manual_seed(0)
        self.dataloader = DataLoader(self.data,self.config["batch_size"],shuffle=True)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.min_loss=np.inf
        self.losses=[]
        self.flow.to(device)
        times=[]
        for step in range(int(self.config["max_steps"])): 
            start=time.time()
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
                times.append(time.time()-start)

            if self.scheduler:         
                self.scheduler.step()
            self.losses[-1]=(self.losses[-1]/(id_batch+1))
            if np.isnan(self.losses[-1]):
                if self.hyperopt:
                    tune.report(w1p=-1,loss=-1,w1efp=-1,w1m=-1)
                else:
                    print("nan loss end")
                with open(tune.get_trial_dir()+'/metrics.json', 'w') as fp:
                    json.dump(self.metrics, fp)
                return 0
            if step%n_eval==n_eval-1:
                print('step: {}, loss: {:.4f}, '.format(step, self.losses[-1]))
                print(r"Average time per step: {:.2f} \pm {:.2f}".format( np.mean(np.array(times)),np.std(np.array(times))))
                with torch.no_grad():
                    self.eval(step)
            if self.losses[-1] > self.min_loss:
                trigger_times += 1
                print("loss not decreasing values {:.2f}, strike {}".format(self.losses[-1],trigger_times))
                if trigger_times >= patience:
                    with torch.no_grad():
                        self.eval(step)
                    with open(tune.get_trial_dir()+'/metrics.json', 'w') as fp:
                         json.dump(self.metrics, fp) 
                    self.plot_vars() 
                    self.plot()
                    return 0
            else:
                trigger_times = 0
                self.min_loss = self.losses[-1]
        with torch.no_grad():
            self.eval(step,N=10000)
        if self.hyperopt:
            with open(tune.get_trial_dir()+'/metrics.json', 'w') as fp:
                        json.dump(self.metrics, fp)
        self.plot_vars()
        self.plot()
        
    
    def eval(self,step,N=10000):
        '''
        Generates a sample and calculates wasserstein metrics from truth to gen.
        
        :param step: the current step of training
        :param N: Number of samples to generate, defaults to 10000 (optional)
        '''
        #Generates a sample and calculates wasserstein metrics from truth to gen
        #The tensors are reshaped in a wierd way because jetnet library wants this
        if N<0 or N>len(self.data):
            N=len(self.data)
        if len(self.metrics.keys())>0:
            self.gen=self.flow(torch.randn([N,90]).to(self.device),rev=True)[0].cpu().detach().numpy()
            self.true=torch.tensor(self.data)[np.random.choice(len(self.data),N,replace=False)].numpy()
            self.metrics["loss"].append(self.losses[-1])
            self.metrics["step"].append(step)
            start=time.time()
            l=self.losses[-1]
            if "w1p" in self.metrics.keys():
                w1p=np.round(jetnet.evaluation.w1p(scaler.inverse_transform(self.gen).reshape((len(self.gen),30,3)),
                scaler.inverse_transform(self.true).reshape((len(self.true),30,3)),num_batches=5),5)[0]
                self.metrics["w1p"].append(w1p)                
                print("w1p: {} s".format(time.time()-start))
            start=time.time()
            if "w1m" in self.metrics.keys():
                w1m=np.round(jetnet.evaluation.w1m(scaler.inverse_transform(self.gen).reshape((len(self.gen),30,3)),
                scaler.inverse_transform(self.true).reshape((len(self.true),30,3)),num_batches=5),5)[0]
                self.metrics["w1m"].append(w1m)
                print("w1m: {} s".format(time.time()-start))
            start=time.time()
            if "w1efp" in self.metrics.keys():
                w1efp=np.round(jetnet.evaluation.w1efp(scaler.inverse_transform(self.gen).reshape((len(self.gen),30,3)),
                scaler.inverse_transform(self.true).reshape((len(self.true),30,3)),num_batches=5),5)[0]
                self.metrics["w1efp"].append(w1efp)
                print("w1efp: {} s".format(time.time()-start))
            if self.hyperopt:
                tune.report(loss=l,w1p=w1p,w1m=w1m,w1efp=w1efp)
        print(self.metrics)

    def plot_vars(self):
        '''
        Plot the generated and true distributions of the variables.
        '''
        if self.hyperopt:
            path=tune.get_trial_dir()+"/plots"
            os.makedirs(path,exist_ok=True)
        else:
            path="../debug/"+self.config["name"]
            os.makedirs(path,exist_ok=True)
        self.gen=self.scaler.inverse_transform(self.gen)
        self.true=self.scaler.inverse_transform(self.true)
        name,label=["eta","phi","pt"],[r"$\eta^{rel}$",r"$\phi$",r"$p_T^{rel}$"]
        for i in range(np.shape(self.gen)[1]):
            a=min(np.quantile(self.gen[:,i],0.01),np.quantile(self.true[:,i],0.01))
            b=max(np.quantile(self.gen[:,i],0.99),np.quantile(self.true[:,i],0.99))
            print("{}: a={}, b={}".format(name[i%3],a,b))
            h=hist.Hist(hist.axis.Regular(100,a,b))
            h2=hist.Hist(hist.axis.Regular(100,a,b))
            h.fill(self.gen[:,i])
            h2.fill(self.true[:,i])
            fig,ax=plt.subplots(2,1,gridspec_kw={'height_ratios': [3, 1]})
            hep.cms.label(data=False,lumi=None ,year=None,rlabel="",llabel="Private Work",ax=ax[0] )
            try:            
                main_ax_artists, sublot_ax_arists = h.plot_ratio(
                    h2,
                    ax_dict={"main_ax":ax[0],"ratio_ax":ax[1]},
                    rp_ylabel=r"Ratio",
                    rp_num_label="Generated",
                    rp_denom_label="Data",
                    rp_uncert_draw_type="line",  # line or bar
                )
            except:
                print("shape true: ",np.shape(self.true))
                print("shape gen: ",np.shape(self.gen))
                print("")
                traceback.print_exc()
            ax[0].set_xlabel("")
            ax[1].set_ylim(0.25,2)
            ax[0].set_xlim(a,b)
            ax[1].set_xlim(a,b)
            plt.xlabel(label[i%3])
            plt.savefig("{}/jet{}_{}.png".format(path,int(i/3)+1,name[i%3]))
            plt.close()

    def plot(self):
        '''
        Plots some summary plots
        '''
        if self.hyperopt:
            path=tune.get_trial_dir()+"/plots"
            os.makedirs(path,exist_ok=True)
        else:
            path="../debug/"+self.config["name"]
            os.makedirs(path,exist_ok=True)
        self.gen=torch.tensor(self.gen).reshape(len(self.gen)*30,3).numpy()
        self.true=torch.tensor(self.true).reshape(len(self.true)*30,3).numpy()
        m=(np.cos(self.gen[:,1])*self.gen[:,2])**2+(np.sin(self.gen[:,1])*self.gen[:,2])**2
        m_t=(np.cos(self.true[:,1])*self.true[:,2])**2+(np.sin(self.true[:,1])*self.true[:,2])**2
        i=0
        for v,name in zip(["eta","phi","pt","m"],[r"$\eta^{rel}$",r"\phi",r"$p_T^{rel}$",r"$m_T^{rel}$"]):
            
            if v!="m":
                a=min(np.quantile(self.gen[:,i],0.01),np.quantile(self.true[:,i],0.01))
                b=max(np.quantile(self.gen[:,i],0.99),np.quantile(self.true[:,i],0.99))     
                h=hist.Hist(hist.axis.Regular(100,a,b))
                h2=hist.Hist(hist.axis.Regular(100,a,b))
                h.fill(self.gen[:,i])
                h2.fill(self.true[:,i])
            else:
                a=min(np.quantile(m,0.01),np.quantile(m_t,0.01))
                b=max(np.quantile(m,0.99),np.quantile(m_t,0.99))
                h=hist.Hist(hist.axis.Regular(100,a,b))
                h2=hist.Hist(hist.axis.Regular(100,a,b))
                h.fill(m)
                h2.fill(m_t)
            print("{}: a={}, b={}".format(v,a,b))

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
            plt.savefig("{}/{}_{}.png".format(path,v,self.config["name"]))
            plt.close()
            i+=1

        
              
    

   

  
if __name__=='__main__':
    start=time.time()
    config = {
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
        "network_layers": tune.randint(2,7),
        "network_nodes":tune.randint(200,1000),
        "lrdecay":0.99,
        "wdecay":0.01,
        "opt":tune.choice(["adam","adamw"]),
        "permute_soft":tune.choice([False,True]),
        "batch_size":6000,
        "lr": tune.uniform(0.00005,0.005),# tune.sample_from(lambda _: 1**(int(-np.random.randint(1, 4))),
        "activation": tune.uniform(0,3),
        "coupling_layers":tune.randint(6,80),
        "max_steps":1000           
        }
    num_samples=1000
    limit=-1

    resources={"cpu":20 , "gpu": 0.5}
    use_scheduler=False
    process="t"
    data_dir=os.environ["HOME"]+"/JetNet_NF/train_{}_jets.csv".format(process)
    configs=[os.environ["HOME"]+'/JetNet_NF/best_results_final/q_best']  
    hyperopt="random"
    if hyperopt!="":
        init("auto")
    # Create HyperBand scheduler 
    scheduler = HyperBandScheduler(metric="loss", mode="min")
   
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
    
    
    
    plt.hist(torch.tensor(jets[-1]).reshape(len(jets[-1])*30,3)[:,2].numpy(),bins=100)
    plt.savefig(os.environ['HOME']+"/debug/{}_in.png".format(process))
    plt.close()
    if hyperopt=="random":
        from ray.tune import CLIReporter

    # Limit the number of rows.
        flow=NF(hyperopt=True,config=bayes_config)
        reporter = CLIReporter(max_progress_rows=40,max_report_frequency=30, sort_by_metric=True,
        metric="w1p",parameter_columns=["network_nodes","coupling_layers","permute_soft","network_nodes"])
        # Add a custom metric column, in addition to the default metrics.
        # Note that this must be a metric that is returned in your training results.
        reporter.add_metric_column("loss")
        reporter.add_metric_column("w1p")
        reporter.add_metric_column("w1efp")
        reporter.add_metric_column("w1m")
        bayes_config["name"]=process+"_finalfinalscan_"
        result = tune.run(tune.with_parameters(
            flow.train,data=jets[-1],scaler=scaler),   
            resources_per_trial=resources,
            config=bayes_config,
            num_samples=num_samples,
             progress_reporter=reporter,
            # scheduler=scheduler if use_scheduler else None,
            name=process+"_finalscan"
        )
    if hyperopt=="":
        for file_name in configs:
            with open("{}/params.json".format(file_name)) as json_file:
                config=json.load(json_file)
            config["max_steps"]=10
            config["name"]=process+"_debug"  
            print(config)
            flow=NF(hyperopt=False,config=config)
               #+file_name.split("/")[-1]
            flow.train(data=jets[-1],config=config,scaler=scaler)

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
        tune.with_parameters(flow.train(),jets=jets,scalar=scaler),
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
        print(best_trial)
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final w1: {}".format(
            best_trial.last_result["w1p"]))

        
    print("finished after {} s".format(time.time()-start))
#cur_reserved = torch.cuda.memory_reserved(device)
#cur_alloc = torch.cuda.memory_allocated(device) 
#https://pypi.org/project/nvidia-htop/