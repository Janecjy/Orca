'''
MIT License
Copyright (c) Soheil Abbasloo - Chen-Yu Yen 2020

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import torch
import gym
import math
import collections
from utils import logger
import os
import sysv_ipc
import signal
import sys
from time import sleep
from models import create_mask

class Env_Wrapper(object):
    def __init__(self, name):

        self.env = gym.make(name)
        self.env.reset()

    def get_dims_info(self):

        return self.env.observation_space.shape[0], self.env.action_space.shape[0]

    def get_action_info(self):
        action_scale = self.env.action_space.high
        action_range = (self.env.action_space.low, self.env.action_space.high)
        return  action_scale, action_range

    def reset(self):
        return self.env.reset()

    def step(self, action,eval_=False):
        s1, r, done, _ = self.env.step(action)

        if done:
            s1 = self.env.reset()

        return s1, r, done, _



class GYM_Env_Wrapper(Env_Wrapper):
    def __init__(self,name, params, for_init_only=True, shrmemory_id=None, shrmem_r=None, shrmem_w=None):
        super().__init__(name)
        logger.info("USE_GYM_Env_Wrapper")
        if not for_init_only:
            self.shrmemory_id = shrmemory_id
            self.shrmem_r = shrmem_r
            self.shrmem_w = shrmem_w
            self.prev_rid = 99

    def test(self):
        print("Hello")

    def get_state(self):

        prev_rid = 0
        s1 = None
        delay_ = 0
        rew0 = 10
        error_code = True
        return prev_rid, s1, delay_, rew0, error_code

    def write_action(self, action):
        pass

    def map_action(self, action):
        out = math.pow(4, action)
        out *= 100
        out = int(out)
        return out

    def step(self, action, eval_=False):

        s1, r, done, _ = self.env.step(action)

        if done:
            s1 = self.env.reset()

        return s1, r, done, True


class TCP_Env_Wrapper(object):
    def __init__(self,name, params, config=None, for_init_only=True, shrmem_r=None, shrmem_w=None,use_normalizer=True):

        self.params = params
        if not for_init_only:
            self.params = params
            self.shrmem_r = shrmem_r
            self.shrmem_w = shrmem_w
            self.prev_rid = 99999
            self.wid = 23
            self.local_counter=0
            self.pre_samples = 0.0
            self.new_samples = 0.0
            self.avg_delay = 0.0
            self.avg_thr = 0.0
            self.thr_ = 0.0
            self.del_ = 0.0
            self.max_bw = 0.0
            self.max_cwnd = 0.0
            self.max_smp = 0.0
            self.min_del = 9999999.0
            # -------------- NEW FIELDS for Palantir --------------
            self.raw_feature_buffer = []    # Will store (raw_6_features, delta_t)
            self.accumulated_time = 0.0     # Track time since last token
            self.token_window = []          # Rolling window of up to 10 tokens
            self.max_token_window_size = 10
            # This will hold the embedding you append to the RL state.
            # Initialize to zeros so if we haven't got 10 tokens yet, we use a zero-vector
            self.embedding_size = 16
            self.current_transformer_embedding = np.zeros(self.embedding_size, dtype=np.float32)

            # If you already know global min/max for base_rtt from offline analysis,
            # set them here; or you can track them online if you prefer:
            self.rtt_min = 0.1
            self.rtt_max = 1.6
            self.base_rtt = 0.0

            # A reference to your trained Transformer model
            self.DEVICE = 'cpu'
            self.transformer_model = torch.load('models/RTT-Checkpoint-BaseTransformer3_64_5_5_16_4_lr_1e-05_vocab-809iter.p', map_location=self.DEVICE)
            self.bucket_boundaries_ccbench = {
                1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.63, 2.12, 4.02, 8, 12],
                2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5],
                3: [0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
                4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
                5: [0.75, 1, 1.001, 1.003, 1.012, 1.25]
            }

            self.use_normalizer=use_normalizer
            if self.use_normalizer==True:
                self.normalizer=Normalizer(params, config)
            else:
                self.normalizer=None
            self.del_moving_win = Moving_Win(params.dict['MVWIN'])
            self.thr_moving_win = Moving_Win(params.dict['MVWIN'])
            self.config = config
            signal.signal(signal.SIGINT, self.handler_term)
            signal.signal(signal.SIGTERM, self.handler_term)
            if self.config.load is not None and self.use_normalizer==True:
                _ = self.normalizer.load_stats()

    def handler_term(self, signum, frame):
        print("python program terminated usking Kill -15")
        if not self.config.eval and self.use_normalizer==True:
            print("save stats by kill -15")
            self.normalizer.save_stats()
        sys.exit(0)

    def get_dims_info(self):
        return self.params.dict['state_dim'], self.params.dict['action_dim']
    
    def orca_get_dims_info(self):
        return self.params.dict['orca_state_dim'], self.params.dict['action_dim']

    def get_action_info(self):
        action_scale = np.array([1.])
        action_range = (-action_scale,action_scale)
        print('action_scale & action_range')
        return  action_scale, action_range

    def reset(self, agent2, orca_s0_rec_buffer):
        # start signal
        self.shrmem_w.write(str(99999) + " " + str(99999) + "\0")
        orca_state, state, delay_, rew0, error_code  = self.get_state(agent2=agent2, orca_s0_rec_buffer=orca_s0_rec_buffer)
        return orca_state, state

    def test(self):
        print("Hello")

    def compute_token(self):
        """
        raw_feature_buffer: list of [(f6, dt), ...] 
                            where f6 is np.array of shape (6,) 
                            and dt is float
        Returns: a single token of shape (6,)
        """
        # Sum up (feature * dt) for each entry
        total_time = 0.0
        weighted_sum = np.zeros(6, dtype=np.float64)
        # logger.info(f"raw_feature_buffer: {self.raw_feature_buffer}")
        for (f6, dt) in self.raw_feature_buffer:
            weighted_sum += f6 * dt
            total_time += dt

        if total_time < 1e-9:
            # Avoid div-by-zero; fallback to the last sample or zeros
            avg_features = self.raw_feature_buffer[-1][0]
        else:
            avg_features = weighted_sum / total_time

        # logger.info(f"avg_features: {avg_features}")

        # avg_features[0] is base_rtt. We min-max normalize it:
        base_rtt_val = avg_features[0]*2/100
        if np.isclose(self.rtt_min, self.rtt_max):
            normalized_rtt = 0.0
        else:
            normalized_rtt = (base_rtt_val - self.rtt_min) / (self.rtt_max - self.rtt_min)
        
        # For features 1..5, do bucketization
        # feature 1 = avg_features[1], feature 2 = avg_features[2], ...
        # boundaries are in your self.bucket_boundaries_ccbench dictionary
        token = [normalized_rtt]  # first slot is normalized base_rtt
        for feat_idx in range(1, 6):
            boundaries = self.bucket_boundaries_ccbench[feat_idx]
            val = avg_features[feat_idx]
            # local bin index
            bin_local = np.searchsorted(boundaries, val, side='right')
            # We often want to offset these bins in a big vocabulary,
            # but for a single token we can just store bin_local directly
            token.append(bin_local)
        
        return np.array(token, dtype=np.float32)

    def update_transformer_embedding(self):
        # Convert self.token_window -> shape (1, 10, 6)
        tokens_np = np.stack(self.token_window, axis=0)  # shape [10, 6]
        tokens_tensor = torch.from_numpy(tokens_np).unsqueeze(0).to(self.DEVICE)  # [1, 10, 6]
        # logger.info("tokens_tensor: "+str(tokens_tensor))

        self.transformer_model.eval()
        with torch.no_grad():
            # Example forward pass. Because we have a Seq2Seq, we need a dummy trg.
            # We'll do something minimal. Real usage might differ.

            enc_input = tokens_tensor[:, :, :].to(self.DEVICE)
            dec_input = (1.5 * torch.ones((tokens_tensor.shape[0], 10, tokens_tensor.shape[2]))).to(self.DEVICE)
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=2, device=self.DEVICE)
            
            # We pass None for padding masks:
            encoder_out = self.transformer_model(
                enc_input, dec_input, 
                src_mask=src_mask, 
                tgt_mask=tgt_mask,
                src_padding_mask=None, 
                tgt_padding_mask=None, 
                memory_key_padding_mask=None
            )

            # logger.info("encoder_out: "+str(encoder_out))
            # shape: [1, 10, 64]

            # You can pick how to compress 10 steps into 1 vector:
            # e.g. average pooling across time
            # [1, 32]
            embedding_tensor = encoder_out.mean(dim=1)
            # logger.info("embedding_tensor: "+str(embedding_tensor))

            # move to CPU and NumPy
            emb_cpu = embedding_tensor.squeeze(0).cpu().numpy()  # shape [32]
            self.current_transformer_embedding = emb_cpu.astype(np.float32)


    def get_state(self, agent2=None, orca_s0_rec_buffer=None, evaluation=False):
        succeed = False
        error_cnt=0
        while(1):
        # Read value from shared memory
            try:
                memory_value = self.shrmem_r.read()

            except sysv_ipc.ExistentialError:
                print("No shared memory Now, python ends gracefully :)")
                logger.info("No shared memory Now, python ends gracefully :)")
                sys.exit(0)

            memory_value = memory_value.decode('unicode_escape')

            i = memory_value.find('\0')

            if i != -1:

                memory_value = memory_value[:i]
                readstate = np.fromstring(memory_value, dtype=float, sep=' ')
                try:
                    rid = readstate[0]
                except :
                    rid = self.prev_rid
                    sleep(0.01)
                    continue
                try:
                    s0 = readstate[1:]
                except :
                    print("s0 waring")
                    sleep(0.01)
                    continue


                if rid != self.prev_rid:
                    succeed = True
                    break
                else:
                    wwwwww=""

            error_cnt=error_cnt+1
            if error_cnt > 24000:
                error_cnt=0
                print("After 3 min, We didn't get any state from the server. Actor "+str(self.config.task)+" is going down down down ...\n")
                sys.exit(0)

            sleep(0.01)

        error_cnt=0
        if succeed == False:
            raise ValueError('read Nothing new from shrmem for a long time')
        reward=0
        state=np.zeros(1)
        w=s0
        # logger.info("s0: "+str(s0))
        # logger.info("s0 length: "+str(len(s0)))
        if len(s0) == (self.params.dict['input_dim']):
            d=s0[0]
            thr=s0[1]
            samples=s0[2]
            delta_t=s0[3]
            # logger.info("delta_t: "+str(delta_t))
            target_=s0[4]
            cwnd=s0[5]
            pacing_rate=s0[6]
            loss_rate=s0[7]
            srtt_ms=s0[8]
            snd_ssthresh=s0[9]
            packets_out=s0[10]
            retrans_out=s0[11]
            max_packets_out=s0[12]
            mss=s0[13]
            min_rtt=s0[14]

            self.local_counter+=1

            if self.use_normalizer==True:
                if evaluation!=True:
                    self.normalizer.observe(s0)
                s0 = self.normalizer.normalize(s0)
                min_ = self.normalizer.stats()
            else:
                min_ = s0-s0

            d_n=s0[0]-min_[0]
            thr_n=s0[1]
            thr_n_min=s0[1]-min_[1]
            samples_n=s0[2]
            samples_n_min=s0[2]-min_[2]
            delta_t_n=s0[3]
            delta_t_n_min=s0[3]-min_[3]

            cwnd_n_min=s0[5]-min_[5]
            pacing_rate_n_min=s0[6]-min_[6]
            loss_rate_n_min=s0[7]-min_[7]
            srtt_ms_min=s0[8]-min_[8]
            snd_ssthresh_min=s0[9]-min_[9]
            packets_out_min=s0[10]-min_[10]
            retrans_out_min=s0[11]-min_[11]
            max_packets_out_min=s0[12]-min_[12]
            mss_min=mss-min_[13]
            min_rtt_min=min_rtt-min_[14]

            if self.use_normalizer==False:
                thr_n=thr_n
                thr_n_min=thr_n_min
                samples_n_min=samples_n_min
                cwnd_n_min=cwnd_n_min
                loss_rate_n_min=loss_rate_n_min
                d_n=d_n
            if self.max_bw<thr_n_min:
                self.max_bw=thr_n_min
            if self.max_cwnd<cwnd_n_min:
                self.max_cwnd=cwnd_n_min
            if self.max_smp<samples_n_min:
                self.max_smp=samples_n_min
            if self.min_del>d_n:
                self.min_del=d_n

            ################# Transfer all of the vars. to Rate/Max(Rate) space
            #cwnd_bytes= cwnd_n_min*mss_min
            #cwnd_n_min=(cwnd_bytes*1000)/srtt_ms_min
            #snd_ssthresh_min=(snd_ssthresh_min*mss_min*1000)/srtt_ms_min
            #packets_out_min=(packets_out_min*mss_min*1000)/srtt_ms_min
            #retrans_out_min=(retrans_out_min*mss_min*1000)/srtt_ms_min
            #max_packets_out_min=(max_packets_out_min*mss_min*1000)/srtt_ms_min
            #inflight_bytes=(packets_out-samples)*mss_min*1000

            if min_rtt_min*(self.params.dict['delay_margin_coef'])<srtt_ms_min:
                delay_metric=(min_rtt_min*(self.params.dict['delay_margin_coef']))/srtt_ms_min
            else:
                delay_metric=1

            reward  = (thr_n_min-5*loss_rate_n_min)/self.max_bw*delay_metric

            if self.max_bw!=0:
                state[0]=thr_n_min/self.max_bw
                tmp=pacing_rate_n_min/self.max_bw
                if tmp>10:
                    tmp=10
                state=np.append(state,[tmp])
                state=np.append(state,[5*loss_rate_n_min/self.max_bw])
            else:
                state[0]=0
                state=np.append(state,[0])
                state=np.append(state,[0])
            state=np.append(state,[samples/cwnd])
            state=np.append(state,[delta_t_n])
            state=np.append(state,[min_rtt_min/srtt_ms_min])
            state=np.append(state,[delay_metric])
            orca_state = state
            # -------------------------------------------------------
            # 1) We read s0 from shared memory
            # 2) Once we parse s0, we add to raw_feature_buffer and maybe create a new token
            #    if we've reached base_rtt)
            raw_6_features = s0[-6:]
            self.raw_feature_buffer.append((raw_6_features, delta_t))
            self.accumulated_time += delta_t*10
            
            new_token_created = False
            if not self.base_rtt:
                self.base_rtt = raw_6_features[0]*2/100  # set the base_rtt once in 100x ms

            # logger.info("accumulated_time: "+str(self.accumulated_time))
            # logger.info("base_rtt: "+str(self.base_rtt))
            if self.accumulated_time >= self.base_rtt > 0:
                # We have enough data for 1 token
                token = self.compute_token()
                # logger.info("New token: "+str(token))
                self.token_window.append(token)
                if len(self.token_window) > self.max_token_window_size:
                    self.token_window.pop(0)
                # we made a new token, so set flag
                new_token_created = True

                # reset accumulation
                self.raw_feature_buffer.clear()
                self.accumulated_time = 0.0

            # 3) Only if a new token was created AND we have the full 10 tokens
            #    do we run the transformer to get a new embedding.
            if new_token_created and len(self.token_window) == self.max_token_window_size:
                self.update_transformer_embedding()

            # -------------------------------------------------------
            # Then finally, we append `transformer_embedding` to the state.

            # state is currently something like shape (N,).
            # We'll do:

            # logger.info(f"transformer_embedding: {self.current_transformer_embedding}")
            # print("-----------------------")
            # print(state)
            # print("buffer~!!!!!!")
            # if orca_s0_rec_buffer is not None:
            #     print(len(state))
            #     print(len(orca_s0_rec_buffer))
            # print(s0_rec_buffer)
            # print(agent2)
            if agent2 is not None:
                hidden_out = agent2.get_action_hidden(orca_s0_rec_buffer)
                # print("hidden_out:")
            # Convert hidden_out from list-of-array to a true NumPy array
            hidden_out = np.array(hidden_out)  # shape might be (1,1,1)

            # Flatten it down to 1D
            hidden_out = hidden_out.ravel()    # now shape is (1,) if there's only one value
            # print(len(hidden_out))
            # print(hidden_out)
            # print(len(self.current_transformer_embedding))
            # print(self.current_transformer_embedding)
            state = np.concatenate([hidden_out, self.current_transformer_embedding], axis=0)

            self.prev_rid = rid
            return orca_state, state, d, reward, True
        else:
            return orca_state, state, 0.0, reward, False

    def map_action(self, action):
        out = math.pow(4, action)
        out *= 100
        out = int(out)
        return out

    def map_action_reverse(self,a):
        out =  math.log(a/100,4)
        return out


    def write_action(self, action):

        modified_action = self.map_action(action)

        msg = str(self.wid)+" "+str(modified_action)+"\0"
        self.shrmem_w.write(msg)
        self.wid = (self.wid + 1) % 1000
        pass

    def step(self, action, agent2=None, orca_s0_rec_buffer=None, eval_=False):
        orca_s1, s1, delay_, rew0, error_code  = self.get_state(agent2=agent2, orca_s0_rec_buffer=orca_s0_rec_buffer, evaluation=eval_)

        return orca_s1, s1, rew0, False, error_code


class Moving_Win():
    def __init__(self,win_size):
        self.queue_main = collections.deque(maxlen=win_size)
        self.queue_aux = collections.deque(maxlen=win_size)
        self.length = 0
        self.avg = 0.0
        self.size = win_size
        self.total_samples=0

    def push(self,sample_value,sample_num):
        if self.length<self.size:
            self.queue_main.append(sample_value)
            self.queue_aux.append(sample_num)
            self.length=self.length+1
            self.avg=(self.avg*self.total_samples+sample_value*sample_num)
            self.total_samples+=sample_num
            if self.total_samples>0:
                self.avg=self.avg/self.total_samples
            else:
                self.avg=0.0
        else:
            pop_value=self.queue_main.popleft()
            pop_num=self.queue_aux.popleft()
            self.queue_main.append(sample_value)
            self.queue_aux.append(sample_num)
            self.avg=(self.avg*self.total_samples+sample_value*sample_num-pop_value*pop_num)
            self.total_samples=self.total_samples+(sample_num-pop_num)
            if self.total_samples>0:
                self.avg=self.avg/self.total_samples
            else:
                self.avg=0.0

    def get_avg(self):
        return self.avg

    def get_length(self):
        return self.length

class Normalizer():
    def __init__(self, params, config):
        self.params = params
        self.config = config
        self.n = 1e-5
        num_inputs = self.params.dict['input_dim']
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)
        self.dim = num_inputs
        self.min = np.zeros(num_inputs)


    def observe(self, x):
        self.n += 1
        last_mean = np.copy(self.mean)
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = self.mean_diff/self.n

    def normalize(self, inputs):
        obs_std = np.sqrt(self.var)
        a=np.zeros(self.dim)
        if self.n > 2:
            a=(inputs - self.mean)/obs_std
            for i in range(0,self.dim):
                if a[i] < self.min[i]:
                    self.min[i] = a[i]
            return a
        else:
            return np.zeros(self.dim)

    def normalize_delay(self,delay):
        obs_std = math.sqrt(self.var[0])
        if self.n > 2:
            return (delay - self.mean[0])/obs_std
        else:
            return 0

    def stats(self):
        return self.min

    def save_stats(self):
        dic={}
        dic['n']=self.n
        dic['mean'] = self.mean.tolist()
        dic['mean_diff'] = self.mean_diff.tolist()
        dic['var'] = self.var.tolist()
        dic['min'] = self.min.tolist()
        import json
        with open(os.path.join(self.params.dict['train_dir'], 'stats.json'), 'w') as fp:
                json.dump(dic, fp)

        print("--------save stats at{}--------".format(self.params.dict['train_dir']))
        logger.info("--------save stats at{}--------".format(self.params.dict['train_dir']))



    def load_stats(self, file='stats.json'):
        import json
        if os.path.isfile(os.path.join(self.params.dict['train_dir'], file)):
            print("Stats exist!, load", self.config.task)
            with open(os.path.join(self.params.dict['train_dir'], file), 'r') as fp:
                history_stats = json.load(fp)
                print(history_stats)
            self.n = history_stats['n']
            self.mean = np.asarray(history_stats['mean'])
            self.mean_diff = np.asarray(history_stats['mean_diff'])
            self.var = np.asarray(history_stats['var'])
            self.min = np.asarray(history_stats['min'])
            return True
        else:
            print("stats file is missing when loading")
            return False

