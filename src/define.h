//============================================================================
// Author      : Soheil Abbasloo
// Version     : 1.0
//============================================================================

/*
  MIT License
  Copyright (c) Soheil Abbasloo 2020 (ab.soheil@gmail.com)

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
*/
#include "flow.h"
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>        // needed for socket(), uint8_t, uint16_t, uint32_t
pthread_mutex_t lockit;
#include <unistd.h>
#include <math.h>
#include <time.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

//Shared Memory ==> Communication with RL-Module -----*
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
//-------------------------*
#define     OK_SIGNAL 99999
#define     TARGET_MARGIN   100      //100% ==> Full Target!
#define     TARGET_CHANGE_TIME 8  //Unit: minutes (Each TARGET_CHANGE_TIME minutes chagne the target)
#define     TARGET_CHANGE_STEP 25
#define     BW_NORM_FACTOR 100     //100 Mbps will be used to normalize throughput signal 
int shmid;
key_t key=123456;
char *shared_memory;
int shmid_rl;
key_t key_rl=12345;
char *shared_memory_rl;
int shmem_size=2048; //Shared Memory size: 2KBytes
int key1=0;
int key2=0;
//----------------------------------------------------*
typedef unsigned int u32;
typedef int s32;
typedef uint64_t  u64;
typedef int64_t  i64;
typedef uint8_t  u8;
                
bool    send_traffic=true;
u32     step_it=0;
u32     duration_steps=0;
unsigned int duration=0;                     //If not zero, it would be the total duration for sending out the traffic.


struct timeval tv_start,tv_start2;	//Start time (after three way handshake)
struct timeval tv_stamped,tv_stamped2;
uint64_t start_of_client;
struct timeval tv_end,tv_end2;		//End time
bool done=false;
bool check=false;
uint64_t setup_time;  //Flow Completion Time
int actor_id;
char *scheme;
char *ssh;
char *ssh_cmd;
char *downlink;
char *uplink;
char *log_file;
char *path;
char *congestion;
int it=0;
bool got_message=0;
int codel=0;
int first_time=0;
int min_thr=2;
int qsize=100; //pkts
int flow_index=0;
int target_ratio=150;
u32 target=50; //50ms
u32 report_period=5;//5s
double mm_loss_rate=0;

#define FLOW_NUM 1
int sock[FLOW_NUM];
int sock_for_cnt[FLOW_NUM];

struct tcp_orca_info {
    u32 min_rtt;      /* min-filtered RTT in uSec */
    u32 avg_urtt;     /* averaged RTT in uSec from the previous info request till now*/
    u32 cnt;          /* number of RTT samples uSed for averaging */
    unsigned long thr;          /*Bytes per second*/
    u32 thr_cnt;
    u32 cwnd;
    u32 pacing_rate;
    u32 lost_bytes;
    u32 srtt_us;            /* smoothed round trip time << 3 in usecs */
    u32 snd_ssthresh;       /* Slow start size threshold*/
    u32 packets_out;        /* Packets which are "in flight"*/
    u32 retrans_out;        /* Retransmitted packets out*/
    u32 max_packets_out;    /* max packets_out in last window */
    u32 mss;

    void init()
    {
        min_rtt=0;
        avg_urtt=0;
        cnt=0;
        thr=0;
        thr_cnt=0;
        cwnd=0;
        pacing_rate=0;
        lost_bytes=0;
        srtt_us=0;
        snd_ssthresh=0;
        retrans_out=0;
        max_packets_out=0;
        mss=0;
    }
    tcp_orca_info& operator =(const tcp_orca_info& a){
        this->min_rtt=a.min_rtt;
        this->avg_urtt=a.avg_urtt;
        this->cnt=a.cnt;
        this->thr=a.thr;
        this->thr_cnt=a.thr_cnt;
        this->cwnd=a.cwnd;
        this->pacing_rate=a.pacing_rate;
        this->lost_bytes=a.lost_bytes;
        this->snd_ssthresh=a.snd_ssthresh;
        this->packets_out=a.packets_out;
        this->retrans_out=a.retrans_out;
        this->max_packets_out=a.max_packets_out;
        this->mss=a.mss;
    }
}orca_info;

                   
struct tcp_sage_info {
    u8	state;
    u8	ca_state;
    u8	retransmits;
    u8	probes;
    u8	backoff;
    u8	options;
    u8	snd_wscale : 4, rcv_wscale : 4;
    u8	delivery_rate_app_limited:1;

    u32	rto;
    u32	ato;
    u32	snd_mss;
    u32	rcv_mss;

    u32	unacked;
    u32	sacked;
    u32	lost;
    u32	retrans;
    u32	fackets;

    /* Times. */
    u32	last_data_sent;
    u32	last_ack_sent;     /* Not remembered, sorry. */
    u32	last_data_recv;
    u32	last_ack_recv;

    /* Metrics. */
    u32	pmtu;
    u32	rcv_ssthresh;
    u32	rtt;
    u32	rttvar;
    u32	snd_ssthresh;
    u32	snd_cwnd;
    u32	advmss;
    u32	reordering;

    u32	rcv_rtt;
    u32	rcv_space;

    u32	total_retrans;

    u64	pacing_rate;
    u64	max_pacing_rate;
    u64	bytes_acked;    /* RFC4898 tcpEStatsAppHCThruOctetsAcked */
    u64	bytes_received; /* RFC4898 tcpEStatsAppHCThruOctetsReceived */
    u32	segs_out;	     /* RFC4898 tcpEStatsPerfSegsOut */
    u32	segs_in;	     /* RFC4898 tcpEStatsPerfSegsIn */

    u32	notsent_bytes;
    u32	min_rtt;
    u32	data_segs_in;	/* RFC4898 tcpEStatsDataSegsIn */
    u32	data_segs_out;	/* RFC4898 tcpEStatsDataSegsOut */

    u64   delivery_rate;

    u64	busy_time;      /* Time (usec) busy sending data */
    u64	rwnd_limited;   /* Time (usec) limited by receive window */
    u64	sndbuf_limited; /* Time (usec) limited by send buffer */

    u32	delivered;
    u32	delivered_ce;

    u64	bytes_sent;     /* RFC4898 tcpEStatsPerfHCDataOctetsOut */
    u64	bytes_retrans;  /* RFC4898 tcpEStatsPerfOctetsRetrans */
    u32	dsack_dups;     /* RFC4898 tcpEStatsStackDSACKDups */
    u32	reord_seen;     /* reordering events seen */



    //u32 min_rtt;      /* min-filtered RTT in uSec */
    //u32 avg_urtt;     /* averaged RTT in uSec from the previous info request till now*/
    //u32 cnt;          /* number of RTT samples uSed for averaging */
    //unsigned long thr;          /*Bytes per second*/
    //u32 thr_cnt;
    //u32 cwnd;
    //u32 pacing_rate;
    //u32 lost_bytes;
    //u32 srtt_us;            /* smoothed round trip time << 3 in usecs */
    //u32 snd_ssthresh;       /* Slow start size threshold*/
    //u32 packets_out;        /* Packets which are "in flight"*/
    //u32 retrans_out;        /* Retransmitted packets out*/
    //u32 max_packets_out;    /* max packets_out in last window */
    //u32 mss;
    /*
    void init()
    {
        min_rtt=0;
        avg_urtt=0;
        cnt=0;
        thr=0;
        thr_cnt=0;
        cwnd=0;
        pacing_rate=0;
        lost_bytes=0;
        srtt_us=0;
        snd_ssthresh=0;
        retrans_out=0;
        max_packets_out=0;
        mss=0;
    }
    tcp_sage_info& operator =(const tcp_sage_info& a){
        this->min_rtt=a.min_rtt;
        this->avg_urtt=a.avg_urtt;
        this->cnt=a.cnt;
        this->thr=a.thr;
        this->thr_cnt=a.thr_cnt;
        this->cwnd=a.cwnd;
        this->pacing_rate=a.pacing_rate;
        this->lost_bytes=a.lost_bytes;
        this->snd_ssthresh=a.snd_ssthresh;
        this->packets_out=a.packets_out;
        this->retrans_out=a.retrans_out;
        this->max_packets_out=a.max_packets_out;
        this->mss=a.mss;
    }
    */
}sage_info;

                   
struct sTrace
{
    double time;
    double bw;
    double minRtt;
};
struct sInfo
{
    sTrace *trace;
    int sock;
    int num_lines;
};
int delay_ms;
double min_base_rtt = 0.1;
double max_base_rtt = 1.6;
double base_rtt;
int client_port;
sTrace *trace;

#define DBGSERVER 0 

#define TARGET_RATIO_MIN 100
#define TARGET_RATIO_MAX 1000

#define TCP_CWND_CLAMP 42
#define TCP_CWND 43
#define TCP_ORCA_ENABLE 44
#define TCP_CWND_CAP 45
#define TCP_ORCA_INFO 46 /* Get          Congestion Control (optional) orca info */
#define TCP_CWND_MIN 47


uint64_t raw_timestamp( void )
{
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    uint64_t us = ts.tv_nsec / 1000;
    us += (uint64_t)ts.tv_sec * 1000000;
    return us;
}
uint64_t timestamp_begin(bool set)
{
        static uint64_t start;
        if(set)
            start = raw_timestamp();
        return start;
}
uint64_t timestamp_end( void )
{
        return raw_timestamp() - timestamp_begin(0);
}

uint64_t initial_timestamp( void )
{
        static uint64_t initial_value = raw_timestamp();
        return initial_value;
}

uint64_t timestamp( void )
{
        return raw_timestamp() - initial_timestamp();
}

//Start server
void start_server(int flow_num, int client_port);

//thread functions
void* DataThread(void*);
void* CntThread(void*);
void* TimerThread(void*);
void* MonitorThread(void*);

//Print usage information
void usage();

int get_orca_info(int sk, struct tcp_orca_info *info)
{
    int tcp_info_length = sizeof(*info);

    return getsockopt( sk, SOL_TCP, TCP_ORCA_INFO, (void *)info, (socklen_t *)&tcp_info_length );
};

int get_sage_info(int sk, struct tcp_sage_info *info)
{
    int tcp_info_length = sizeof(*info);

    return getsockopt( sk, SOL_TCP, TCP_INFO, (void *)&sage_info, (socklen_t *)&tcp_info_length );
};

void handler(int sig) {
    void *array[10];
    size_t size;
    DBGMARK(DBGSERVER,2,"=============================================================== Start\n");
    size = backtrace(array, 20);
    fprintf(stderr, "We got signal %d:\n", sig);
    DBGMARK(DBGSERVER,2,"=============================================================== End\n");
    shmdt(shared_memory);
    shmctl(shmid, IPC_RMID, NULL);
    shmdt(shared_memory_rl);
    shmctl(shmid_rl, IPC_RMID, NULL);
    exit(1);
}

